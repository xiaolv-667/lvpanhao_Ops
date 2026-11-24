"""Acceptance metrics for TileOPs reports: MFU / MU / latency.

The legacy report headline was ``roofline_utilization_percent`` (achieved over a
model bound).  Acceptance instead asks, per operator class:

* compute-bound **CV**      -> MFU  = FLOPs / (T_op * Peak_TFLOPS)   target 80%
* sparse CV                 -> sparse MFU                            target 50%
* memory-bound (CV or Vec)  -> MU   = Bytes / (T_op * Peak_BW)       target 90%
* latency-bound small ops   -> T_op / T_baseline                     target <= 1.0

``T_op`` is the whole-operator time: the sum of every device kernel launched by
one operator call (including torch-side helpers such as ``ZerosLike``/``Range``;
host-padding ops such as ``PadV3`` are excluded).

The numerator always comes from the manifest ``roofline`` expressions, never
from the profiler's own per-kernel ``Perf(TOps/s)`` counter.

Peaks are read from ``peaks.json`` next to this module; they carry a provenance
block.  Until 950PR is calibrated they are marked ``provisional``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _product(shape) -> int:
    value = 1
    for item in shape:
        value *= int(item)
    return value


_SAFE = {
    "__builtins__": {},
    "product": _product,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "abs": abs,
    "list": list,
    "zip": zip,
    "range": range,
}

_DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "int32": 4,
    "int64": 8,
}

_HOST_PAD = re.compile(r"^Pad", re.I)

DEFAULT_PEAKS = {
    # Primary denominators: vendor SKU table (auditable, conservative).
    "compute_tflops": 378.0,
    "cube_tflops": {
        "bfloat16": 378.0, "float16": 378.0, "float32": 189.0,
        "float8_e4m3fn": 756.0, "int8": 756.0,
    },
    "bandwidth_tbs": 1.6,
    # Supplementary: measured on this host (reported next to the primary value).
    "compute_tflops_measured": 370.5,
    "cube_tflops_measured": {
        "bfloat16": 370.5, "float16": 370.5, "float32": 185.2,
        "float8_e4m3fn": 741.0, "int8": 741.0,
    },
    "bandwidth_tbs_measured": 1.424,
    "vector_tflops": {"bfloat16": 47.0, "float16": 47.0, "float32": 23.0, "int32": 11.0},
    "sparse_tflops": None,
    "latency_floor_us": 50.0,
    # Acceptance policy: MFU is the single metric; 90% is the "others" tier of
    # the requirement (sparse CV 50% / CV 80% / others 90%).
    "acceptance": {"metric": "MFU", "target_percent": 90.0},
    "provenance": {"status": "vendor_spec_plus_measured"},
}


def peak_for_dtype(peaks: dict[str, Any], dtype: Any, *, measured: bool = False) -> float:
    """Cube peak for one dtype.

    Primary (``measured=False``) is the vendor SKU table; the measured value is
    the supplementary column used for the ``value_measured`` field.
    """
    table = peaks.get("cube_tflops_measured" if measured else "cube_tflops") or {}
    value = table.get(str(dtype))
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    key = "compute_tflops_measured" if measured else "compute_tflops"
    return float(peaks.get(key) or 0.0)


def load_compute_units(path: str | Path | None = None) -> dict[str, Any]:
    """Load the operator -> compute-unit map measured from msprof AIC/AIV ratios."""
    candidate = Path(path) if path else Path(__file__).with_name("compute_units.json")
    try:
        data = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, dict)}


def peak_for_unit(
    peaks: dict[str, Any], dtype: Any, unit: str = "cube", *, measured: bool = False
) -> float | None:
    """Peak for the compute unit a kernel actually runs on.

    Cube kernels use the Cube table; Vector kernels use ``vector_tflops``.
    There is no measured Vector peak yet, so ``measured=True`` returns ``None``
    for Vector instead of pretending the vendor number is a calibration.
    """
    if str(unit).lower() == "vector":
        if measured:
            return None
        table = peaks.get("vector_tflops") or {}
        value = table.get(str(dtype))
        return float(value) if isinstance(value, (int, float)) and value > 0 else None
    return peak_for_dtype(peaks, dtype, measured=measured)


def load_peaks(path: str | Path | None = None) -> dict[str, Any]:
    """Load calibrated peaks; fall back to the documented provisional values."""
    peaks = dict(DEFAULT_PEAKS)
    candidate = Path(path) if path else Path(__file__).with_name("peaks.json")
    try:
        data = json.loads(candidate.read_text())
    except (OSError, json.JSONDecodeError):
        return peaks
    for key, value in (data.get("peaks") or {}).items():
        if key in peaks:
            peaks[key] = value
    if isinstance(data.get("acceptance"), dict):
        peaks["acceptance"] = {**peaks.get("acceptance", {}), **data["acceptance"]}
    peaks["provenance"] = data.get("provenance") or peaks["provenance"]
    return peaks


def _normalize(expr: str, params: dict[str, Any]) -> str:
    """Rewrite the manifest's pseudo-calls into evaluable Python.

    The manifest is a spec, not code: it uses ``product(shape)``, phrases like
    ``element size of T`` and predicates like ``slot_mapping entries >= 0``.
    ``product`` is provided by the safe namespace; this function handles the
    phrases.  Upper bounds are used when the exact value needs host data
    (e.g. ``slot_mapping entries >= 0`` -> every token written, which is the
    same upper bound the Op's own ``eval_roofline`` documents before the first
    forward).
    """
    dtype = str(params.get("dtype") or "bfloat16")
    cache_dtype = str(params.get("cache_dtype") or dtype)
    dtype_vars = {
        "T": _DTYPE_BYTES.get(dtype, 2),
        "KV": _DTYPE_BYTES.get(cache_dtype, 2),
        "int32": 4,
        "int64": 8,
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
    }

    def _element_size(match: re.Match) -> str:
        name = match.group(1)
        return str(dtype_vars.get(name, _DTYPE_BYTES.get(name, 2)))

    expr = re.sub(r"element size of ([A-Za-z_][A-Za-z0-9_]*)", _element_size, expr)
    tokens = params.get("tokens") or params.get("batch")
    if "slot_mapping entries >= 0" in expr:
        expr = expr.replace("slot_mapping entries >= 0", str(tokens or 0))
    return expr


def _eval_value(expr: str | None, env: dict[str, Any]) -> Any:
    if not expr:
        return None
    try:
        return eval(expr, _SAFE, env)  # noqa: S307 - manifest expressions
    except Exception:
        return None


def _eval(expr: str | None, env: dict[str, Any]) -> float | None:
    value = _eval_value(expr, env)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def evaluate_roofline(entry: dict[str, Any], params: dict[str, Any]) -> tuple[float | None, float | None]:
    """Evaluate the manifest ``roofline`` block for one case's params.

    Returns ``(flops, bytes)`` in absolute units; either may be ``None`` when the
    expression cannot be evaluated.
    """
    roof = (entry or {}).get("roofline") or {}
    env: dict[str, Any] = dict(params or {})

    aliases = {
        "B": params.get("batch"),
        "S_q": params.get("seqlen_q"),
        "S_kv": params.get("seqlen_kv"),
        "D_qk": params.get("dim_qk"),
        "D_v": params.get("dim_v"),
        "M": params.get("M"),
        "seq_len": params.get("seq_len"),
        "d": params.get("d"),
        "N": params.get("N"),
        "G": params.get("G"),
        "NC": params.get("NC"),
        "chunk_len": params.get("chunk_len"),
    }
    for key, value in aliases.items():
        if value is not None:
            env.setdefault(key, value)
    if params.get("heads") is not None:
        env["H"] = params["heads"]
    if params.get("hidden_states_shape"):
        env.setdefault("T", params["hidden_states_shape"][0])
        env.setdefault("H", params["hidden_states_shape"][1])
    if params.get("w_down_shape"):
        env.setdefault("S", params["w_down_shape"][1])

    # family-specific shape aliases
    batch = params.get("batch")
    if batch is not None:
        if "num_draft" in params and "vocab" in params:
            nd, vocab = params["num_draft"], params["vocab"]
            env.setdefault("draft_probs_shape", [batch, nd, vocab])
            env.setdefault("target_probs_shape", [batch, nd + 1, vocab])
            env.setdefault("draft_token_ids_shape", [batch, nd])
        if all(k in params for k in ("chunk_len", "d_state", "n_groups")):
            nc = params.get("num_chunks")
            cl = params.get("chunk_len")
            seq = params.get("seq_len") or (nc * cl if nc and cl else None)
            env.setdefault("C_shape", [batch, seq, params.get("n_groups"), params.get("d_state")])
            env.setdefault("chunk_size", cl)
        if "d_mem" in params and "d" in params:
            dmem, dd = params["d_mem"], params["d"]
            length = params.get("conv_len", params.get("L"))
            mcl = params.get("max_conv_len", length)
            env.setdefault("e_t_shape", [batch, dmem])
            env.setdefault("h_t_shape", [batch, dd])
            env.setdefault("conv_state_shape", [batch, length, dd])
            env.setdefault("new_conv_state_shape", [batch, mcl, dd])
    if all(k in params for k in ("M", "seq_len", "d")):
        shape = [params["M"], params["seq_len"], params["d"]]
        taps = params.get("conv_kernel_size", 4)
        env.setdefault("H_shape", shape)
        env.setdefault("dY_shape", shape)
        env.setdefault("conv_w_shape", [taps, params["d"]])
    for key, value in list(params.items()):
        if key.endswith("_shape"):
            env.setdefault(key, value)

    dtype = params.get("dtype") or params.get("cache_dtype") or "bfloat16"
    env.setdefault("elem_bytes", _DTYPE_BYTES.get(str(dtype), 2))
    # dtype names used in manifest comparisons (cache_dtype == float8_e4m3fn)
    for name in ("float8_e4m3fn", "float16", "bfloat16", "float32", "int32", "int64"):
        env.setdefault(name, name)

    # variables the manifest expects but does not define for every family
    env.setdefault("rotary_dim", params.get("rotary_dim") or params.get("head_size") or 0)
    env.setdefault("rows_read", params.get("tokens") or params.get("batch") or 0)

    for _ in range(4):
        for name, expr in (roof.get("vars") or {}).items():
            value = _eval_value(_normalize(str(expr), params), env)
            if value is not None:
                env[name] = value

    flops = _eval(_normalize(str(roof.get("flops") or "0"), params), env)
    nbytes = _eval(_normalize(str(roof.get("bytes") or ""), params), env)
    return flops, nbytes


def classify(flops: float | None, nbytes: float | None, t_op_us: float | None,
             peaks: dict[str, Any]) -> str:
    """Return ``compute`` / ``memory`` / ``latency`` / ``unknown``."""
    if not t_op_us or t_op_us <= 0:
        return "unknown"
    floor = float(peaks.get("latency_floor_us") or 0.0)
    if (flops is None or flops <= 0) and (not nbytes or nbytes < 1 * 1024 * 1024):
        return "latency"
    if flops is None or flops <= 0:
        return "memory" if nbytes else "unknown"
    if not flops or not nbytes:
        return "memory" if nbytes else "unknown"
    ai = flops / nbytes
    peak_tflops = float(peaks.get("compute_tflops") or 0.0)
    peak_bw = float(peaks.get("bandwidth_tbs") or 0.0)
    balance = (peak_tflops / peak_bw) if peak_bw else float("inf")
    if ai >= balance:
        return "compute"
    if t_op_us <= floor and flops < 1e6:
        return "latency"
    return "memory"


def compute_case(entry: dict[str, Any], params: dict[str, Any], t_op_us: float | None,
                 *, peaks: dict[str, Any] | None = None,
                 reported_us: float | None = None,
                 unit: str | None = None,
                 dtype: str | None = None) -> dict[str, Any]:
    """Compute the acceptance record for one benchmark case.

    **MFU is the single acceptance metric**: ``MFU = FLOPs / (T_op * peak)``.
    The denominator is the peak of the unit the kernel actually runs on
    (``unit="cube"`` or ``"vector"``); using the Cube peak for a pure Vector
    kernel understates its MFU by ~8x.  ``value`` uses the theoretical (vendor)
    peak and ``value_measured`` the measured one where a calibration exists.
    MU is retained only as a diagnostic (``mu_value``) and never decides
    pass/fail.  For a memory-bound shape the MFU ceiling ``AI / balance`` is
    recorded so a low MFU is not mistaken for a bad kernel.

    A throughput above the theoretical peak is physically impossible, so the case
    is marked invalid instead of being published as a (fake) efficiency number.
    """
    peaks = peaks or load_peaks()
    # Manifest workloads sometimes omit dtype (e.g. ChainSpec is float32 by
    # construction); the unit map supplies the default, but explicit params win.
    dtype = params.get("dtype") or params.get("cache_dtype") or dtype
    if dtype and not (params.get("dtype") or params.get("cache_dtype")):
        eval_params = {**params, "dtype": dtype}
    else:
        eval_params = params
    flops, nbytes = evaluate_roofline(entry, eval_params)
    klass = classify(flops, nbytes, t_op_us, peaks)
    unit = (unit or "cube").lower()
    peak_t = peak_for_unit(peaks, dtype, unit)
    peak_t_meas = peak_for_unit(peaks, dtype, unit, measured=True)
    if peak_t is None:  # unknown dtype for this unit -> fall back to the Cube table
        fallback = "vector" if unit != "vector" else "cube"
        peak_t = peak_for_unit(peaks, dtype, fallback)
        peak_t_meas = peak_for_unit(peaks, dtype, fallback, measured=True)
        unit = fallback
    peak_bw = float(peaks.get("bandwidth_tbs") or 0.0)
    peak_bw_meas = float(peaks.get("bandwidth_tbs_measured") or peak_bw)

    record: dict[str, Any] = {
        "class": klass,
        "compute_unit": unit,
        "status": "measured",
        "t_op_us": t_op_us,
        "roofline_flops": flops,
        "roofline_bytes": nbytes,
        "arithmetic_intensity": (flops / nbytes) if flops and nbytes else None,
        "metric": None,
        "value": None,
        "value_measured": None,
        "target_percent": None,
        "pass": None,
        "peak_used": {
            "unit": unit,
            "compute_tflops": peak_t,
            "compute_tflops_measured": peak_t_meas,
            "bandwidth_tbs": peak_bw,
            "bandwidth_tbs_measured": peak_bw_meas,
            "dtype": dtype,
        },
        "notes": [],
    }
    if flops and t_op_us:
        record["achieved_tflops"] = flops / (t_op_us * 1e-6) / 1e12
    if nbytes and t_op_us:
        record["achieved_tbps"] = nbytes / (t_op_us * 1e-6) / 1e12

    balance = (peak_t / peak_bw) if peak_bw else None
    target = float((peaks.get("acceptance") or {}).get("target_percent", 90.0))
    if record.get("achieved_tflops") is not None:
        # MFU is the requested metric for every operator that has a FLOP count.
        record["metric"] = "MFU"
        record["target_percent"] = target
        if peak_t:
            record["value"] = 100.0 * record["achieved_tflops"] / peak_t
            record["pass"] = record["value"] >= target
        if peak_t_meas:
            record["value_measured"] = 100.0 * record["achieved_tflops"] / peak_t_meas
        if record["arithmetic_intensity"] and balance:
            ceiling = 100.0 * min(1.0, record["arithmetic_intensity"] / balance)
            record["mfu_ceiling_percent"] = ceiling
            if klass == "memory":
                record["notes"].append(
                    "memory-bound: MFU is capped at AI/balance = "
                    f"{ceiling:.2f}% by HBM bandwidth"
                )
    elif klass == "latency":
        record["metric"] = "latency"
        record["value"] = t_op_us
        record["notes"].append(
            "latency-bound (FLOPs=0): MFU is undefined; compare against a baseline"
        )

    # MU is diagnostic only -- never used for pass/fail.
    if record.get("achieved_tbps") is not None and peak_bw:
        record["mu_value"] = 100.0 * record["achieved_tbps"] / peak_bw
        if peak_bw_meas:
            record["mu_value_measured"] = 100.0 * record["achieved_tbps"] / peak_bw_meas

    impossible = False
    if record.get("achieved_tflops") and peak_t and record["achieved_tflops"] > peak_t * 1.01:
        record["notes"].append("physically-impossible: achieved compute exceeds the peak")
        impossible = True
    if record.get("achieved_tbps") and peak_bw and record["achieved_tbps"] > peak_bw * 1.01:
        record["notes"].append("physically-impossible: achieved bandwidth exceeds the peak")
        impossible = True
    if impossible:
        record["status"] = "invalid"
        record["pass"] = False
    if reported_us and t_op_us and reported_us > 0 and t_op_us > 2.0 * reported_us:
        record["notes"].append("report-underreports: use t_op_us")
    record["peaks_provenance"] = peaks.get("provenance")
    return record


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case acceptance records by operator.

    MFU% and MU% live on different denominators and must never be compared
    with each other, so the summary keeps one bucket per metric.  The
    headline ``metric``/``best`` fields point at the metric covering the most
    cases (MFU wins ties), which is what the operator-level table shows.
    """
    by_op: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        acc = case.get("acceptance") or {}
        if acc:
            by_op.setdefault(case.get("operator"), []).append(acc)

    summary: dict[str, Any] = {}
    for operator, records in by_op.items():
        per_metric: dict[str, dict[str, Any]] = {}
        for record in records:
            # Physically impossible throughputs are capture/launch defects, not
            # efficiency: never let them set an operator's best value.
            if record.get("status") == "invalid":
                continue
            metric = record.get("metric")
            if not metric:
                continue
            bucket = per_metric.setdefault(
                metric, {"cases": 0, "best": None, "worst": None, "passed": 0, "failed": 0}
            )
            bucket["cases"] += 1
            value = record.get("value")
            if isinstance(value, (int, float)):
                bucket["best"] = value if bucket["best"] is None else max(bucket["best"], value)
                bucket["worst"] = value if bucket["worst"] is None else min(bucket["worst"], value)
            if record.get("pass") is True:
                bucket["passed"] += 1
            elif record.get("pass") is False:
                bucket["failed"] += 1
        headline = (
            max(per_metric, key=lambda m: (per_metric[m]["cases"], m == "MFU"))
            if per_metric
            else None
        )
        lead = per_metric.get(headline) or {}
        summary[operator] = {
            "classes": sorted({r.get("class") for r in records if r.get("class")}),
            "cases": len(records),
            "metric": headline,
            "best": lead.get("best"),
            "worst": lead.get("worst"),
            "passed": sum(1 for r in records if r.get("pass") is True),
            "failed": sum(1 for r in records if r.get("pass") is False),
            "invalid": sum(1 for r in records if r.get("status") == "invalid"),
            "blocked": sum(1 for r in records if r.get("status") == "blocked"),
            "metrics": per_metric,
            "blocked_notes": sorted({n for r in records for n in r.get("notes") or []}),
        }
    return summary
