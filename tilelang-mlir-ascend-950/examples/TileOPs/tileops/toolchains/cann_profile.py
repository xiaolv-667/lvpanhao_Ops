"""Native CANN profile decoding shared by the CANN toolchain providers."""

import csv
import glob
import logging
import os

_logger = logging.getLogger(__name__)


def benchmark_profile_result(output_dir: str) -> tuple[list[float], str, list[str]]:
    """Return durations, the first op name, and all captured op names.

    The msprof output directory has the structure::

        <output_dir>/
          └── OPPROF_<timestamp>_<random>/
              └── <OpName>/
                  ├── 0/OpBasicInfo_<ts>.csv
                  ├── 1/OpBasicInfo_<ts>.csv
                  ...
                  └── N/OpBasicInfo_<ts>.csv

    Each CSV has one row with a ``Task Duration(us)`` column.
    """
    csv_files = sorted(
        glob.glob(
            os.path.join(output_dir, "**", "OpBasicInfo*.csv"),
            recursive=True,
        )
    )
    if not csv_files:
        raise FileNotFoundError(f"No OpBasicInfo*.csv found under {output_dir}")

    durations: list[float] = []
    op_name = ""
    op_names: set[str] = set()
    for csv_path in csv_files:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                val = row.get("Task Duration(us)")
                if val is not None and val.strip():
                    try:
                        durations.append(float(val))
                    except ValueError:
                        _logger.warning(
                            "Non-numeric Task Duration(us) in %s: %r",
                            csv_path,
                            val,
                        )
                name_val = row.get("Op Name", "")
                if name_val and name_val.strip():
                    captured_name = name_val.strip()
                    op_names.add(captured_name)
                    if not op_name:
                        op_name = captured_name

    if not durations:
        raise ValueError(
            f"No Task Duration(us) values found in OpBasicInfo CSVs under {output_dir}"
        )

    return durations, op_name, sorted(op_names)
