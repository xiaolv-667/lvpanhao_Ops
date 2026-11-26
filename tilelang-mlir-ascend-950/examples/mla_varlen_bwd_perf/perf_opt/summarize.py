"""Print the collected capture summaries (T_op, us/iteration, pipe busy) as a table."""

import json
import sys
from pathlib import Path


def row(path):
    payload = json.loads(Path(path).read_text())
    cube = payload.get("pipe", {}).get("cube", {})
    vect = payload.get("pipe", {}).get("vector", {})
    return (
        f"{payload['case']:34s} {payload.get('impl', '?'):28s} "
        f"T_op {payload['task_duration_us']:9.1f} "
        f"main {payload.get('main_us', 0):9.1f} "
        f"it {payload.get('iterations', 0):9d} "
        f"us/it {payload.get('us_per_iteration', 0):6.3f} "
        f"blocks {payload.get('blocks', 0):6d} it/blk {payload.get('iters_per_block', 0):6.1f} "
        f"n {payload.get('captured_samples', 0):2d}{'ok' if payload.get('window_ok') else 'BAD'} "
        f"| cube% mad {100 * cube.get('aic_cube_share', 0):5.1f} "
        f"mte1 {100 * cube.get('aic_mte1_share', 0):5.1f} "
        f"mte2 {100 * cube.get('aic_mte2_share', 0):5.1f} "
        f"fix {100 * cube.get('aic_fixpipe_share', 0):5.1f} "
        f"scal {100 * cube.get('aic_scalar_share', 0):5.1f} "
        f"bw {cube.get('aic_mte2_active_bw(GB/s)', 0):5.1f}/{cube.get('aic_mte1_active_bw(GB/s)', 0):5.1f} "
        f"| aiv% vec {100 * vect.get('aiv_vec_share', 0):5.1f} "
        f"mte2 {100 * vect.get('aiv_mte2_share', 0):5.1f} "
        f"mte3 {100 * vect.get('aiv_mte3_share', 0):5.1f}"
    )


for path in sys.argv[1:]:
    print(row(path))
