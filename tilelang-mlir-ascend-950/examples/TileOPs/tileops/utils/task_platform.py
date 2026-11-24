"""Read the selected Agent task when TileOPs runs inside a task."""

import os
import sys
from pathlib import Path


def get_task_platform():
    context_module = sys.modules.get("platform_context")
    if context_module is not None:
        return context_module.get_platform(required=False)
    if not (os.environ.get("TILEOPS_TASK_DIR") or os.environ.get("TILELANG_TASK_PLATFORM")):
        return None
    tools = Path(__file__).resolve().parents[4] / ".agents/tools"
    sys.path.insert(0, str(tools))
    from platform_context import get_platform

    return get_platform(required=False)


def shared_memory_budget(device=None):
    context = get_task_platform()
    if context is None:
        from tileops.device import get_device_backend

        return get_device_backend().shared_memory_budget(device)
    if context.hardware is not None:
        budget = context.design_budget()
        return budget["capacity_bytes"][budget["inflation_level"]]
    return context.adapter.shared_memory_budget(device)
