"""Compose the self-contained deliverable from the winning branch + the base parts.

The winning branch (``MoEPermuteAlignFwdOp_r6.py``) imports the constants,
helpers and the untouched fused small-batch factory from ``_base_module.py``.
This script inlines those so ``MoEPermuteAlignFwdOp.py`` stands alone (the file a
conductor copies out of the task directory), and adds the combined-pipeline
precision harness.

    python _assemble_deliverable.py <branch.py>
"""

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
BASE = (HERE / "_base_module.py").read_text()


def section(text, start_pat, end_pat):
    start = text.index(start_pat)
    end = text.index(end_pat, start)
    return text[start:end]


def main():
    branch = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                          else HERE / "MoEPermuteAlignFwdOp_r6.py").read_text()

    # branch body: drop the sibling import and the sys.path bootstrap
    branch = re.sub(r"sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\n", "", branch)
    branch = re.sub(r"from _base_module import \([^)]*\)\n", "", branch)

    helpers = section(BASE, "FILL_TILE = 2048", "_SCRATCH = {}")
    small = section(BASE, "# ---------------------------------------------------------------------------\n"
                          "# Fused small-batch stage", "# ---------------------------------------------------------------------------\n"
                          "# Combined-pipeline precision harness")
    harness = (HERE / "_harness.part.py").read_text()

    head_end = branch.index("\n_FILL_TARGET_ENTRIES = ") + 1
    head = branch[:head_end]
    # the branch header only carries the imports its kernels need; the harness
    # part needs the base module's remaining stdlib imports
    if "import argparse" not in head:
        head = head.replace("import functools\n", "import argparse\nimport functools\n", 1)
    if "import time" not in head:
        head = head.replace("import sys\n", "import sys\nimport time\n", 1)
    body = branch[head_end:]
    (HERE / "MoEPermuteAlignFwdOp.py").write_text(
        head + helpers + "\n" + body.rstrip("\n") + "\n\n" + small + harness
    )
    text = (HERE / "MoEPermuteAlignFwdOp.py").read_text()
    print(f"wrote MoEPermuteAlignFwdOp.py ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
