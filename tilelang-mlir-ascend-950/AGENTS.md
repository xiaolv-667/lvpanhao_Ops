# TileLang Agent Guide

## Task platform

Resolve hardware and toolchain through `python3 .agents/tools/statectl.py platform --hardware <model> --toolchain <toolchain>`. For an existing task use `--task-dir <task>` and retain its saved selection. Read the returned `ruleset` before applying API, programming-mode, memory, synchronization or runtime requirements.

The common stage skills define the workflow and artifacts. The selected platform supplies the implementation rules, API evidence, examples, knowledge and kernel template. Validate APIs in that toolchain's language module and build; evidence from a different backend does not establish availability.

## Operator implementation baseline

- Start from the selected platform's kernel template and compatible examples/tests.
- Read the source operator for migrations and preserve its mathematical semantics and input/output contract.
- Confirm API names and parameters against the selected toolchain's documented or installed implementation.
- Use that platform's memory hierarchy, limits and synchronization rules.
- State explicitly when no compatible example exists before constructing a minimal implementation.

## Execution context

Every stage reads the task platform's `execution` context before running a kernel or prototype, including design-time probes. Follow its command and working-directory requirements. Static API inspection without device computation can use the platform's API-inspection instructions.

The conductor forwards the complete execution context to child stages, together with the task directory and platform rules. A child missing that context reads it through `statectl platform --task-dir <task>` before running device code.

## Stage skills

Use the platform's `stage_skills` for design, review, development and optimization; use `diagnostics` for error classification and debugging. Backend-specific domain skills and documentation are selected by the platform rules.

## Compiler capability gaps

When a stage concludes that the selected toolchain cannot implement a required capability:

1. Identify the missing capability and its compiler/runtime layer.
2. Record or deduplicate the finding in the platform's `knowledge.capability_gaps` entry, following its schema.
3. Cite applicable documentation or measured evidence; label an unverified assumption explicitly.
4. A gap identified independently in two or more tasks is recurring and must produce a capability-fill proposal for the user. External issue publication requires user approval.
5. Bind the conclusion to the toolchain version; re-verify after changes using the existing knowledge tools with the task directory.

## Pre-PR formatting

Before creating or updating a PR, run `bash format.sh --files changed_files` from the repository root.

## Documentation and evidence

Follow the selected platform's documentation routing and source priority. Algorithm references from other backends may inform mathematical reasoning, while API availability, layouts, synchronization and measured costs require evidence for the task's selected combination.

For conflicting API evidence, verify the actual selected language module, its installed/source location and version. Record an unavailable or undocumented capability with the evidence and applicable boundary rather than substituting another backend's API rules.
