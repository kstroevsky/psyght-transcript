# AGENTS.md

## Read first

- Read `PROJECT_BRIEF.md` before making changes.
- For complex or ambiguous work, produce a short plan first.
- Stay inside the smallest relevant module/package and keep context narrow.

## Project rules

- Preserve strong modularity. Prefer feature/domain modules with explicit boundaries.
- Keep business logic close to its module. Avoid cross-module coupling.
- Do not introduce new services, global abstractions, or large refactors unless clearly justified by the task.
- Reuse existing patterns before inventing new ones.
- Keep public APIs explicit. Do not import another module's internals.

## Editing rules

- Change only what is needed for the task.
- Prefer small, reviewable diffs over broad rewrites.
- Keep names explicit and domain-meaningful.
- Reduce duplication, but do not create premature abstractions.
- Put detailed architecture/process docs outside this file and link to them.

## Verification

- Add or update focused tests for behavior changes.
- Run the smallest relevant checks first, then broader required checks.
- Do not mark work done until relevant lint/typecheck/tests pass, or clearly report what could not be verified.

## Docs

- If behavior, contracts, or architecture change, update the nearest relevant docs.
- After each completed task, update `PROJECT_BRIEF.md`:
  - current status
  - decisions made
  - new constraints/invariants
  - next recommended step
- Keep `PROJECT_BRIEF.md` short and factual.

## Output format

- Summarize:
  - what changed
  - files changed
  - checks run and results
  - risks / open questions / follow-ups

## Anklav task-control protocol

- Start from the active Anklav task and load its Anklav context pack.
- Read the Git-backed canonical artifacts before interpreting task context.
- Use the Anklav task identifier in branches, commits, and pull requests.
- Send progress, evidence, and unfinished-work handoffs to Anklav.
- Create out-of-scope discoveries in Anklav Inbox.
- Never treat retrieved session text as canonical without verification.

## Instruction hygiene

- Keep this file short.
- Put module-specific rules in nested `AGENTS.md` files near that code.
- Put deep task playbooks in `docs/` or skills, not here.
