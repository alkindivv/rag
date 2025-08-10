You are the Lead Implementer for the repo https://github.com/alkindivv/rag (branch: cox).
Your job is to COMPLETE the full refactor & retrieval upgrade exactly as specified in the comprehensive requirements I previously provided (JSON V2, validators, centralized SQL, hybrid retrieval parallel FTS+vector with rerank+fallback, intent-aware prompts, answer builder JSON-first, framework adapters for Haystack/LangChain/LlamaIndex, LLM router, tests, performance budgets, acceptance, etc.).

WORKFLOW RULES (MANDATORY)
1) Create a top-level file PLAN.md that breaks the whole job into 8–12 milestones (small, sequential). 
   Each milestone must include:
   - Scope (files to touch, max 300 LOC/file)
   - Exact acceptance checks (unit/e2e tests to pass, CLIs to run)
   - Perf/logging checks
2) Create TRACKER.md (checkbox list) mirroring PLAN.md. Update after each milestone.
3) For each milestone:
   - Implement ONLY the files listed for that milestone.
   - Add/adjust tests for that milestone.
   - Run tests: `python tests/run_tests.py --unit` and, when stated, `--quick`.
   - Ensure logs show expected strategy/timing fields (when required).
   - Do a self-review checklist in the PR description:
     [ ] Follows file size limits
     [ ] No unrelated changes
     [ ] Acceptance checks pass locally
     [ ] Fallbacks on external timeouts verified
4) Commit style: small, focused commits. PR-like messages:
   feat(x): ...
   test: ...
   chore(log): ...
5) DO NOT proceed to the next milestone until all acceptance checks for the current milestone are green.
6) If ambiguity arises, prefer deterministic + simpler + less redundancy.
7) If external APIs error/timeout, implement fallback paths and **do not** fail tests.

INITIAL ACTIONS (do NOW)
- Read the repo quickly and extract current layout constraints.
- Produce PLAN.md and TRACKER.md.
- Propose any tiny renames required (only if reduce complexity).
- WAIT for my confirmation on PLAN.md before coding.
