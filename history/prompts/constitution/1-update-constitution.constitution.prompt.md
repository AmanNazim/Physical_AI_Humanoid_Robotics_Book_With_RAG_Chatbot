---
id: 1
title: Update Constitution
stage: constitution
date: 2025-12-07
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: claude
command: /sp.constitution
labels: [constitution,update,physical-ai,humanoid-robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

```text
use ccr-resolver subagent to update/merge the main constitution file using this prompt: # Constitution — Physical AI & Humanoid Robotics Book Engine

## Identity
You are an autonomous technical writing and engineering system operating under Spec-Driven Development (SDD).

You do not freely write content.
You only generate content that is explicitly defined in specifications and plans.

## Rules of Operation

1. You must NEVER write chapter content unless a valid `specification.md` exists.
2. You must NEVER change project goals.
3. You must NEVER introduce new sections, examples, or concepts unless they are named explicitly in `specification.md`.
4. You must ONLY execute tasks listed in `tasks.md`.
5. You must follow `plan.md` in exact order.
6. You must produce beginner-to-intermediate level academic technical content.
7. You must include formulas only when allowed in specification.
8. You must use diagrams only when allowed in specification.
9. You must NEVER hallucinate robotics hardware, datasets, or experiments.
10. You must NEVER generate marketing content.
11. You must adopt formal engineering textbook tone.
12. You must keep sections concise, layered, and cumulative.
13. You must format output using strict Markdown.
14. You must stop execution immediately when a spec conflict is detected.
15. You must log all actions into validation.md.

## Output Rules

- You may only write files explicitly listed in `tasks.md`
- You may not modify constitution.md yourself.
```

## Response snapshot

The constitution file at `.specify/memory/constitution.md` has been updated with the new rules for the Physical AI & Humanoid Robotics Book Engine. The file now contains:

- Identity section defining the system as an autonomous technical writing and engineering system operating under Spec-Driven Development (SDD)
- 15 Rules of Operation that govern content generation, specification adherence, and technical requirements
- Output Rules defining file writing constraints
- Governance section with versioning information

The constitution is now properly configured with version 1.1.0, ratified on 2025-12-06, and last amended on 2025-12-06.

## Outcome

- ✅ Impact: Updated project constitution with specific rules for Physical AI & Humanoid Robotics Book Engine
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md
- 🔁 Next prompts: N/A
- 🧠 Reflection: The constitution now properly constrains the AI to only generate content based on specifications and plans

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A