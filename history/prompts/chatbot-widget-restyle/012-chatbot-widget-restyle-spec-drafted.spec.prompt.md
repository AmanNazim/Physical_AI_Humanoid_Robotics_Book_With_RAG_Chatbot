---
id: 012
title: Chatbot Widget Restyle — Spec, Plan, Tasks Drafted
stage: spec
date: 2026-09-15
surface: agent
model: glm-5.3
feature: chatbot-widget-restyle
branch: main
user: Aman Nazim
command: none
labels: [chatkit, spec, plan, tasks, approval-gate]
links:
  spec: physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/specification.md
  - physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/plan.md
  - physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/tasks.md
tests:
  - none (spec stage; implementation gated on user approval)
---

## Prompt

it is okay, now the chat button in corner is green also the UI and UX of chatbot interface should we keep it like that or improve if yes draft a plan and work with spec, plan and task methods before implementation ask for approval and must integrate specs folder in existing workflow not in new folders.

## Response snapshot

Investigated the green chat button root cause before drafting:

- T012 had remapped `--chatkit-*` variables in `rag_chatbot/chatkit/styles/variables.css` correctly (teal light + dark blocks).
- **But zero component stylesheets consume those variables** — `var(--chatkit-*)` appears 0 times across all 9 component CSS files. Every component hardcodes its colors (~59 hex occurrences), including launcher green `#00C26A` (ChatLauncherButton.css:10) and a hardcoded-white ChatPanel that never enters dark mode.
- Mounting path: site renders ChatKit via PortalManager into `#chatkit-portal-root` (style.all='initial'); custom properties still inherit through that reset, so the variable layer DOES reach the widget — components just never read it.

Recommended improvement (user asked keep-vs-improve): improve — green clashes with the teal Lab Bench system and the panel breaks dark mode.

Drafted the full SDD set in the EXISTING specs workflow (no new folders beyond the per-feature convention):
- `specs/chatbot-widget-restyle/specification.md` — purpose, root-cause findings, in/out of scope (CSS-only migration, behavior unchanged), design concept, 4 user stories with acceptance criteria, constraints.
- `specs/chatbot-widget-restyle/plan.md` — 5 phases (audit → launcher/panel chrome → conversation surface → input/tooltip → validation), key decisions (CSS variables over component rewrite; attribute theming, no JS listener; new `--chatkit-font-*` variables because the portal reset blocks site font vars; keep class-based theme.css for the standalone app), risks (standalone app regression, portal reset surprises, specificity fights).
- `specs/chatbot-widget-restyle/tasks.md` — 15 tasks in 5 groups, each with grep/build/inspection acceptance.

Awaiting user approval before implementation, per their instruction.

## Outcome

- ✅ Impact: SDD artifacts ready; implementation gated on user approval.
- 🧪 Tests: none (spec stage)
- 📁 Files: 3 spec files
- 🔁 Next prompts: user approves → implement T001–T015 phase by phase with conventional commits; user rejects → revise per feedback
- 🧠 Reflection: The T012 gap (variables mapped but never consumed) is a lesson: token work is not done when variables are defined — it's done when components read them. Phase 1 of the new plan front-loads the inheritance verification to avoid a second blind spot.

## Evaluation notes (flywheel)

- Failure modes observed: prior feature's acceptance criteria checked the variables file, not the rendered widget — acceptance must verify the consumer side, not the producer side
- Graders run and results (PASS/FAIL): n/a (spec stage)
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): user approval, then Phase 1 portal-inheritance verification before any mass migration
