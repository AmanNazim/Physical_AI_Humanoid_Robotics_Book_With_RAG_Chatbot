---
id: 006
title: Configure Design MCPs and Skill
stage: misc
date: 2026-09-14
surface: agent
model: glm-5.3
feature: ui-ux-redesign
branch: main
user: Aman Nazim
command: none
labels: [mcp, skills, configuration, design, logo]
links:
  spec: physical-ai-humanoid-robotics-book/specs/ui-ux-redesign/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - .mcp.json
  - .claude/skills/lab-bench-design/SKILL.md
  - history/prompts/ui-ux-redesign/006-configure-design-mcps.misc.prompt.md
tests:
  - claude mcp list (after restart) — servers should connect; canva/figma need OAuth on first use
---

## Prompt

I want to configure skills and mcp so you can build a better one human like.

(On rejecting the initial Playwright-only add, the user added:) I think we should also add some designer apps mcps like canva and others from which we can make better personalized logo

## Response snapshot

Configured design tooling for the project:

- **.mcp.json** (project scope): `canva` (official remote HTTP MCP, https://mcp.canva.com/mcp) for personalized logo/brand design creation; `figma` (official remote HTTP MCP, https://mcp.figma.com/mcp) for reading Figma designs; `playwright` (local stdio, npx @playwright/mcp@latest) for browser screenshots and dual-mode visual verification — closing the gap from the redesign session where designs could not be visually verified.
- **New project skill** `.claude/skills/lab-bench-design/SKILL.md`: captures the Lab Bench design system (palette, typography, Signal Path signature, Signal Bot badge logo direction, copy voice, quality floor) so all future design work stays consistent and human.
- Noted: MCP servers load on session restart; canva/figma require OAuth login on first use; remote servers send brief context to those services.

## Outcome

- ✅ Impact: Design capabilities configured — visual verification + designer-app integration + persistent design-system skill.
- 🧪 Tests: pending session restart to verify MCP connections
- 📁 Files: .mcp.json, .claude/skills/lab-bench-design/SKILL.md
- 🔁 Next prompts: restart session → verify MCP connections → build personalized logo (Signal Bot badge concept) using Canva/Playwright, then wire into docusaurus.config.ts (task T016)
- 🧠 Reflection: User rejected Playwright-only scope to include designer apps — surface assumptions about scope before configuring tools.

## Evaluation notes (flywheel)

- Failure modes observed: initially proposed Playwright only; user wanted designer-app MCPs included — ask for the full desired toolset upfront
- Graders run and results (PASS/FAIL): pending restart
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): restart, run claude mcp list, then logo concept work
