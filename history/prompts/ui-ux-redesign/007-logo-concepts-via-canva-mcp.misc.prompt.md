---
id: 007
title: Logo Concepts via Canva MCP
stage: misc
date: 2026-09-14
surface: agent
model: glm-5.3
feature: ui-ux-redesign
branch: main
user: Aman Nazim
command: none
labels: [logo, canva-mcp, playwright-mcp, review]
links:
  spec: physical-ai-humanoid-robotics-book/specs/ui-ux-redesign/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - logo-review/canva-1.png
  - logo-review/canva-2.png
  - logo-review/canva-3.png
  - logo-review/canva-4.png
  - logo-review/signal-bot.svg
  - logo-review/index.html
tests:
  - file verified: 4 PNGs are 400x400 image data (not HTML)
---

## Prompt

ok so I am back and configured mcp also.

## Response snapshot

Canva, Figma, and Playwright MCP servers confirmed live after session restart. Executed the logo workflow:

1. Called Canva generate-design (design_type: logo) with the Signal Bot badge brief (navy #0A1526 badge, teal #2DD4BF robot face with oscilloscope-trace mouth, amber #F59E0B antenna dot, flat vector, favicon-crisp). Canva returned 4 candidates.
2. Thumbnail URLs were JS redirects; used Playwright navigation to resolve each to its signed S3 URL and downloaded the real 400x400 PNGs into logo-review/.
3. Hand-built the fifth candidate: logo-review/signal-bot.svg — a 64x64 SVG from the exact Lab Bench tokens, with the site's Signal Path trace as the robot's mouth.
4. Built logo-review/index.html — a self-contained candidate sheet showing all 5 on light + dark surfaces and a 32px favicon strip, styled with the design system.
5. Asked the user to open the sheet and pick; noted harness limitation (image Read uploads to CDN rather than rendering to the agent, so the user is the visual judge).

## Outcome

- ✅ Impact: 5 logo candidates ready for human review with a proper comparison sheet.
- 🧪 Tests: PNG integrity verified via file(1); sheet is static HTML, no build needed
- 📁 Files: 6 files in logo-review/ (uncommitted until decision)
- 🔁 Next prompts: user picks a candidate → wire into docusaurus.config.ts (navbar + favicon) as task T016 → Playwright dual-mode screenshot on the live site → commit + validation log
- 🧠 Reflection: First real use of the design MCPs. Canva candidates arrived fast but need human eyes; the hand-built SVG guarantees token fidelity. The review sheet pattern (light/dark/32px) is reusable for any future visual decision.

## Evaluation notes (flywheel)

- Failure modes observed: curl of design.canva.ai thumbnail URLs returns HTML (JS redirect) — resolve via Playwright navigation to the signed S3 URL first
- Graders run and results (PASS/FAIL): PASS (4/4 PNGs valid)
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): user picks candidate; if Canva wins, also export a transparent PNG via export-design
