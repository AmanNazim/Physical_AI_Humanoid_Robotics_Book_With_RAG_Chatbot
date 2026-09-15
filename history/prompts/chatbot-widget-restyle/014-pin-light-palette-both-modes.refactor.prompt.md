---
id: 014
title: Chat Widget Pinned to Light Palette in Both Site Modes
stage: refactor
date: 2026-09-15
surface: agent
model: glm-5.3
feature: chatbot-widget-restyle
branch: main
user: Aman Nazim
command: none
labels: [chatkit, theming, css-specificity, production-fix]
links:
  spec: physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - rag_chatbot/chatkit/styles/variables.css
tests:
  - Dev-server verification PASS: under [data-theme='dark'], portal --chatkit-primary computes to #0e7490 (light pin), panel #ffffff, header title ink #16283e, floating-card shadow rgba(10,21,38,0.35) 0 16px 40px
  - npm run build exit 0 (pre-existing warnings only)
  - Production-bundle serve verification not run (user said push and review themselves)
---

## Prompt

ok so what happened when use playwrite mcp it dislpayed correct theme of interface dark with dark theme and light with theme but after deploying it is only showing light theme of chat interface with both theme but I am liking it because it is not giving too overwhelmed or overlaaping look can we tweak little without changing backgroung to dark color.

(then, when the agent attempted a production-serve verification: "what are you doing if you have completed then push it I will review it.")

## Response snapshot

Diagnosed the dev-vs-production theme difference: the `[data-theme='dark']` override and the `:root` light block have equal CSS specificity (0,1,0 vs 0,1,0), so the winner is decided by stylesheet order — the production bundler ordered the sheets differently than the dev server, and the light `:root` values won in the deployed bundle. So production was light-by-accident.

User likes the light widget in both modes (avoids the heavy overlapping dark-on-dark look) and asked to make it deliberate without changing the background to dark.

Implementation:
1. Pinned the light Lab Bench palette on `#chatkit-portal-root` in variables.css — an id selector (1,0,0) outranks both `:root` and `[data-theme='dark']` regardless of bundle order, making the pin deterministic in dev AND production.
2. Kept the `[data-theme='dark']` block (annotated) for the standalone app / future dark-widget opt-in.
3. Small elevation tweak per "tweak little": `--chatkit-shadow-heavy` on the pinned scope deepened to `0 16px 40px rgba(10, 21, 38, 0.35)` so the light panel reads as an intentional floating whiteboard card over the dark viewport instead of a flat white box.

Verified on the dev server via Playwright: with site theme forced to dark, portal computes #0e7490 primary, white panel, ink header text, deep shadow — light widget over dark site, deterministic. Screenshot chatbot-pinned-light-over-dark.png captured. Production-bundle serve verification was not run (user instructed to push and review themselves).

## Outcome

- ✅ Impact: the light-widget look the user liked is now guaranteed behavior in any bundler, plus a deliberate floating-card elevation.
- 🧪 Tests: dev dual-mode verification PASS; build PASS; production-serve verification skipped per user instruction
- 📁 Files: rag_chatbot/chatkit/styles/variables.css
- 🔁 Next prompts: user reviews deployed site; if the elevation tweak is too strong, one-value change
- 🧠 Reflection: The "bug" (production staying light) was a specificity tie broken differently by bundler order — and the user preferred the accidental result. Turning accidental behavior into an intentional, deterministic rule (id-selector pin) is the right way to honor that preference.

## Evaluation notes (flywheel)

- Failure modes observed: equal-specificity blocks (`:root` vs `[data-theme='dark']` on different elements can still tie through inheritance order) — when theme behavior differs dev vs prod, suspect stylesheet order, not logic
- Graders run and results (PASS/FAIL): PASS (dev verification + build)
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): user review of deployed site
