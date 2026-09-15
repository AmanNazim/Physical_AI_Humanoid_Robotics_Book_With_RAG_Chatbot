---
id: 013
title: Chatbot Widget Restyle Implemented and Validated
stage: green
date: 2026-09-15
surface: agent
model: glm-5.3
feature: chatbot-widget-restyle
branch: main
user: Aman Nazim
command: none
labels: [chatkit, implementation, dark-mode, validation]
links:
  spec: physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - rag_chatbot/chatkit/styles/variables.css
  - rag_chatbot/chatkit/components/ChatLauncherButton.css
  - rag_chatbot/chatkit/components/ChatPanel.css
  - rag_chatbot/chatkit/components/MobileChatDrawer.css
  - rag_chatbot/chatkit/components/ChatMessageList.css
  - rag_chatbot/chatkit/components/ChatMessageBubble.css
  - rag_chatbot/chatkit/components/MarkdownRenderer.css
  - rag_chatbot/chatkit/components/ErrorBubble.css
  - rag_chatbot/chatkit/components/ChatInputBar.css
  - rag_chatbot/chatkit/components/SelectionTooltip.css
  - physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/tasks.md
tests:
  - T002 PASS: --chatkit-* custom properties cascade into #chatkit-portal-root (computed --chatkit-primary #0e7490 on launcher; flips to #2dd4bf under [data-theme='dark'])
  - T013 PASS: grep gate — zero green hexes (#00C26A|#00a85c|#007a4d) and zero non-whitelisted hexes in components/*.css
  - T014 PASS: npm run build exit 0 (only pre-existing chatkit re-export warnings, unrelated); dual-mode interactive check via Playwright (launcher, panel, header, input bar, send button all token-driven and mode-flipping); standalone-consumer inspection: only consumer is the book site
  - T015 screenshots captured: chatbot-light-open.png, chatbot-dark-open.png
---

## Prompt

ok I approve the plan.

## Response snapshot

Executed the approved plan end to end (T001–T015):

**Phase 1 — audit + verify (T001-T002):** Enumerated every hardcoded hex across the 9 component CSS files. Verified on the live dev server (Playwright) that --chatkit-* custom properties cascade through the portal's style.all='initial' reset onto the launcher — light #0e7490, dark #2dd4bf — and that .chatkit-theme-* classes are never set by the book site. Confirmed the green was hardcoded, not variable-driven.

**Phases 2–4 — migration (T003-T012):** Rewrote all 9 component stylesheets onto --chatkit-* tokens: launcher (green→primary, hover scale-jump softened to 1.05 + elevation), ChatPanel + MobileChatDrawer (token surfaces, hairline borders, --chatkit-shadow-heavy, header font), message list (token scrollbar), bubbles (user=primary/white, assistant=background-light/ink with hairline border, sources/typing/error states), MarkdownRenderer (blockquote/code/links/tables de-greened, mono font token), ErrorBubble (error tokens), ChatInputBar (focus ring via primary-light, send button primary, placeholder token), SelectionTooltip (primary pill). Added --chatkit-font-display/body/mono variables (portal reset blocks site font inheritance; standalone falls back to system stacks).

**Phase 5 — validation (T013-T015):** Grep gate zero green/stray hexes. npm run build exit 0. Live dual-mode verification via Playwright evaluate: every widget surface flips correctly (light: white panel/teal button; dark: navy panel #101e33/bright teal #2dd4bf/light text). Screenshots captured in both modes. Standalone-consumer risk resolved by inspection: the book site is the only consumer (preview/docs/tests dirs empty). All 19 task/US checkboxes marked complete.

Commits: 4fb2a64 (feat: migration, T001-T015), 21feed6 (spec set, prior commit).

## Outcome

- ✅ Impact: chat widget fully in the Lab Bench design system with working dark mode; green eliminated.
- 🧪 Tests: all four gates PASS (inheritance, grep, build, live dual-mode)
- 📁 Files: 11 files
- 🔁 Next prompts: user reviews screenshots/live site; pre-existing chatkit re-export warnings remain a separate future fix
- 🧠 Reflection: The plan's front-loaded inheritance verification (Phase 1) paid off — migration was then purely mechanical. Front-loading the risky assumption is the pattern to repeat.

## Evaluation notes (flywheel)

- Failure modes observed: none during execution; CSS arithmetic in var() (var(--a) + 4px) is invalid — caught and replaced with the nearest radius token before it could ship
- Graders run and results (PASS/FAIL): PASS on all gates
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): user visual review; if happy, close feature and update lab-bench-design SKILL.md with the token-consumption pattern
