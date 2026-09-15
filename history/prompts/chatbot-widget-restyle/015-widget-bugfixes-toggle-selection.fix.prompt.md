---
id: 015
title: Chat Widget Bugfixes — Toggle, Selection Delivery, One-Shot Context, Tooltip Anchor
stage: fix
date: 2026-09-15
surface: agent
model: glm-5.3
feature: chatbot-widget-restyle
branch: main
user: Aman Nazim
command: none
labels: [chatkit, bugfix, theme-toggle, selection, e2e-verification]
links:
  spec: physical-ai-humanoid-robotics-book/specs/chatbot-widget-restyle/specification.md
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/docusaurus.config.ts
  - rag_chatbot/chatkit/services/chatService.js
  - rag_chatbot/chatkit/components/ChatInputBar.jsx
  - rag_chatbot/chatkit/components/SelectionTooltip.jsx
  - rag_chatbot/chatkit/components/SelectionTooltip.css
tests:
  - Toggle PASS: binary cycle verified — light→dark→light→dark→light, every single click flips (aria-label confirms no third "system" state)
  - Selection delivery PASS: fetch interception captured wire payload 'Explain me this\n\n[Selected text from the book, use as primary context]: "Robots that learn by doing..."' after full UI flow (select → Ask AI → type → send)
  - One-shot PASS: chip cleared after first send (placeholder reset), second payload 'What is a humanoid robot?' contains no selection marker
  - Tooltip anchor PASS: docs page scrolled 500px, tooltip 12px above selection, gap preserved at 12px after +250px scroll (anchored, not viewport-stuck), inside viewport
  - Build PASS: npm run build exit 0
---

## Prompt

ok so there are two bugs one is when switching between dark mode and light mode you have to click two time for one mode to change for example if I am switching from light mode to dark it needs only one click but when I am switching back to dark I have to click two times to switch and another bugs is that selecting text is not going to LLM it is only showing selected in UI both when writing prompt and sent prompt but the LLM is not getting it and I am saying by selecting explain me this the LLM is unaware and telling me which topics you want to understand and giving genric answers because it is not receiving selected text.

(Subsequent additions during verification:)
"there is one thing also after prompting with selected text the selected text is not clearing up for next prompt in UI and I don't whether it is also going again in LLM."
"explain me this button is only working with upper text of books and not working everywhere also sometimes the location of that button is changing it should at top seleted text."
"the ask button is moving with scrolling and sticking to selected at top when scrolling."

## Response snapshot

Diagnosed and fixed five interrelated bugs, each verified live via Playwright before committing:

1. **Theme toggle double-click**: Docusaurus with respectPrefersColorScheme cycles THREE states (light→dark→system). The "system" stop looks like a dead click when the OS preference matches the current theme. Fixed: respectPrefersColorScheme: false → binary toggle. Verified: 5 consecutive clicks each flip the theme.
2. **Selection not reaching LLM**: ChatInputBar put selected_text into messageData.context, but chatService.sendMessage built the wire payload from scratch (query/session_id/max_context only) — context dropped client-side; backend ChatRequest schema also has no context field. Fixed: buildQuery() composes the selection into the query string (trimmed 1500 chars, tagged "[Selected text from the book, use as primary context]"), used by both stream and regular endpoints. Verified via fetch interception: payload contains the selection.
3. **Selection not clearing after send**: handleSubmit never called setSelectedText(null) — chip stayed AND every later prompt silently re-sent the same selection. Fixed: one-shot context, cleared right after the payload is built. Verified: second payload has no selection marker.
4. **Tooltip only worked at top of page**: position:fixed (viewport coords) fed document coords (rect.top + scrollY) — coincident at scrollY=0, off-screen once scrolled. Also the 1000-char limit suppressed tooltips on long docs paragraphs (raised to 2000) and rangeCount>0 guard added; fixed keyup-listener cleanup leak (anonymous removal never fired).
5. **Tooltip sticking to viewport while scrolling**: position:fixed anchors to viewport, not content. Fixed: position:absolute + document coordinates (rect + scroll offsets), clamped to the viewport at selection time. Verified: gap to selection preserved at 12px across a 250px scroll.

All verified on the dev server, then npm run build exit 0. Commit 382ffc7 pushed.

## Outcome

- ✅ Impact: all five user-reported behaviors fixed and e2e-verified; no backend changes needed.
- 🧪 Tests: five targeted verifications + build, all PASS
- 📁 Files: 5 files
- 🔁 Next prompts: user re-reviews deployed site
- 🧠 Reflection: Three of five bugs were invisible at the top of the homepage (scrollY=0, short text, no scrolling) — the hero is the worst place to test a widget. Future widget QA should happen on a real docs page, scrolled, with long selections.

## Evaluation notes (flywheel)

- Failure modes observed: coordinate-system mismatch (fixed vs absolute) and dropped payload field (context) — both silent because the UI layer showed correct state while the transport/position layer was wrong. Verification must intercept the WIRE (fetch body) and the GEOMETRY (bounding rects), not the React state.
- Graders run and results (PASS/FAIL): PASS (5/5 verifications + build)
- Prompt variant (if applicable): null
- Next experiment (smallest change to try): user review on deployed site
