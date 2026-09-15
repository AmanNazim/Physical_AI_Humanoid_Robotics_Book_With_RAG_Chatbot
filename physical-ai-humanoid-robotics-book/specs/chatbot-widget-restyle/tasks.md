# Tasks: Chatbot Widget Restyle (ChatKit Component Layer)

**Input**: Design documents from `specs/chatbot-widget-restyle/` (specification.md, plan.md)
**Status**: Approved 2026-09-15 — implementation in progress.

**Organization**: Tasks grouped by phase; each phase lands as one conventional commit referencing task IDs. Acceptance per task is testable by grep, build, or inspection in both color modes.

## Task Group 1 — Audit + verification

- [ ] T001 Build the hex → token mapping table for all 9 component CSS files (~59 hardcoded hexes); flag any value with no token equivalent and decide its resolution (currentColor / new token / whitelisted literal)
- [ ] T002 Verify on the served site that `--chatkit-*` custom properties cascade into `#chatkit-portal-root` (computed style on the launcher) and that `.chatkit-theme-*` classes are never set by the book site

## Task Group 2 — Launcher + panel chrome

- [ ] T003 Migrate `ChatLauncherButton.css`: green `#00C26A` → `--chatkit-primary`, hover/streaming `#00a85c` → `--chatkit-primary-dark`, focus ring, shadow → `--chatkit-shadow`; soften hover to elevation + ≤1.05 scale
- [ ] T004 Add `--chatkit-font-display/body` variables to `variables.css` (site bundled font stacks with system fallbacks)
- [ ] T005 Migrate `ChatPanel.css`: white panel → `--chatkit-background`, header/borders → border/text tokens, shadow → `--chatkit-shadow-heavy`, header title → `--chatkit-font-display`
- [ ] T006 Migrate `MobileChatDrawer.css` onto the same tokens (drawer surface, header, backdrop)

## Task Group 3 — Conversation surface

- [ ] T007 Migrate `ChatMessageList.css`: list background → `--chatkit-background-light`, scrollbar → token-tinted
- [ ] T008 Migrate `ChatMessageBubble.css`: user bubble → primary/white, assistant → background-light/ink, sources divider + citation green (`#007a4d`) → tokens, timestamps/meta → text-secondary
- [ ] T009 Migrate `MarkdownRenderer.css`: blockquote/code/link/table colors → tokens matching docs conventions
- [ ] T010 Migrate `ErrorBubble.css` → `--chatkit-error(-light)` tokens

## Task Group 4 — Input + tooltip

- [ ] T011 Migrate `ChatInputBar.css`: input surface/border/focus, send button green → primary, disabled state, hint/status green → tokens
- [ ] T012 Migrate `SelectionTooltip.css` onto tokens

## Task Group 5 — Validation + shipping

- [ ] T013 Grep gate: zero `#00C26A|#00a85c|#007a4d` and no non-whitelisted hexes remain in `rag_chatbot/chatkit/components/*.css`
- [ ] T014 `npm run build` passes; interactive dual-mode pass (open/close, send, stream, mobile drawer, tooltip, error state) on the served site; standalone rag_chatbot app smoke-render
- [ ] T015 Dual-mode screenshots via Playwright MCP for the user's visual confirmation; commit, push, log hashes in `validation.md`

## Acceptance Criteria (from specification.md)

- [ ] US1: no green anywhere in the widget; grep gate passes
- [ ] US2: widget follows light/dark mode on all surfaces
- [ ] US3: no behavioral regression; build passes; standalone app renders
- [ ] US4: typography/spacing/focus polish visible in both modes
