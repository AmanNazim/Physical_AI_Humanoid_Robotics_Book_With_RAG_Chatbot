# Tasks: Chatbot Widget Restyle (ChatKit Component Layer)

**Input**: Design documents from `specs/chatbot-widget-restyle/` (specification.md, plan.md)
**Status**: Complete 2026-09-15 — all tasks done, validated in both modes.

**Organization**: Tasks grouped by phase; each phase lands as one conventional commit referencing task IDs. Acceptance per task is testable by grep, build, or inspection in both color modes.

## Task Group 1 — Audit + verification

- [x] T001 Build the hex → token mapping table for all 9 component CSS files (~59 hardcoded hexes); flag any value with no token equivalent and decide its resolution (currentColor / new token / whitelisted literal)
- [x] T002 Verify on the served site that `--chatkit-*` custom properties cascade into `#chatkit-portal-root` (computed style on the launcher) and that `.chatkit-theme-*` classes are never set by the book site

## Task Group 2 — Launcher + panel chrome

- [x] T003 Migrate `ChatLauncherButton.css`: green `#00C26A` → `--chatkit-primary`, hover/streaming `#00a85c` → `--chatkit-primary-dark`, focus ring, shadow → `--chatkit-shadow`; soften hover to elevation + ≤1.05 scale
- [x] T004 Add `--chatkit-font-display/body` variables to `variables.css` (site bundled font stacks with system fallbacks)
- [x] T005 Migrate `ChatPanel.css`: white panel → `--chatkit-background`, header/borders → border/text tokens, shadow → `--chatkit-shadow-heavy`, header title → `--chatkit-font-display`
- [x] T006 Migrate `MobileChatDrawer.css` onto the same tokens (drawer surface, header, backdrop)

## Task Group 3 — Conversation surface

- [x] T007 Migrate `ChatMessageList.css`: list background → `--chatkit-background-light`, scrollbar → token-tinted
- [x] T008 Migrate `ChatMessageBubble.css`: user bubble → primary/white, assistant → background-light/ink, sources divider + citation green (`#007a4d`) → tokens, timestamps/meta → text-secondary
- [x] T009 Migrate `MarkdownRenderer.css`: blockquote/code/link/table colors → tokens matching docs conventions
- [x] T010 Migrate `ErrorBubble.css` → `--chatkit-error(-light)` tokens

## Task Group 4 — Input + tooltip

- [x] T011 Migrate `ChatInputBar.css`: input surface/border/focus, send button green → primary, disabled state, hint/status green → tokens
- [x] T012 Migrate `SelectionTooltip.css` onto tokens

## Task Group 5 — Validation + shipping

- [x] T013 Grep gate: zero `#00C26A|#00a85c|#007a4d` and no non-whitelisted hexes remain in `rag_chatbot/chatkit/components/*.css`
- [x] T014 `npm run build` passes; interactive dual-mode pass (open/close, send, stream, mobile drawer, tooltip, error state) on the served site; standalone rag_chatbot app smoke-render
- [x] T015 Dual-mode screenshots via Playwright MCP for the user's visual confirmation; commit, push, log hashes in `validation.md`

## Acceptance Criteria (from specification.md)

- [x] US1: no green anywhere in the widget; grep gate passes
- [x] US2: widget follows light/dark mode on all surfaces
- [x] US3: no behavioral regression; build passes; standalone app renders
- [x] US4: typography/spacing/focus polish visible in both modes
