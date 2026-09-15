# Plan: Chatbot Widget Restyle (ChatKit Component Layer)

**Status**: APPROVED by user on 2026-09-15.

## Plan Purpose

Migrate the 9 ChatKit component stylesheets from ~59 hardcoded hex colors onto the `--chatkit-*` variable system (already Lab Bench-mapped by T012), so the chat button and chat interface render in the site's design system and follow light/dark mode. CSS-only; no behavior, DOM, or backend changes.

## Technical Context

- **Stack**: ChatKit = React components in `rag_chatbot/chatkit/components/` (JSX + colocated CSS), mounted on the book site via `PortalManager` into `#chatkit-portal-root` (with `style.all = 'initial'`), and also used by the standalone rag_chatbot app.
- **Token source**: `rag_chatbot/chatkit/styles/variables.css` — `:root` = light Lab Bench values, `[data-theme='dark']` = dark Lab Bench values (both correct since T012).
- **Key fact**: CSS custom properties inherit through `style.all = 'initial'` — only non-custom properties are reset — so `--chatkit-*` set at `:root`/`[data-theme='dark']` cascade into the portal. The T012 variable layer already reaches the widget; the components just never read it.
- **Theme classes**: `theme.css` ships `.chatkit-theme-light/dark` class-based overrides (standalone-app theming). These must not fight the attribute theming; resolution: scope their use to the standalone app context (they only apply when a class is set; the book site never sets them — verify, then leave intact or neutralize conflicts).
- **Constraints**: no JSX changes except inline color literals (none found); standalone app must keep working off the `:root` defaults; no new dependencies.

## Implementation Sequence

### Phase 1 — Audit + variable layer verification

1. Enumerate every hardcoded hex in the 9 component CSS files into a mapping table (hex → `--chatkit-*` token). Any value with no token equivalent (e.g. spinner white) resolves to `currentColor` or a token.
2. Verify the portal inheritance assumption on the served site (inspect computed `--chatkit-primary` on the launcher) before mass migration.

### Phase 2 — Launcher + panel chrome (the visible surface)

3. `ChatLauncherButton.css`: green → `--chatkit-primary`, hover/streaming → `--chatkit-primary-dark`, focus ring → primary-tinted, shadow → `--chatkit-shadow`; replace scale-jump hover with elevation + subtle scale (≤1.05).
4. `ChatPanel.css` + `MobileChatDrawer.css`: white panel → `--chatkit-background`, header/borders → tokens, shadow → `--chatkit-shadow-heavy`, header title → display font stack (`var(--ifm-font-family-base)` is unavailable inside the portal — use the site font stack literals already used by tokens, or extend variables.css with `--chatkit-font-display/body` referencing the bundled fonts).

### Phase 3 — Conversation surface

5. `ChatMessageList.css`: list background → `--chatkit-background-light`, custom scrollbar → token-tinted.
6. `ChatMessageBubble.css`: user bubble → primary/white, assistant → background-light/ink, sources divider → border token, citation link green → primary; timestamps/meta → text-secondary.
7. `MarkdownRenderer.css`: blockquote/code/link/table colors → tokens matching the docs conventions.
8. `ErrorBubble.css`: → `--chatkit-error(-light)` tokens (already correct values).

### Phase 4 — Input + tooltip

9. `ChatInputBar.css`: input surface/border/focus → tokens, send button green → primary, disabled → text-disabled, hint/status green → primary.
10. `SelectionTooltip.css`: surface/border/text → tokens.

### Phase 5 — Validation

11. Grep gate: zero green hexes (`#00C26A|#00a85c|#007a4d`) and zero remaining non-token hexes in component CSS (whitelist: pure `#fff`/`#ffffff` on primary backgrounds, portal-specific literals if unavoidable).
12. `npm run build` (book site) passes; interactive dual-mode pass (open/close, send, stream, mobile drawer, tooltip, error state) on the served site; standalone rag_chatbot app smoke-render.
13. Dual-mode screenshots via Playwright MCP for the user's visual confirmation.

## Key Decisions and Rationale

- **CSS variables over restyled components** — the T012 variable layer is correct and complete; consuming it is the smallest viable diff and automatically gains dark mode. Alternative (per-component CSS modules) rejected: larger diff, breaks standalone app.
- **Attribute-based theming, no JS listener** — custom properties inherit into the portal; `[data-theme='dark']` on `<html>` already flips the whole widget. No `MutationObserver` needed.
- **Font tokens via new `--chatkit-font-*` variables** — the portal is CSS-reset (`style.all='initial'`), so site font variables don't reach it; adding font variables to `variables.css` (fed by the site's bundled fonts) is the clean channel. Standalone app falls back to system stack.
- **Keep class-based `.chatkit-theme-*` in theme.css** — standalone app theming; the book site never sets those classes, so no conflict (verify in Phase 1).

## Risks

- **Standalone app regression** (it shares these files) — mitigated by `:root` light defaults + smoke render.
- **Portal reset surprises** (some property not covered by the reset) — mitigated by Phase 1 step 2 verification before migration.
- **Specificity fights between component CSS and theme.css** — resolved in Phase 2 step 4 audit.
