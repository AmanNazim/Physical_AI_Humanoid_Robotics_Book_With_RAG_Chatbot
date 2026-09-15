# Specification: Chatbot Widget Restyle (ChatKit Component Layer)

## 1. Purpose and Scope

The ui-ux-redesign feature (T012) remapped the `--chatkit-*` CSS variables in `rag_chatbot/chatkit/styles/variables.css` to the Lab Bench design tokens. However, investigation found that **no ChatKit component stylesheet consumes those variables** — `var(--chatkit-*)` appears 0 times across the 9 component CSS files. Every component carries hardcoded hex colors (~59 occurrences), including the launcher green `#00C26A` (`ChatLauncherButton.css:10`) and a hardcoded white ChatPanel that never enters dark mode.

This feature migrates the ChatKit component layer onto the existing `--chatkit-*` variables so the widget renders in the Lab Bench design system (signal teal / ink navy / amber accents) and follows the site's light/dark mode, plus targeted UX polish of the chat interface.

**User decision (binding):** the chat button and chatbot interface should be **improved** to match the site's design system — drafted via spec/plan/task method, integrated into the existing `specs/` workflow, approved before implementation.

**In scope:**

- Migrate all 9 component stylesheets (`ChatLauncherButton`, `ChatPanel`, `ChatMessageList`, `ChatMessageBubble`, `ChatInputBar`, `MarkdownRenderer`, `MobileChatDrawer`, `SelectionTooltip`, `ErrorBubble`) from hardcoded hex colors to the `--chatkit-*` variables.
- Green `#00C26A` / `#00a85c` / `#007a4d` → signal teal (light `#0E7490` / dark `#2DD4BF`) via `--chatkit-primary(-dark)`.
- Neutral grays (`#eee`, `#f9f9f9`, `#333`, `#666`, `#ccc`, …) → `--chatkit-background(-light)`, `--chatkit-border`, `--chatkit-text-primary/secondary/disabled`.
- Full dark-mode support: the widget follows the Docusaurus `data-theme` attribute (variables cascade into the portal root; `style.all = 'initial'` does not reset custom-property inheritance).
- UX polish within existing structure: typography scale inside the panel (Space Grotesk headers / Source Sans 3 body where fonts are already loaded on the site), message bubble spacing and radius, input focus state, header hierarchy, scrollbar tint.
- Resolve the `theme.css` `.chatkit-theme-light/dark` class overrides so they don't fight the attribute-based theming.

**Out of scope:**

- Chatbot behavior, backend, RAG logic, streaming, or API changes.
- Component structure, props, DOM, or class names (CSS-only changes; JSX untouched except where a literal color is inline).
- The standalone rag_chatbot app's layout (it shares these components — the variable defaults keep it working standalone).
- Sidebar/information architecture or any book content.

## 2. Design Concept

The widget is the book's *Study Companion* — on the site it lives in the Lab Bench system, so it uses the same tokens the rest of the site uses:

- **Launcher**: signal-teal circle (light `#0E7490`, dark `#2DD4BF`), white icon; hover = `--chatkit-primary-dark`, no scale-jump (subtle elevation instead); focus ring from primary.
- **Panel**: `--chatkit-background` surface, hairline `--chatkit-border` edges, radius `--chatkit-radius-lg`, `--chatkit-shadow-heavy`.
- **Bubbles**: assistant = `--chatkit-background-light` with `--chatkit-text-primary`; user = `--chatkit-primary` background, white text; error states = `--chatkit-error(-light)`.
- **Header**: display font (Space Grotesk, already bundled on the site), assistant name + status dot (amber `--chatkit-*` accent if present, else primary).
- **Markdown renderer**: links/codes/quotes de-greened to tokens, matching docs styling conventions.

Dark mode is attribute-driven (`[data-theme='dark']` in `variables.css`, already present and correct) — no JS theme listener needed.

## 3. User Stories and Acceptance Criteria

**US1 — Consistent brand:** As a reader, the chat button and panel use the site's teal/ink/amber palette in light mode, with no green visible anywhere in the widget.
- Acceptance: grep of `rag_chatbot/chatkit/components/*.css` finds zero `#00C26A`, `#00a85c`, `#007a4d`; visual check on the served site.

**US2 — Dark mode parity:** As a reader in dark mode, the launcher, panel, bubbles, input, and markdown content render in the dark Lab Bench palette (navy surface, bright teal, light text).
- Acceptance: toggle site theme with the widget open and closed — all widget surfaces switch; no white panel, no unreadable text.

**US3 — No behavioral regression:** As a reader, opening/closing the panel, sending messages, streaming responses, mobile drawer, and selection tooltip behave exactly as before.
- Acceptance: interactive pass in both modes; `npm run build` passes; standalone rag_chatbot app still renders (variables default to light values).

**US4 — Polish:** As a reader, the chat interface typography, spacing, and focus states feel as considered as the docs reading experience.
- Acceptance: header uses the display font; bubbles have consistent radius/spacing; input focus ring visible in both modes; reduced-motion respected (existing `prefers-reduced-motion` rules retained).

## 4. Constraints and Non-Goals

- CSS-only migration — no JSX restructuring, no new dependencies, no build changes.
- The standalone rag_chatbot app must keep working: `variables.css` `:root` defaults remain the light Lab Bench values.
- `--chatkit-z-index-*`, `--chatkit-spacing-*`, `--chatkit-radius-*`, `--chatkit-transition-*` scales are adopted as-is (already defined in T012).
- Keep the existing animation keyframes; only their colors change.
