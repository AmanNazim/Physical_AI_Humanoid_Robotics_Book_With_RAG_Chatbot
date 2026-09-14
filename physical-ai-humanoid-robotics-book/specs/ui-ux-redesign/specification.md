# Specification: UI/UX Redesign for Physical AI Humanoid Robotics Book

## 1. Purpose and Scope

The book's web UI is currently the unmodified Docusaurus starter template: a default green palette, generic feature cards ("Educational Resource", "Practical Implementation", "Cutting-Edge Technology"), and stock visuals. This redesign replaces that anonymous, AI-templated presentation with a distinctive, human design system grounded in the book's own world — the robotics lab — while keeping the Docusaurus/Infima architecture intact.

**Design brief (user decisions, binding):**

- **Visual direction**: Dual mode — one refined design system tuned for both light and dark modes.
- **Scope**: Full — homepage rebuild, docs reading experience, and ChatKit chatbot widget restyle.
- **Copy**: New human-toned copy replaces the generic template copy; the author reviews and approves all copy before it ships.

**In scope:**

- A named design token system (color, type, spacing, radius, shadow) for light and dark modes, implemented through Infima CSS variables.
- Webfont pairing: display, body, and utility (mono) typefaces.
- Homepage rebuild: hero, module journey cards (real book structure), RAG chatbot callout.
- Docs reading experience: typography, sidebar, links, code blocks, admonitions, table of contents, navbar, footer.
- ChatKit widget restyle via its existing `--chatkit-*` CSS variables, including dark-mode support.
- Removal of unused template assets.

**Out of scope:**

- Changes to book content (markdown chapters), sidebar information architecture, or site navigation structure.
- Changes to chatbot behavior, backend, or RAG logic.
- New Docusaurus plugins or a framework migration.
- Logos and imagery redesign beyond removing unused template files (the existing `physical-ai-logo.png` stays).

## 2. Design Concept — "The Lab Bench"

The book's core metaphor (from the preface) is the robot's *nervous system* — signals traveling through a body that learns by doing. The design borrows from the learner's actual world: dark Isaac Sim viewports, oscilloscope signal traces, blueprint grids, and the safety-amber of robotics labs.

- **Signature element**: *The Signal Path* — a single SVG oscilloscope-style trace in the hero, with a slow pulse animation. It returns as a connecting line between the four module cards, because the modules genuinely form a prerequisite sequence. The signature lives in exactly these two places; everything else stays quiet.
- **Numbered modules (01–04)** are justified here: the book's modules are a true ordered progression, and the numbers encode real prerequisite information.

### 2.1 Color Tokens (named, dual mode)

**Light mode — "whiteboard session"**

| Token | Value | Role |
|---|---|---|
| `--lab-paper` | `#F7F9FB` | Page background (cool white, not cream) |
| `--lab-surface` | `#FFFFFF` | Cards and raised surfaces |
| `--ink-navy` | `#16283E` | Primary text |
| `--signal-teal` | `#0E7490` | Primary/brand, links, active states |
| `--human-amber` | `#B45309` text / `#F59E0B` decorative | Accent: highlights, chatbot callout, pulse |
| `--grid-line` | `rgba(14, 116, 144, 0.06)` | Hero blueprint grid |

**Dark mode — "Isaac viewport"**

| Token | Value | Role |
|---|---|---|
| `--viewport-navy` | `#0A1526` | Page background |
| `--viewport-surface` | `#101E33` | Cards and raised surfaces |
| `--paper-text` | `#E4ECF4` | Primary text |
| `--signal-teal` | `#2DD4BF` | Primary/brand, links, active states |
| `--human-amber` | `#FBBF24` | Accent |
| `--grid-line` | `rgba(45, 212, 191, 0.05)` | Hero blueprint grid |

Both modes carry **two** accents (signal teal + human amber) on a navy-grounded base — the palette of a robotics HMI, not the single-accent-on-black default. Contrast: all text tokens meet WCAG AA (≥ 4.5:1) against their backgrounds in their respective modes.

### 2.2 Typography

| Role | Typeface | Weights | Usage |
|---|---|---|---|
| Display | Space Grotesk | 500, 700 | Hero headline, section headings, module card titles |
| Body | Source Sans 3 | 400, 600 | Paragraphs, docs body text, UI copy |
| Utility | IBM Plex Mono | 400, 500 | Eyebrows, module numbers, code blocks, metadata |

Typefaces load via `@fontsource/*` packages bundled locally (no external font CDN dependency at runtime). Base body size 16–17px; docs measure capped at ~68ch; headings follow a fixed scale.

### 2.3 Homepage Wireframe

```
┌──────────────────────────────────────────────────────┐
│ NAV   logo · Book                     GitHub · ☀/☾   │
├──────────────────────────────────────────────────────┤
│ ▒▒ faint blueprint grid + signal trace ▒▒            │
│   EYEBROW (mono, teal): FOR CURIOUS BEGINNERS        │
│   H1 (display): Robots that learn by doing.          │
│                 You will, too.                       │
│   sub: The four-module path from ROS 2 to robots     │
│        that see, listen, and act — no degree needed. │
│   [ Start with the preface ]  [ Meet your study      │
│                                companion ]           │
├──────────────────────────────────────────────────────┤
│   THE JOURNEY (eyebrow)                               │
│   ──●──────────●──────────●──────────●── signal path │
│   [01 ROS 2]   [02 Twin]  [03 Isaac] [04 VLA]        │
│   each card: title, one human sentence, chapter count │
├──────────────────────────────────────────────────────┤
│   ASK THE BOOK  (amber-accent callout, chat icon)    │
│   RAG chatbot pitch + "how it answers" honesty note  │
├──────────────────────────────────────────────────────┤
│   FOOTER (token-driven restyle, structure unchanged) │
└──────────────────────────────────────────────────────┘
```

## 3. User Stories & Acceptance Scenarios

### User Story 1 — Landing with confidence (Priority: P1)

A curious beginner with no robotics background lands on the homepage and within ten seconds understands what the book is, who it's for, and how to start reading.

**Why this priority**: The homepage is the book's front door; the current generic hero fails this job completely.

**Independent test**: Load `/` in both color modes; the hero, journey cards, and chatbot callout render with the new design and working CTAs.

**Acceptance scenarios**:

1. Given a first-time visitor, when the homepage loads, then the hero states the book's promise in plain human language (no "cutting-edge" filler) and offers a working "Start with the preface" link to `/docs/preface/`.
2. Given the visitor scans below the hero, then they see the four real modules in reading order with one-sentence descriptions and chapter counts — not generic marketing cards.
3. Given the visitor prefers dark mode (OS or toggle), then the homepage renders the "Isaac viewport" variant with all tokens swapped and no illegible or invisible elements.
4. Given the visitor has `prefers-reduced-motion` enabled, then the signal-trace animation is disabled and the trace renders as a static line.

### User Story 2 — Reading in comfort (Priority: P1)

A learner spends hours in the docs and finds the reading experience calm, readable, and consistent with the homepage identity.

**Why this priority**: The docs ARE the book; reading comfort outranks decoration.

**Independent test**: Open any module chapter in both modes; typography, sidebar, links, admonitions, and code blocks render per the token system.

**Acceptance scenarios**:

1. Given any docs page, when rendered, then body text uses the body typeface at ≥16px with a measure ≤70ch, and headings use the display typeface per the fixed scale.
2. Given the docs sidebar, when navigating, then active items are marked with the signal-teal indicator, hover/focus states are visible, and the previous conflicting sidebar CSS overrides are resolved (single source of truth in `custom.css`).
3. Given inline links and admonitions (info/tip/warning/danger), when rendered in either mode, then they use the new palette with AA contrast and do not use Infima's default green.
4. Given code blocks, when rendered, then they use the utility mono typeface with a code background that matches the surface token in both modes.

### User Story 3 — Chatting with a matching companion (Priority: P2)

A learner opens the ChatKit widget and finds it visually consistent with the book — palette, radius, and dark-mode behavior included.

**Why this priority**: The chatbot is a headline feature but currently ships a separate green identity that breaks cohesion.

**Independent test**: Open the chat launcher on any page in both modes; panel, launcher, and tooltip render with the new tokens.

**Acceptance scenarios**:

1. Given the chat launcher and panel, when rendered, then `--chatkit-*` variables map to the new design tokens (signal teal primary, matching neutrals, radius, shadows).
2. Given dark mode, when the widget opens, then it renders dark surfaces and light text — currently it has no dark variant at all.
3. Given the widget on mobile, then existing breakpoints/behavior are unchanged (styling only).

### User Story 4 — Trusting the chrome (Priority: P3)

Navbar and footer feel like part of the same design system instead of stock Docusaurus furniture.

**Independent test**: Visual inspection in both modes; all footer links still resolve.

**Acceptance scenarios**:

1. Given the navbar, when scrolled or static, then it uses surface tokens with a visible bottom border and correct contrast in both modes.
2. Given the footer, when rendered, then it uses navy/surface tokens (no default `style: "dark"` green footer) and every existing link still works (build passes with `onBrokenLinks: "throw"`).

## 4. Non-Functional Requirements

- **Accessibility**: WCAG AA text contrast in both modes; visible keyboard focus on all interactive elements; `prefers-reduced-motion` respected; color is never the only signal.
- **Performance**: No runtime external font/script CDN; webfonts subset via @fontsource; CSS-only animations (GPU-friendly `opacity`/`stroke-dashoffset`); no layout-affecting JS added.
- **Compatibility**: Existing Docusaurus v4-compat config, `respectPrefersColorScheme: true` retained; responsive from 360px to wide desktop.
- **Build integrity**: `npm run build` passes with `onBrokenLinks: "throw"` unchanged.

## 5. Constraints and Non-Goals

- **Smallest viable diff per task**: token layer first, then components; no refactor of unrelated code (e.g., the duplicated `src/client-modules/` vs `src/clientModules/` folders are left alone).
- **No content changes**: markdown chapters, sidebar structure, and footer link targets are frozen.
- **Copy freeze pending review**: no new user-facing copy ships until the author approves the draft copy (review checkpoint in the plan).

## 6. Definition of Done

- [ ] All four user stories' acceptance scenarios pass in both light and dark mode.
- [ ] `npm run build` succeeds with zero broken links.
- [ ] No unused template assets (undraw SVGs etc.) remain referenced.
- [ ] All user-facing copy matches the author-approved draft verbatim.
- [ ] Specs (specification/plan/tasks) committed under `specs/ui-ux-redesign/` with PHRs recorded.
