# Plan: UI/UX Redesign for Physical AI Humanoid Robotics Book

**Status**: APPROVED by user on 2026-09-14 (plan and homepage copy). Implementation in progress.

## Plan Purpose

Implement the "Lab Bench" design system defined in `specification.md` across the Docusaurus book site in dual (light/dark) mode, covering the homepage, the docs reading experience, and the ChatKit widget, with author-approved copy.

## Technical Context

- **Stack**: Docusaurus (classic preset, Infima CSS, v4 future flag), React/TSX components, CSS modules
- **Styling surface**: `src/css/custom.css` (global tokens — the single source of truth), CSS modules for page components
- **Fonts**: `@fontsource/space-grotesk`, `@fontsource/source-sans-3`, `@fontsource/ibm-plex-mono` (bundled, no CDN)
- **ChatKit**: token-driven via `rag_chatbot/chatkit/styles/variables.css` (`--chatkit-*`); currently light-only
- **Constraints**: `onBrokenLinks: "throw"`; no markdown/sidebar/content changes; `respectPrefersColorScheme: true` retained
- **Testing**: visual verification in both modes (build + serve), `npm run build` as the acceptance gate; no unit tests (CSS/TSX presentation layer)

## Implementation Sequence

### Phase 1 — Foundation (tokens + fonts)

1. Add @fontsource dependencies; import weights in `custom.css`.
2. Replace Infima color variables in `custom.css` with the dual-mode token system (`:root` = light, `[data-theme='dark']` = dark), including fonts, radius, and shadow scales.
3. Resolve the conflicting sidebar overrides (`.menu` duplicated rules, hardcoded 280px/60px) into one coherent block driven by tokens.

### Phase 2 — Homepage rebuild

4. Rebuild `src/pages/index.tsx` + `index.module.css`: blueprint-grid hero with signal-trace SVG (CSS `stroke-dashoffset` pulse, gated by `prefers-reduced-motion`), eyebrow, headline, sub, two CTAs.
5. Replace `HomepageFeatures` with `ModuleJourney` (4 module cards + connecting signal line, mono module numbers) and `StudyCompanion` (amber-accent chatbot callout).
6. Restyle navbar and footer via tokens (footer `style: "dark"` stays; its colors now come from tokens).

### Phase 3 — Docs reading experience

7. Docs typography: body/display/mono assignment, heading scale, measure cap (≤70ch), blockquote and table styling.
8. Links, admonitions (info/tip/warning/danger), code block backgrounds, TOC on-right page styling — all de-greened and dual-mode.
9. Sidebar: active indicator, hover/focus states, category headers using tokens.

### Phase 4 — ChatKit restyle

10. Remap `--chatkit-*` variables to the new tokens in `rag_chatbot/chatkit/styles/variables.css`; add `[data-theme='dark']` chatkit block; verify breakpoints/behavior unchanged.

### Phase 5 — Cleanup + validation

11. Remove unused template assets (`undraw_*.svg`, unused illustrations) after grep-verifying no references.
12. Full validation: `npm run build`, visual pass in both modes at 360px/768px/1280px, keyboard focus check, reduced-motion check; screenshots for review.

## Copy Draft (pending author review — ships only after approval)

**Hero**
- Eyebrow (mono): `FOR CURIOUS BEGINNERS`
- H1: `Robots that learn by doing. You will, too.`
- Sub: `A four-module path from your first ROS 2 node to robots that see, listen, and act — no robotics degree required.`
- CTA primary: `Start with the preface` · CTA secondary: `Meet your study companion`

**Module journey** (eyebrow: `THE JOURNEY`)
1. `01 · ROS 2 — the nervous system` · `Wire up your robot's senses and muscles. Every chapter that follows runs on the communication skills you build here.`
2. `02 · Digital twin — Gazebo & Unity` · `Rehearse in simulation before touching hardware. Break things for free and learn from every crash.`
3. `03 · The AI brain — NVIDIA Isaac` · `Give your robot judgment: process what it senses and decide what to do next.`
4. `04 · Vision-Language-Action` · `Put it all together — robots that understand your words, see their surroundings, and act on both.`

**Study companion** (eyebrow: `ASK THE BOOK`)
- Title: `Stuck at 2 a.m.? Ask the book.`
- Body: `A chatbot that answers from these pages, not the whole internet — and shows you where it read the answer.`

## Constitution Check

- [ ] Smallest viable diff per phase; no unrelated refactors (duplicated client-module folders untouched)
- [ ] No content/sidebar IA changes; no invented APIs or data
- [ ] Accessibility floor: AA contrast, focus visibility, reduced-motion support
- [ ] Copy frozen until author approval (spec §5)

## Risks

- ChatKit CSS is shared with the standalone `rag_chatbot` app — variable remap must not degrade that surface (verified in Phase 4).
- `onBrokenLinks: "throw"` means any homepage link typo fails the build — caught by Phase 5 build.
- Infima variable coverage: some components need explicit overrides beyond the palette vars — contained in `custom.css`.

## Approval Gate

User approves: (a) this plan, (b) the copy draft above. Then `tasks.md` is authored and implementation begins, one conventional commit per task (`feat:`/`chore:`, referencing Module: UI/UX Redesign).
