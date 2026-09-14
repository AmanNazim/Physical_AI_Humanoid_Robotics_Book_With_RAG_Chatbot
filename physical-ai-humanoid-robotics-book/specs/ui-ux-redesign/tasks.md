# Tasks: UI/UX Redesign for Physical AI Humanoid Robotics Book

**Input**: Design documents from `/specs/ui-ux-redesign/` (specification.md, plan.md)
**Status**: Approved 2026-09-14 — implementation in progress

**Organization**: Tasks grouped by phase; each phase lands as one conventional commit referencing task IDs. Acceptance per task is testable by build or inspection in both color modes.

## Task Group 1 — Foundation (tokens + fonts)

- [x] T001 Add @fontsource webfont dependencies (space-grotesk, source-sans-3, ibm-plex-mono) to the book's package.json
- [x] T002 Replace the default green Infima palette in `src/css/custom.css` with the dual-mode "Lab Bench" token system (`:root` light / `[data-theme='dark']` dark), including font-family assignments, code backgrounds, and admonition colors
- [x] T003 Consolidate the conflicting sidebar overrides in `custom.css` into one token-driven block (remove duplicate `.menu` rules, replace hardcoded 280px/60px with vars, add hover/focus/active states)

## Task Group 2 — Homepage rebuild

- [x] T004 Rebuild `src/pages/index.tsx` + `src/pages/index.module.css`: blueprint-grid hero, eyebrow, approved headline/sub, two CTAs
- [x] T005 Implement the Signal Path: animated SVG oscilloscope trace in the hero (`stroke-dashoffset` animation, gated by `prefers-reduced-motion`) and connecting node-line on module cards
- [x] T006 Create `src/components/ModuleJourney/` (4 real module cards with approved copy, mono numbers, chapter counts, links to each module introduction)
- [x] T007 Create `src/components/StudyCompanion/` (amber-accent "Ask the book" callout with approved copy) and remove the replaced `HomepageFeatures`
- [x] T008 Restyle navbar and footer via tokens (border, typefaces, footer link colors; structure and links unchanged)

## Task Group 3 — Docs reading experience

- [x] T009 Docs typography: heading font/weight/scale, 70ch measure, line-height, h2 hairline dividers, blockquote and table styling in `custom.css`
- [x] T010 Restyle links, admonitions (info/tip/warning/danger), code blocks (mono font), and TOC active states — de-greened, dual-mode
- [x] T011 Verify sidebar states on real docs pages (active indicator, hover, keyboard focus) and correct any Infima defaults that still surface green

## Task Group 4 — ChatKit restyle

- [x] T012 Remap `--chatkit-*` variables in `rag_chatbot/chatkit/styles/variables.css` to the design tokens and add a `[data-theme='dark']` block so the widget follows Docusaurus color mode

## Task Group 5 — Cleanup + validation

- [x] T013 Remove unused template assets (undraw/illustration SVGs) after grep-verifying zero references
- [x] T014 Run `npm run build` (with `onBrokenLinks: "throw"`) and `npm run typecheck`; fix any failures
- [x] T015 Final dual-mode review pass; log final commit hash into `validation.md`

## Task Group 6 — Logo (user-built HR monogram)

- [x] T016 Wire the user-built HR monogram ("Black White Minimalist Professional Initial Logo.png") into the site: extract the glyph to transparent PNGs (white variant for dark mode, ink-to-teal remap for light mode, 512px) and the black square as favicon; configure navbar `logo: { src, srcDark }` + `favicon` in `docusaurus.config.ts`

## Acceptance Criteria (from specification.md)

- [x] All four user stories pass in light and dark mode
- [x] Build passes with zero broken links
- [x] No unused template assets remain referenced
- [x] All user-facing copy matches the approved draft verbatim
- [x] Specs and PHRs committed under `specs/ui-ux-redesign/` and `history/prompts/ui-ux-redesign/`
