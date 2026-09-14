---
name: lab-bench-design
description: Design system for the Physical AI & Humanoid Robotics book site ("Lab Bench"). Use when creating or modifying any UI, styling, components, logos, or visual assets for physical-ai-humanoid-robotics-book — colors, type, spacing, the Signal Path signature, and the site's human copy voice all live here.
---

# Lab Bench Design System

The book's visual identity is grounded in its own subject: the robotics lab.
Dark Isaac Sim viewports, oscilloscope signal traces, blueprint grids, and the
safety-amber of a lab bench. The reader is a curious beginner — the design
should feel like a patient teacher's workbench, not a marketing site.

**Source of truth for tokens**: `physical-ai-humanoid-robotics-book/src/css/custom.css`
(`:root` = light, `[data-theme='dark']` = dark). Never hardcode hex values in
components — always reference the CSS variables.

## Palette (dual mode)

| Token | Light ("whiteboard session") | Dark ("Isaac viewport") |
|---|---|---|
| Page background `--lab-paper` | `#f7f9fb` | `#0a1526` |
| Surface `--lab-surface` | `#ffffff` | `#101e33` |
| Text | `--ink-navy` `#16283e` | `#e4ecf4` |
| Primary **signal teal** | `#0e7490` | `#2dd4bf` |
| Accent **human amber** | `#b45309` / bright `#f59e0b` | `#fbbf24` |
| Hairline borders `--hairline` | `#e3e9ef` | `#1d2f4a` |

Rules:
- **Two accents, never one.** Signal teal is the workhorse; human amber is
  rationed to the moments that address the reader as a human (the "Ask the
  book" chatbot callout, the antenna dot in the logo). Never use amber for
  more than one element per view.
- Text contrast must stay WCAG AA (≥ 4.5:1) in both modes.
- Avoid the generic AI looks: cream + serif + terracotta, single acid accent
  on near-black, broadsheet hairline grids. We are none of those.

## Typography

- **Display**: Space Grotesk (500/700) — headings, hero, card titles
- **Body**: Source Sans 3 (400/600) — all reading text
- **Utility/mono**: IBM Plex Mono (400/500) — eyebrows, module numbers, code
- Eyebrows: mono, uppercase, `letter-spacing: 0.22em`, signal teal
- Docs measure: `max-width: 70ch`; line-height 1.65
- All fonts bundled via @fontsource — no CDN

## The signature — the Signal Path

One oscilloscope trace (flat–spike–flat polyline) appears in exactly two
places: the homepage hero (animated `stroke-dashoffset`, disabled under
`prefers-reduced-motion`) and the connecting line across the module journey
cards. **Spend boldness here; keep everything else quiet.** The numbered
module sequence (01–04) is real information — modules are prerequisites —
not decoration.

Logo direction (when creating logo/brand assets): the "Signal Bot badge" —
rounded `--viewport-navy` badge, robot face whose mouth IS the signal trace,
teal eyes, single amber antenna dot. Works on light and dark backgrounds
because the badge carries its own background.

## Copy voice

- Plain verbs, sentence case, active voice. Specific beats clever.
- Buttons say what happens: "Start with the preface," not "Learn more."
- The reader is capable and busy: "Stuck at 2 a.m.? Ask the book."
- Never: "cutting-edge," "revolutionary," "empower," "seamless," emoji in
  UI chrome, exclamation marks in headings.
- All user-facing copy requires author approval before shipping (spec rule).

## Quality floor (non-negotiable)

- Responsive down to 360px; visible keyboard focus (`outline: 2px solid
  var(--signal-teal)`); `prefers-reduced-motion` respected
- `npm run build` must pass with `onBrokenLinks: "throw"`
- Specs live in `physical-ai-humanoid-robotics-book/specs/ui-ux-redesign/`

## Verification workflow

When Playwright MCP is available: screenshot changes in **both color modes**
before considering them done. A design you haven't seen is a design you
haven't made.
