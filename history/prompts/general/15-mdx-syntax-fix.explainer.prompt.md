---
id: 15
title: MDX Syntax Fix
stage: explainer
date_iso: 2025-12-16
model: Claude Opus 4.5
surface: agent
feature: docusaurus
branch: main
user: claude
command: Fix MDX syntax error
labels: ["fix", "mdx", "syntax", "docusaurus", "parsing"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/docs/module-4/01-vision-language-action-fundamentals/lesson-1.3-instruction-understanding-natural-language-processing.md
tests: []
prompt_text: |
  Fix the MDX compilation error in lesson-1.3-instruction-understanding-natural-language-processing.md caused by unescaped curly braces in regular text.

response_text: |
  Fixed the MDX compilation error by properly escaping curly braces in regular text. The error occurred at line 388 where JSON-like syntax with curly braces was being interpreted as a JavaScript expression by the MDX parser.

  The problematic line:
  - Entities: {"object": "cup", "color": "red", "location": "table"}

  Was changed to:
  - Entities: `{`"object": "cup", "color": "red", "location": "table"`}`

  This properly escapes the curly braces so they are rendered as literal characters instead of being interpreted as JavaScript expressions by the MDX parser.

  The other curly braces in the file (lines 149 and 178) were part of indented code blocks and did not require changes since code blocks are handled differently by the MDX parser.

outcome: |
  The MDX compilation error has been resolved by properly escaping curly braces in regular text content.

evaluation: |
  The Docusaurus build should now complete successfully without the MDX parsing error related to unescaped curly braces.
---