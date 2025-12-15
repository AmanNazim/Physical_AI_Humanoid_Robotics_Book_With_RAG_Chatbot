---
id: 0011
title: Chapter 3 plan generation for Module 3
stage: plan
date: 2025-12-15
surface: agent
model: claude-opus-4-5-20251101
feature: module-3-chapter-3
branch: main
user: unknown
command: /sp.phr
labels: ["module-3", "chapter-3", "plan", "cognitive-architectures", "ai-reasoning"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/plan.md
tests:
 - none
---

## Prompt

You are operating as a Specification-Driven Development assistant. Your task is to generate **plan.md** for **Chapter 3** of Module 3.

LEVEL CONTEXT:
- Level: Chapter
- Book: Physical_AI_Humanoid_Robotics_Book
- Module: Module 3
- Chapter: Chapter 3
- Hierarchy: Book → Module → Chapter → Lesson

INPUT REFERENCES: (Mandatory To Read For Consistency)
- physical-ai-humanoid-robotics-book/specs/module-3/constitution.md
- physical-ai-humanoid-robotics-book/specs/module-3/specification.md
- physical-ai-humanoid-robotics-book/specs/module-3/plan.md
- physical-ai-humanoid-robotics-book/specs/module-3/tasks.md (for validation)
- physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/specification.md
- /specify/memory/constitution.md (Book level)
- Humanoid-Robotics-Book-writing-Hackathon/specification.md (Book level in main directory)
- Humanoid-Robotics-Book-writing-Hackathon/plan.md (Book level in main directory)
- Humanoid-Robotics-Book-writing-Hackathon/tasks.md (Book level in main directory)

OUTPUT TARGET:
- physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/plan.md
- File type: Markdown

YOU MUST DEFINE:
1. ✅ Lesson-wise roadmap based on chapter-3/specification.md
2. ✅ Step-by-step lesson sequence
3. ✅ Milestones and expected outcomes per lesson
4. ✅ Integration references to Module 4
5. ✅ Dependencies between lessons

STRICT PROHIBITIONS:
- ❌ Do NOT include lesson content or coding steps
- ❌ Do NOT include Module 4 content
- ❌ Do NOT hallucinate tools or milestones

CONTRADICTION DETECTION:
- Validate lesson sequence vs Module 3 plan.md
- Validate lessons vs chapter-3/specification.md
- Output "CONTRADICTION REPORT" if mismatches

FORMAT:
- Markdown hierarchy:
  # Chapter 3 – [Title]
  ## Lessons Roadmap
  ### Lesson 1
  - Estimated Duration:
  - Milestones:
  - Dependencies:
  (Repeat for all lessons)
  ## Integration Notes
  ## Preparation for Chapter module 3 chapter 4.

## Response snapshot

Created the plan for Chapter 3 of Module 3 (Cognitive Architectures) including lesson roadmap, implementation sequence, milestones, dependencies, and integration references to Module 4, while ensuring consistency with existing specifications and plans.

## Outcome

- ✅ Impact: Chapter 3 plan for Module 3 created with detailed roadmap and milestones
- 🧪 Tests: none
- 📁 Files: Created plan.md file for Module 3 Chapter 3
- 🔁 Next prompts: none
- 🧠 Reflection: Plan aligned with Module 3 specifications and prepared for Chapter 3 implementation

## Evaluation notes (flywheel)

- Failure modes observed: create-phr.sh script not found in expected location, causing fallback to manual PHR creation
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Verify script location or implement alternative PHR creation mechanism