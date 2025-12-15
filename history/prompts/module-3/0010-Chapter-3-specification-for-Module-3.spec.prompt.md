---
id: 0010
title: Chapter 3 specification for Module 3
stage: spec
date: 2025-12-15
surface: agent
model: claude-opus-4-5-20251101
feature: module-3-chapter-3
branch: main
user: unknown
command: /sp.phr
labels: ["module-3", "chapter-3", "specification", "cognitive-architectures", "ai-reasoning"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/specification.md
tests:
 - none
---

## Prompt

You are operating as a Specification-Driven Development assistant. Your task is to generate **specification.md** for **Chapter 3** of **Module 3** in the book "Physical_AI_Humanoid_Robotics_Book".

LEVEL CONTEXT:
- Level: Chapter
- Book: Physical_AI_Humanoid_Robotics_Book
- Module: Module 3 – The AI-Robot Brain (NVIDIA Isaac™)
- Chapter: Chapter 3
- Hierarchy: Book → Module → Chapter → Lesson

INPUT REFERENCES (MANDATORY To Read For Consistency):
- physical-ai-humanoid-robotics-book/specs/module-3/constitution.md
- physical-ai-humanoid-robotics-book/specs/module-3/specification.md
- physical-ai-humanoid-robotics-book/specs/module-3/plan.md
- physical-ai-humanoid-robotics-book/specs/module-3/tasks.md (for scope validation)
- /specify/memory/constitution.md (Book level)
- Humanoid-Robotics-Book-writing-Hackathon/specification.md (Book level)
- Humanoid-Robotics-Book-writing-Hackathon/plan.md (Book level)
- Humanoid-Robotics-Book-writing-Hackathon/tasks.md (Book level)

OUTPUT TARGET:
- File path: physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/specification.md
- File type: Markdown

YOU MUST DEFINE:
1. ✅ Chapter 3 Title (derived strictly from Module 3 scope)
2. ✅ Chapter 3 Description aligned with Module 3 trajectory
3. ✅ Complete Lesson Breakdown:
   - Small, understandable, sequential lessons
   - Traceable to Module 3 specification + plan
4. ✅ For EACH lesson:
   - Learning Objective
   - Conceptual Scope
   - Expected Learning Outcome
   - Tools/Technologies (ONLY if in Module 3 spec)
5. ✅ Chapter Dependencies:
   - Relation to Chapter 2 of module 3
   - Preparation for module 3 chapter 4.

STRICT PROHIBITIONS:
- ❌ Do NOT invent technologies outside Module 3 scope
- ❌ Do NOT define implementation steps, code, exercises
- ❌ Do NOT include content from Module 4

CONTRADICTION DETECTION:
- Cross-check lessons vs Module 3 specification.md
- Cross-check scope vs Module 3 plan.md
- If any conflict exists, output a "CONTRADICTION REPORT"

FORMAT:
- Markdown hierarchy:
  # Chapter 3 – [Title]
  ## Chapter Description
  ## Learning Objectives
  ## Lessons Breakdown
  ### Lesson 1 – [Title]
  - Objective:
  - Scope:
  - Expected Outcome:
  - Tools:
  (Repeat for all lessons)
  ## Chapter Dependencies

VALIDATION:
- [ ] Lessons fully inside Module 3 scope
- [ ] Traceable to Module 3 spec & plan
- [ ] No hallucinations.

## Response snapshot

Created the specification for Chapter 3 of Module 3 (Cognitive Architectures) including chapter description, learning objectives, lesson breakdown with objectives and expected outcomes, and dependencies on previous chapter and preparation for next chapter.

## Outcome

- ✅ Impact: Chapter 3 specification for Module 3 created with detailed content covering Cognitive Architectures
- 🧪 Tests: none
- 📁 Files: Created specification.md file for Module 3 Chapter 3
- 🔁 Next prompts: none
- 🧠 Reflection: Specification aligned with Module 3 scope and trajectory, focusing on cognitive architectures and AI reasoning

## Evaluation notes (flywheel)

- Failure modes observed: create-phr.sh script not found in expected location, causing fallback to manual PHR creation
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Verify script location or implement alternative PHR creation mechanism