---
id: 0012
title: Chapter 3 tasks generation for Module 3
stage: tasks
date: 2025-12-15
surface: agent
model: claude-opus-4-5-20251101
feature: module-3-chapter-3
branch: main
user: unknown
command: /sp.phr
labels: ["module-3", "chapter-3", "tasks", "cognitive-architectures", "ai-reasoning"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/tasks.md
tests:
 - none
---

## Prompt

You are operating as a Specification-Driven Development assistant. Your task is to generate **tasks.md** for Chapter 3 of Module 3.

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
- physical-ai-humanoid-robotics-book/specs/module-3/tasks.md
- physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/specification.md
- physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/plan.md
- /specify/memory/constitution.md (Book level)
- Humanoid-Robotics-Book-writing-Hackathon/specification.md (Book level in main directory)
- Humanoid-Robotics-Book-writing-Hackathon/plan.md (Book level in main directory)
- Humanoid-Robotics-Book-writing-Hackathon/tasks.md (Book level in main directory)

OUTPUT TARGET:
- physical-ai-humanoid-robotics-book/specs/module-3/chapter-3/tasks.md
- file type: Markdown format

YOU MUST DEFINE:
1. ✅ Task for Chapter 3 Introduction
   - Create `physical-ai-humanoid-robotics-book/docs/module-3/chapter-3(already present with chapter name)/index.md`
   - Detailed introduction with full concept covering with easy to understand content with easy and detailed steps.
   - easily explained and understandable
   - Based strictly on chapter-3/specification.md
2. ✅ Task for each lesson
   - Create `/docs/module-3/chapter-3/[lesson-name].md`
   - Detailed, step-by-step, easily understandable
   - Detailed High Quality Lesson content
   - With easy explanations and full concept coverage
   - Include learning objectives, conceptual explanations, tools, diagram, examples and code snippets (only if in specs)
3. ✅ Task sequencing based on chapter-3/plan.md
4. ✅ File naming and folder structure:
   - All files in `/docs/module-3/chapter-[03-(Chapter Name)]/`
5. ✅ Validation tasks:
   - Check each lesson file exists
   - Check content aligns with chapter-3/specification.md & plan.md
   - No hallucinations, no cross-module content

STRICT PROHIBITIONS:
- ❌ Do NOT include Module 4 content.

## Response snapshot

Successfully generated the tasks.md file for Chapter 3 of Module 3, including tasks for Chapter 3 Introduction, tasks for each lesson (3.1, 3.2, 3.3), proper task sequencing based on chapter-3/plan.md, and validation tasks to ensure content alignment with specifications.

## Outcome

- ✅ Impact: Chapter 3 tasks for Module 3 created with detailed task breakdown for all lessons
- 🧪 Tests: none
- 📁 Files: Created tasks.md file for Module 3 Chapter 3
- 🔁 Next prompts: none
- 🧠 Reflection: Tasks aligned with Module 3 specifications and prepared for Chapter 3 implementation

## Evaluation notes (flywheel)

- Failure modes observed: create-phr.sh script not found in expected location, causing fallback to manual PHR creation
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Verify script location or implement alternative PHR creation mechanism