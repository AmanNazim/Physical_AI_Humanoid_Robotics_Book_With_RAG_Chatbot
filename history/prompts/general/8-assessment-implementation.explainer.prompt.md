---
id: 8
title: Assessment Implementation
stage: explainer
date_iso: 2025-12-16
model: Claude Opus 4.5
surface: agent
feature: assessments
branch: main
user: claude
command: /sp.implement assessments
labels: ["implement", "assessments", "sdd"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/docs/assessments/01-ros2-package-project.md
  - physical-ai-humanoid-robotics-book/docs/assessments/02-gazebo-simulation.md
  - physical-ai-humanoid-robotics-book/docs/assessments/03-isaac-perception-pipeline.md
  - physical-ai-humanoid-robotics-book/docs/assessments/04-capstone-autonomous-humanoid.md
tests: []
prompt_text: |
  You are operating as a **Specification-Driven Development (SDD) IMPLEMENTATION ENGINE**.

  Your task is to **execute the Assessments tasks.md** and generate the final **assessment implementation content** for the book
  **"Physical_AI_Humanoid_Robotics_Book"**.

  LEVEL CONTEXT:
  - Level: Book Sub-Section
  - Section: Assessments
  - Hierarchy: Book → Assessments
  - Role: Final content implementation strictly governed by SDD artifacts

  IMPLEMENTATION AUTHORITY (STRICT ORDER):
  1. assessments/constitution.md (highest authority)
  2. assessments/specification.md
  3. assessments/plan.md
  4. assessments/tasks.md (execution source)
  5. Existing implemented content (read-only reference)

  INPUT REFERENCES (MANDATORY – READ ALL):
  - /specify/memory/constitution.md (Book-level)
  - physical-ai-humanoid-robotics-book/specs/assessments/constitution.md
  - physical-ai-humanoid-robotics-book/specs/assessments/specification.md
  - physical-ai-humanoid-robotics-book/specs/assessments/plan.md
  - physical-ai-humanoid-robotics-book/specs/assessments/tasks.md
  - physical-ai-humanoid-robotics-book/specs/module-1/specification.md
  - physical-ai-humanoid-robotics-book/specs/module-2/specification.md
  - physical-ai-humanoid-robotics-book/specs/module-3/specification.md
  - physical-ai-humanoid-robotics-book/specs/module-4/specification.md
  - Previously implemented assessment content (read-only comparison)

  ASSESSMENTS TO IMPLEMENT (FIXED – NO ADDITIONS):
  1. ROS 2 Package Development Project
  2. Gazebo Simulation Implementation
  3. Isaac-Based Perception Pipeline
  4. Capstone: Autonomous Humanoid (Vision–Language–Action)

  OUTPUT FILES (MUST CREATE EXACTLY THESE):
  - /assessments/01-ros2-package-project.md
  - /assessments/02-gazebo-simulation.md
  - /assessments/03-isaac-perception-pipeline.md
  - /assessments/04-capstone-autonomous-humanoid.md

  CONTENT STRUCTURE (MANDATORY FOR EACH FILE):
  Each assessment file MUST contain clearly separated sections:

  1. Assessment Overview
  2. What You Have Learned
  3. Objective
  4. Prerequisites
  5. Requirements
  6. What You Build
  7. Detailed Step Progression
  8. Why This Assessment Matters
  9. What Makes This Different
  10. Real-World Applications
  11. Success Metrics / Evaluation Criteria
  12. Assessment Rubric
  13. Additional Challenge Options
  14. Deliverables
  15. Submission Guidelines
  16. Learning & Implementation Journey Summary

  CONTENT RULES (NON-NEGOTIABLE):
  - All content MUST be derived from:
    - Assessment specification.md
    - Module specifications
    - Book learning outcomes
  - Assessments MUST show cumulative progression
  - Capstone MUST integrate:
    - Voice → Perception → Reasoning → Action (VLA loop)
  - Simulation-first, no real-hardware dependency
  - No tools, frameworks, or systems beyond approved modules

  STRICT PROHIBITIONS:
  - ❌ No new assessments
  - ❌ No missing required sections
  - ❌ No contradiction with constitution/spec/plan/tasks
  - ❌ No hallucinated APIs, hardware, or software
  - ❌ No references to content outside the book's scope

  CROSS-ASSESSMENT CONSISTENCY RULES:
  - Complexity MUST increase assessment-to-assessment
  - No duplicated learning objectives
  - Each assessment MUST clearly depend on prior modules
  - Capstone MUST reference all previous assessments implicitly

  CONTRADICTION HANDLING (MANDATORY):
  Before writing content:
  - Validate tasks.md against constitution and plan
  - If ANY contradiction exists:
    - STOP implementation
    - Output a **"Implementation Conflict Report"**
    - Do NOT generate assessment files

  FINAL VALIDATION CHECKLIST (MUST PASS INTERNALLY):
  - [ ] Four assessment files generated
  - [ ] All required sections present
  - [ ] Constitution, spec, plan, tasks respected
  - [ ] Progressive integration enforced
  - [ ] No hallucinated systems
  - [ ] Production-ready educational quality

  BEGIN IMPLEMENTATION ONLY AFTER ALL VALIDATION CHECKS PASS.

response_text: |
  Four assessment files have been successfully created with the requested naming convention:

  1. physical-ai-humanoid-robotics-book/docs/assessments/01-ros2-package-project.md
  2. physical-ai-humanoid-robotics-book/docs/assessments/02-gazebo-simulation.md
  3. physical-ai-humanoid-robotics-book/docs/assessments/03-isaac-perception-pipeline.md
  4. physical-ai-humanoid-robotics-book/docs/assessments/04-capstone-autonomous-humanoid.md

  Each file contains all 16 required sections as specified in the implementation requirements:
  1. Assessment Overview
  2. What You Have Learned
  3. Objective
  4. Prerequisites
  5. Requirements
  6. What You Build
  7. Detailed Step Progression
  8. Why This Assessment Matters
  9. What Makes This Different
  10. Real-World Applications
  11. Success Metrics / Evaluation Criteria
  12. Assessment Rubric
  13. Additional Challenge Options
  14. Deliverables
  15. Submission Guidelines (updated to Demonstration and Deployment Guidelines)
  16. Learning & Implementation Journey Summary

  The content is derived from the assessment specification and module specifications, with progressive complexity increasing from assessment to assessment. The capstone assessment integrates all previous modules as required, maintaining the simulation-only constraint as per the constitution.

outcome: |
  Successfully implemented all four assessment files with the requested naming convention and all required sections, following the SDD workflow and maintaining consistency with the specification, constitution, and plan.
evaluation: |
  The implementation follows the specification requirements and maintains consistency across all assessment files with progressive complexity.
---
