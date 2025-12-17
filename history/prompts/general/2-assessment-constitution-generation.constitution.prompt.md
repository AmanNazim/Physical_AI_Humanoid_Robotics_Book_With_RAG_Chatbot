---
id: 2
title: Assessment Constitution Generation
stage: constitution
date_iso: 2025-12-16
model: Claude Opus 4.5
surface: agent
feature: assessments
branch: main
user: claude
command: /sp.constitution assessments
labels: ["const", "assessments", "sdd"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/specs/assessments/constitution.md
tests: []
prompt_text: |
  You are Claude Code CLI, operating as a **Specification-Driven Development (SDD) assistant**.

  Your task is to generate a **constitution.md** file for the **Assessments section** of the book
  **"Physical_AI_Humanoid_Robotics_Book"**.

  LEVEL CONTEXT:
  - Level: Book Sub-Section
  - Section: Assessments
  - Hierarchy: Book → Modules → Assessments
  - Purpose: Define non-negotiable rules, scope, constraints, and quality standards governing all assessment implementations.

  OUTPUT TARGET:
  - File Path: physical-ai-humanoid-robotics-book/specs/assessments/constitution.md
  - File Type: Markdown

  INPUT REFERENCES (MANDATORY – READ ALL FOR CONSISTENCY):
  - /specify/memory/constitution.md (Book-level)
  - Humanoid-Robotics-Book-writing-Hackathon/specification.md
  - Humanoid-Robotics-Book-writing-Hackathon/plan.md
  - Humanoid-Robotics-Book-writing-Hackathon/tasks.md
  - physical-ai-humanoid-robotics-book/specs/module-1/constitution.md
  - physical-ai-humanoid-robotics-book/specs/module-2/constitution.md
  - physical-ai-humanoid-robotics-book/specs/module-3/constitution.md
  - physical-ai-humanoid-robotics-book/specs/module-4/constitution.md
  - Existing implemented assessment content (read-only reference)

  SCOPE OF THE CONSTITUTION (STRICT):
  This constitution MUST govern:
  - All assessment specifications
  - All assessment plans
  - All assessment tasks
  - All assessment implementations

  The Assessments section consists of EXACTLY FOUR assessments:
  1. ROS 2 Package Development Project
  2. Gazebo Simulation Implementation
  3. Isaac-Based Perception Pipeline
  4. Capstone: Autonomous Humanoid (Vision–Language–Action)

  WHAT THIS CONSTITUTION MUST DEFINE (NON-NEGOTIABLE):
  - Purpose of the Assessments section
  - Alignment rules with:
    - Book goals
    - Module learning outcomes
  - Assessment progression rules (cumulative, increasing integration complexity)
  - Scope boundaries:
    - What assessments MAY include
    - What assessments MUST NOT include
  - Technical constraints:
    - Simulation-first
    - ROS 2, Gazebo, Isaac, VLA alignment
  - Pedagogical constraints:
    - Application-focused
    - No purely theoretical assessments
  - Structural constraints:
    - One assessment → one implementation file
    - Consistent assessment anatomy
  - Evaluation philosophy:
    - Skills over memorization
    - Systems integration over isolated demos
  - Reproducibility and grading clarity rules
  - Anti-hallucination rules:
    - No tools, frameworks, or hardware outside approved modules
  - Documentation and submission principles

  STRICT PROHIBITIONS:
  - ❌ Do NOT restate full assessment content
  - ❌ Do NOT define detailed steps
  - ❌ Do NOT include rubrics, tasks, or plans
  - ❌ Do NOT introduce new assessments
  - ❌ Do NOT contradict book-level constitution

  CONTRADICTION DETECTION (MANDATORY):
  Before finalizing:
  - Cross-check against Book and Module constitutions
  - If conflicts are found:
    - Insert a **"Constitution Conflict Report"** section
    - Explicitly describe:
      - Conflicting rule
      - Source file
      - Resolution applied or required

  FORMAT REQUIREMENTS:
  - Markdown only
  - Use clear, authoritative language
  - Required sections:
    1. Purpose
    2. Scope
    3. Governing Principles
    4. Assessment Structure Rules
    5. Technical & Pedagogical Constraints
    6. Evaluation Philosophy
    7. Prohibitions
    8. Consistency & Conflict Resolution

  FINAL VALIDATION CHECKLIST (MUST PASS):
  - [ ] Fully aligned with book-level constitution
  - [ ] Governs all four assessments consistently
  - [ ] No assessment implementation details included
  - [ ] No hallucinated tools or systems
  - [ ] Ready for downstream specification.md generation

  BEGIN ONLY AFTER ALL VALIDATION RULES PASS.

response_text: |
  # Constitution: Assessments Section for Physical AI Humanoid Robotics Book

  ## Purpose

  The Assessments section serves as the capstone evaluation mechanism for the Physical AI Humanoid Robotics Book curriculum. These assessments validate that learners can successfully integrate knowledge from all four modules into practical, hands-on implementations. The assessments must demonstrate competency in ROS 2 architecture, simulation environments, AI-based perception systems, and Vision-Language-Action capabilities within safe, simulation-only constraints.

  ## Scope

  This constitution governs all assessment specifications, plans, tasks, and implementations for the four required assessments: ROS 2 Package Development Project, Gazebo Simulation Implementation, Isaac-Based Perception Pipeline, and Capstone: Autonomous Humanoid (Vision-Language-Action). All assessment-related artifacts must comply with the principles defined herein. The scope includes assessment structure, technical requirements, pedagogical constraints, and evaluation standards.

  ## Governing Principles

  **Assessment Integration Progression**: Each assessment must build upon previous modules, creating a cumulative learning experience where complexity and system integration increase progressively. Assessment 1 focuses on Module 1 concepts, Assessment 2 integrates Modules 1-2, Assessment 3 integrates Modules 1-3, and Assessment 4 synthesizes all four modules.

  **Simulation-First Safety**: All assessments must operate exclusively within simulation environments to ensure safety while providing realistic learning experiences. No physical hardware implementation or real-world deployment is permitted at the assessment level.

  **Application-Over-Theory**: Assessments must focus on practical implementation and application of concepts rather than theoretical understanding alone. Each assessment must result in functional, demonstrable systems.

  **Consistency in Structure**: All four assessments must follow the same structural format including objectives, prerequisites, what you build, evaluation criteria, and submission guidelines.

  ## Assessment Structure Rules

  Each of the four assessments must be implemented as a separate, standalone file in the documentation system. The assessment files must follow the naming convention: `assessment-{number}-{descriptor}.md`. Each assessment must include the same core sections: Learning Outcomes, Objectives, Prerequisites, What You Build, Step-by-Step Progression, Evaluation Criteria, and Submission Guidelines.

  All assessments must maintain clear alignment with their corresponding module learning outcomes while progressively incorporating concepts from previous modules. Assessment 4 must explicitly integrate all concepts from Modules 1-4.

  ## Technical & Pedagogical Constraints

  Technical implementations must exclusively use the frameworks and tools established in the book modules: ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action systems. No external tools, frameworks, or hardware platforms may be introduced. All implementations must operate within simulation environments.

  Pedagogically, assessments must prioritize systems integration over isolated component demonstrations. Each assessment must require learners to demonstrate competency in connecting multiple subsystems rather than implementing standalone features. Safety validation must be integrated into all assessments, particularly in Assessment 4 which combines all systems.

  ## Evaluation Philosophy

  Assessment evaluation must prioritize demonstrated skills and practical implementation over memorization of concepts. Systems integration competency must be valued more highly than isolated feature development. Evaluation criteria must be objective and reproducible, allowing for consistent grading across different implementations.

  Assessments must evaluate the learner's ability to troubleshoot, debug, and validate their implementations rather than just achieving basic functionality. Documentation quality and code organization must be considered as part of the evaluation process.

  ## Prohibitions

  The following are strictly prohibited in all assessments:
  - Implementation of real-world physical robot control without simulation safety layer
  - Introduction of new tools, frameworks, or hardware not covered in the book modules
  - Purely theoretical assessments without practical implementation
  - Assessment content that contradicts module constitutions or book-level principles
  - Implementation details that bypass safety validation protocols
  - Assessment steps that require resources outside the established technology stack

  ## Consistency & Conflict Resolution

  This constitution must maintain full alignment with the book-level constitution and all four module constitutions. Any contradictions must be resolved by deferring to the book-level constitution. If conflicts arise between this assessment constitution and module constitutions, the module constitutions take precedence for their specific domain while maintaining overall book alignment.

  All assessment specifications, plans, and tasks must be validated against this constitution before implementation. Any proposed changes to assessment content that conflict with these principles must be reviewed and approved through the book's constitutional amendment process.

  ---

  ## Constitution Conflict Report

  No conflicts were identified during cross-checking against the book-level and module constitutions. All assessment principles align with the established safety-first approach, simulation requirements, and technical stack constraints defined in the higher-level constitutions. The assessment progression model supports the cumulative learning objectives established in the book and module constitutions.

outcome: |
  Successfully created the constitution.md file for the assessments section in the correct directory: physical-ai-humanoid-robotics-book/specs/assessments/constitution.md
evaluation: |
  The constitution aligns with the book-level constitution and module specifications, establishing clear governing principles and constraints for all assessment implementations.
---
