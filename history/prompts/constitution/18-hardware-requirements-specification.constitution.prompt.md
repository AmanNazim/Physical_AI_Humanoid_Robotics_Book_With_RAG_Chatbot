---
id: 18
title: Hardware Requirements Specification
stage: constitution
date_iso: 2025-12-16
model: Claude Opus 4.5
surface: agent
feature: hardware-requirements
branch: main
user: claude
command: Create hardware requirements specification
labels: ["specification", "hardware", "requirements", "spec"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/specs/hardware-requirements/specification.md
tests: []
prompt_text: |
  You are operating as a SPEC-DRIVEN-DEVELOPMENT (SDD) Specification Author.

  Your task is to create a `specification.md` file for a standalone, appendix-level section of a technical book titled:

  Hardware Requirements

  This specification MUST strictly comply with the existing `constitution.md` for this appendix.
  If any instruction would violate the constitution, you must refuse to include it.

  INPUT REFERENCE: (Mandatory to read.)
  file path: physical-ai-humanoid-robotics-book/specs/hardware-requirements/constitution.md

  ========================
  SPECIFICATION PURPOSE
  ========================

  The purpose of this specification is to precisely define:
  - WHAT information the Hardware Requirements section presents
  - HOW that information is structured
  - WHAT is explicitly allowed and disallowed

  This specification does NOT define implementation steps or learning content.

  ========================
  SCOPE (MANDATORY)
  ========================

  The Hardware Requirements section SHALL:
  - Provide environmental execution context for the book
  - Help readers understand computational capability needs
  - Support both simulation-first and physical-extension workflows

  The Hardware Requirements section SHALL NOT:
  - Introduce mandatory dependencies
  - Affect assessments or grading
  - Constrain module or lesson content
  - Require physical hardware ownership

  ========================
  REQUIRED STRUCTURE
  ========================

  The Hardware Requirements section MUST contain the following sections, in order:

  1. Purpose and Scope Disclaimer
  2. Tier 1 — Minimum Requirements
  3. Tier 2 — Recommended Requirements
  4. Tier 3 — Optional / Premium (Advanced or Physical AI)
  5. Cloud and Remote Execution Equivalence
  6. Tables
  7. Architectural Overview (Conceptual, Non-Operational)
  8. Limitations and Tradeoffs (Declarative Only)

  ========================
  TIER DEFINITIONS
  ========================

  ### Tier 1 — Minimum Requirements
  - Describes the lowest-capability environment sufficient for:
    - Reading the book
    - Running lightweight simulations
    - Completing conceptual and planning exercises
  - MUST NOT assume access to discrete GPUs or edge devices

  ### Tier 2 — Recommended Requirements
  - Describes a local development environment capable of:
    - Running full simulation workloads
    - Supporting perception pipelines
    - Executing Vision–Language–Action workflows
  - MAY reference representative hardware classes (e.g., RTX-capable GPUs, multi-core CPUs)
  - MUST remain non-mandatory

  ### Tier 3 — Optional / Premium (Advanced or Physical AI)
  - Describes environments for:
    - Sim-to-real experimentation
    - Edge deployment
    - Physical robot integration
  - MUST explicitly state that this tier is optional and not required for course completion
  - MAY reference edge AI devices, sensors, and robots as examples ONLY

  Each tier MAY inherit capabilities from lower tiers.

  ========================
  CONTENT RULES (STRICT)
  ========================

  1. NO PRICING
  - The specification MUST forbid:
    - Prices
    - Cost estimates
    - Budget ranges
    - Cost comparisons (CapEx vs OpEx)

  2. NO TUTORIALS
  - The specification MUST prohibit:
    - Setup instructions
    - Installation steps
    - Configuration guides
    - OS, driver, or firmware walkthroughs

  3. NO PROCUREMENT GUIDANCE
  - The specification MUST forbid:
    - Purchase advice
    - Vendor recommendations framed as requirements
    - Shopping lists

  4. LANGUAGE CONSTRAINTS
  - Use capability-based, descriptive language
  - Avoid imperative phrases like "must buy", "required purchase", or "mandatory hardware"
  - Use "may", "can", "is suitable for", "supports" instead

  ========================
  CLOUD SUPPORT REQUIREMENTS
  ========================

  The specification MUST require a dedicated section stating that:
  - Cloud-based GPU workstations are valid alternatives to local machines
  - Simulation-first workflows are fully supported
  - Performance and latency tradeoffs may be described, but not framed as blockers

  ========================
  ARCHITECTURE DESCRIPTION RULES
  ========================

  If an architectural overview is included:
  - It MUST be conceptual and descriptive
  - It MUST describe roles (simulation, inference, sensing, actuation)
  - It MUST NOT include wiring diagrams, commands, or deployment steps

  ========================
  OUTPUT REQUIREMENTS
  ========================

  - High Quality/Level Docusaurus Markdown.
  - Output ONLY valid `specification.md` content
  - file path: physical-ai-humanoid-robotics-book/specs/hardware-requirements/specification.md
  - file type: Markdown.
  - Do NOT include implementation text
  - Do NOT include tables with prices
  - Do NOT include references to assessments or grading
  - Ensure full consistency with the appendix `constitution.md`

  The resulting specification must be precise enough that a separate `plan.md` and `tasks.md` can be generated without introducing hardware mandates or instructional leakage.

response_text: |
  Successfully created the Hardware Requirements specification file at physical-ai-humanoid-robotics-book/specs/hardware-requirements/specification.md. The specification includes:

  1. Purpose and Scope Disclaimer - clearly stating non-normative status
  2. Tier 1 — Minimum Requirements - describing basic capability needs
  3. Tier 2 — Recommended Requirements - outlining development environment capabilities
  4. Tier 3 — Optional / Premium (Advanced or Physical AI) - for sim-to-real experimentation
  5. Cloud and Remote Execution Equivalence - guaranteeing cloud alternatives
  6. Tables - for capability comparisons (without pricing)
  7. Architectural Overview - conceptual roles description
  8. Limitations and Tradeoffs - declarative information only

  The specification complies with all constitutional requirements, avoiding prohibited content like pricing, tutorials, and procurement guidance while using capability-based, descriptive language.

outcome: |
  Hardware Requirements specification created with proper tiered structure and constitutional compliance, ensuring all content remains informational without mandating hardware purchases.

evaluation: |
  The specification successfully meets all requirements by maintaining non-normative status, ensuring simulation-first workflows are supported, and preventing hardware ownership from becoming a requirement for course completion.
---