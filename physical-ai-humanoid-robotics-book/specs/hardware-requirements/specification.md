# Specification: Hardware Requirements for Physical AI Humanoid Robotics Book

## 1. Purpose and Scope Disclaimer

The Hardware Requirements section serves as a non-normative reference for readers to understand the computational environments that support the Physical AI Humanoid Robotics curriculum. This section provides capability-based guidance for execution environments while ensuring that all book content remains completable using simulation and cloud-based environments.

This section is purely informational and does not affect learning flow, assessment validity, or curriculum scope. No learner is required to own, purchase, or access any listed hardware. All book content and assessments are completable via simulation and/or cloud-based environments. This appendix does not impose learning or grading constraints.

## 2. Tier 1 — Minimum Requirements

The minimum requirements describe the lowest-capability environment sufficient for:
- Reading the book
- Running lightweight simulations
- Completing conceptual exercises

The minimum requirements include:
- **CPU**: Modern multi-core CPU
- **RAM**: 16–32 GB
- **GPU**: Any capable of running lightweight simulations
- **OS**: Linux/Windows/Mac
- **Optional**: Cloud-based simulation for LLM/VLA tasks

This tier does not assume access to discrete GPUs or physical devices. The environment should support basic development tasks and educational content consumption, emphasizing accessibility and flexibility.

## 3. Tier 2 — Recommended Requirements

The recommended requirements describe a local development environment suitable for:
- Physics-based simulation
- Visual perception pipelines
- Vision–Language–Action workflows

The recommended requirements include:
- **GPU**: RTX-capable GPU with at least 12GB VRAM (e.g., NVIDIA RTX 4070 Ti or higher) to handle USD assets and VLA models simultaneously
- **CPU**: High-performance multi-core processor (e.g., Intel Core i7 13th Gen+ or AMD Ryzen 9) for physics calculations
- **RAM**: 64 GB DDR5 for complex scene rendering and multi-process workloads
- **OS**: Ubuntu 22.04 LTS for optimal ROS 2 compatibility
- **Software**: Gazebo, Isaac Sim for simulation environments

This tier uses capability-based descriptions for recommended requirements and mentions representative hardware classes as examples only. These requirements are non-mandatory and may be substituted with equivalent capabilities.

## 4. Tier 3 — Optional / Premium (Advanced or Physical AI)

The optional/premium tier describes optional environments for:
- Sim-to-real workflows
- Edge deployment
- Physical robot integration

This tier includes:
- **Edge devices**: NVIDIA Jetson platforms (Orin Nano, Orin NX) for deployment to resource-constrained environments, described as illustrative examples only
- **Sensors**: Depth cameras (Intel RealSense), IMUs, and microphones for perception and interaction, described as illustrative examples only
- **Robots**: Humanoid or quadruped platforms for physical deployment, described as illustrative examples only

This tier explicitly states that it is optional and advanced. Physical hardware is not required for course completion, and these are extensions beyond the core curriculum.

Each higher tier may inherit capabilities from lower tiers, allowing users to build upon their existing setup.

## 5. Cloud and Remote Execution Equivalence

This section must state that cloud-based GPU environments are valid alternatives to local machines. Simulation-first workflows are fully supported in cloud environments.

The section should acknowledge that performance and latency considerations may exist in cloud environments but must not frame these as blockers to completing the curriculum. Cloud options provide an alternative path for users without access to high-end local hardware.

Cloud execution provides equivalent functionality to local setups at a conceptual level. Latency or performance considerations may exist in cloud environments, but these should not be considered blockers to completing the curriculum. Cloud options provide an alternative path for users without access to high-end local hardware.

## 6. Conceptual Architecture Overview

The architecture overview describes the roles of different system components:
- **Simulation Systems**: High-performance workstations for Isaac Sim, Gazebo, and Unity
- **Inference / AI Execution**: Systems for executing AI models, including VLA processing and perception pipelines
- **Sensing**: Cameras, IMUs, and other perception hardware for collecting environmental data
- **Actuation**: Robot platforms for physical interaction and control

This overview must be conceptual and descriptive, focusing on the roles and relationships between different hardware components rather than operational details. The descriptions remain high-level and non-operational, avoiding diagrams, commands, or deployment steps in the architecture overview.

## 7. Limitations and Tradeoffs (Declarative Only)

This section describes declarative information about:
- Performance characteristics of different hardware configurations
- Capability boundaries of various platforms
- Tradeoffs between local and cloud execution
- Tradeoffs between local and physical setups
- Tradeoffs between cloud and physical setups
- Resource constraints in edge environments

The section must avoid prescriptive guidance about specific hardware purchases or configurations. All information should be descriptive rather than advisory about which options users should select.

All tradeoffs are framed as contextual considerations, not blockers. The focus avoids cost-based framing of tradeoffs and emphasizes capability-based comparisons.

## 8. Summary and Reader Guidance

This section must include:
- Clear summary of the three hardware tiers
- Explicit statement that hardware choice does not affect learning outcomes
- Encouragement for readers to select environments based on access and goals
- Confirmation that all core curriculum content remains accessible through simulation and cloud environments

## Prohibited Content

This specification explicitly forbids:
- Prices, cost estimates, budget ranges, or cost comparisons
- Setup instructions, installation steps, or configuration guides
- OS, driver, or firmware walkthroughs
- Purchase advice or vendor recommendations framed as requirements
- Shopping lists or procurement guidance
- Imperative phrases like "must buy", "required purchase", or "mandatory hardware"
- Any references to assessments, grading, or learning prerequisites

All content must use capability-based, descriptive language with terms like "may", "can", "is suitable for", "supports" instead of prescriptive language.