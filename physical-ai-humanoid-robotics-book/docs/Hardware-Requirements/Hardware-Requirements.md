# Hardware Requirements

## Purpose and Non-Normative Disclaimer

The Hardware Requirements section serves as a non-normative reference for readers to understand the computational environments that support the Physical AI Humanoid Robotics curriculum. This section provides capability-based guidance for execution environments while ensuring that all book content remains completable using simulation and cloud-based environments.

This section is purely informational and does not affect learning flow, assessment validity, or curriculum scope. No learner is required to own, purchase, or access any listed hardware. All book content and assessments are completable via simulation and/or cloud-based environments. This appendix does not impose learning or grading constraints.

## Tier 1 — Minimum Requirements

The minimum requirements describe the lowest-capability environment sufficient for engaging with the book content:

- **CPU**: Modern multi-core CPU that can handle basic development tasks and educational content consumption
- **RAM**: 16–32 GB of memory for standard development workloads
- **GPU**: Integrated graphics or entry-level discrete GPU capable of running lightweight simulations
- **OS**: Linux, Windows, or macOS operating systems
- **Optional**: Cloud-based simulation environments for LLM/VLA tasks

This tier does not assume access to high-end discrete GPUs or physical devices. The environment should support basic development tasks and educational content consumption, emphasizing accessibility and flexibility for readers who wish to engage with the theoretical concepts and lighter computational workloads.

## Tier 2 — Recommended Requirements

The recommended requirements describe a local development environment suitable for intensive computational workloads common in robotics:

- **GPU**: RTX-capable GPU with at least 12GB VRAM (such as NVIDIA RTX 4070 Ti or higher) to handle USD assets and Vision-Language-Action (VLA) models simultaneously
- **CPU**: High-performance multi-core processor (such as Intel Core i7 13th Gen+ or AMD Ryzen 9) for physics calculations and complex computations
- **RAM**: 64 GB DDR5 for complex scene rendering and multi-process workloads involving simultaneous simulation and AI processing
- **OS**: Ubuntu 22.04 LTS for optimal ROS 2 compatibility and simulation framework support
- **Software**: Access to Gazebo, Isaac Sim, and similar simulation environments

This tier represents capabilities suitable for physics-based simulation, visual perception pipelines, and Vision-Language-Action workflows. The requirements are described in terms of capability classes rather than specific purchases, allowing users to select equivalent alternatives based on availability and access.

## Tier 3 — Optional / Premium (Advanced or Physical AI)

The optional/premium tier describes environments for advanced applications and sim-to-real workflows:

- **Edge devices**: NVIDIA Jetson platforms (such as Orin Nano or Orin NX) for deployment to resource-constrained environments, described as illustrative examples only
- **Sensors**: Depth cameras (such as Intel RealSense series), IMUs, and microphones for perception and interaction, described as illustrative examples only
- **Robots**: Humanoid or quadruped platforms for physical deployment, described as illustrative examples only

This tier is explicitly optional and represents advanced applications beyond the core curriculum. Physical hardware is not required for course completion, and these represent extensions for readers interested in sim-to-real experimentation and edge deployment scenarios.

## Cloud and Remote Execution Equivalence

Cloud-based GPU environments serve as valid alternatives to local machines. Simulation-first workflows are fully supported in cloud environments, providing equivalent functionality to local setups at a conceptual level.

Performance and latency considerations may exist in cloud environments, but these should not be considered blockers to completing the curriculum. Cloud options provide an alternative path for users without access to high-end local hardware, enabling access to the full range of computational capabilities needed for the course content.

Cloud execution environments can provide the necessary GPU resources for physics simulation, visual perception, and VLA workflows, allowing readers to engage with the full curriculum without local hardware investments.

## Conceptual Architecture Overview

The computational architecture for robotics applications involves several key system roles:

- **Simulation Systems**: High-performance workstations for Isaac Sim, Gazebo, Unity, and training of LLM/VLA models
- **Inference / AI Execution**: Systems for executing AI models, including VLA processing and perception pipelines
- **Sensing**: Cameras, IMUs, and other perception hardware for collecting environmental data
- **Actuation**: Robot platforms for physical interaction and control

These roles represent functional categories rather than specific implementation requirements. Different computational environments may combine or distribute these roles differently while maintaining equivalent capabilities.

## Limitations, Tradeoffs, and Warnings

Different computational environments present various tradeoffs and considerations:

- **Local vs. Cloud Setups**: Local machines offer lower latency and direct hardware control, while cloud environments provide scalable resources and eliminate hardware maintenance
- **Local vs. Physical Setups**: Simulation environments enable rapid iteration and safe experimentation, while physical robots provide real-world validation but introduce safety and debugging complexities
- **Cloud vs. Physical Setups**: Cloud simulation offers unlimited scalability but may introduce network latency when interfacing with physical systems

Resource constraints in edge environments may limit the complexity of AI models that can be deployed, requiring optimization techniques that differ from workstation-scale computing.

All tradeoffs should be considered as contextual factors in choosing computational environments rather than barriers to achieving learning objectives.

## Summary and Reader Guidance

The hardware requirements are organized into three tiers:

- **Tier 1**: Minimum capabilities for basic engagement with book content
- **Tier 2**: Recommended capabilities for full simulation and perception workflows
- **Tier 3**: Optional capabilities for advanced physical AI applications

Hardware choice does not affect learning outcomes, as all core curriculum content remains accessible through simulation and cloud environments. Readers are encouraged to select computational environments based on their access and goals, with the assurance that all book content and assessments can be completed regardless of specific hardware choices.

The course materials are designed to be flexible with respect to execution environment, allowing readers to progress through the curriculum using the computational resources available to them.