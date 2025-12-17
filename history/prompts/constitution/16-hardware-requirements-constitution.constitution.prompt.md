---
id: 16
title: Hardware Requirements Constitution
stage: constitution
date_iso: 2025-12-16
model: Claude Opus 4.5
surface: agent
feature: hardware-requirements
branch: main
user: claude
command: Create hardware requirements constitution
labels: ["constitution", "hardware", "requirements", "specification"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/specs/hardware-requirements/constitution.md
tests: []
prompt_text: |
  You are Claude Code CLI operating as a SPEC-DRIVEN-DEVELOPMENT (SDD) Constitution Author.

  Your task is to generate a `constitution.md` file for a standalone, appendix-level section of a technical book titled:

  Hardware Requirements

  This Hardware section is informational only and must NOT affect learning flow, assessment validity, or curriculum scope.

  ========================
  CORE CONSTITUTIONAL RULES
  ========================

  1. NON-NORMATIVE STATUS
  - The Hardware Requirements section is NON-NORMATIVE.
  - No learner is required to own, purchase, or access any listed hardware.
  - No assessment, module, or capstone depends on physical hardware ownership.
  - All book content MUST remain completable using simulation and/or cloud-based environments.

  2. SDD ISOLATION
  - Hardware requirements MUST NOT influence:
    - Module specifications
    - Lesson content
    - Assessment criteria
    - Task definitions
  - This appendix exists purely as execution context, not as instructional logic.

  3. TIERED STRUCTURE (MANDATORY)
  The hardware information MUST be organized into exactly three tiers, in this order:

  Tier 1 — Minimum Requirements

  CPU: Modern multi-core CPU

  RAM: 16–32 GB

  GPU: Any capable of running lightweight simulations

  OS: Linux/Windows/Mac

  Optional: Cloud-based simulation for LLM/VLA tasks

  Tier 2 — Recommended Requirements:

  Hardware Requirements

  This course is technically demanding. It sits at the intersection of three heavy computational loads: Physics Simulation (Isaac Sim/Gazebo), Visual Perception (SLAM/Computer Vision), and Generative AI (LLMs/VLA).

  Because the capstone involves a "Simulated Humanoid," the primary investment must be in High-Performance Workstations. However, to fulfill the "Physical AI" promise, you also need Edge Computing Kits (brains without bodies) or specific robot hardware.

  1. The "Digital Twin" Workstation (Required per Student)

  This is the most critical component. NVIDIA Isaac Sim is an Omniverse application that requires "RTX" (Ray Tracing) capabilities. Standard laptops (MacBooks or non-RTX Windows machines) will not work.

  GPU (The Bottleneck): NVIDIA RTX 4070 Ti (12GB VRAM) or higher.
  Why: You need high VRAM to load the USD (Universal Scene Description) assets for the robot and environment, plus run the VLA (Vision-Language-Action) models simultaneously.
  Ideal: RTX 3090 or 4090 (24GB VRAM) allows for smoother "Sim-to-Real" training.

  CPU: Intel Core i7 (13th Gen+) or AMD Ryzen 9.
  Why: Physics calculations (Rigid Body Dynamics) in Gazebo/Isaac are CPU-intensive.

  RAM: 64 GB DDR5 (32 GB is the absolute minimum, but will crash during complex scene rendering).

  OS: Ubuntu 22.04 LTS.
  Note: While Isaac Sim runs on Windows, ROS 2 (Humble/Iron) is native to Linux. Dual-booting or dedicated Linux machines are mandatory for a friction-free experience.

  Components: Jetson Orin Nano, RealSense camera, microphone

  Software: Gazebo, Isaac Sim

  Tier 3 — Optional / Premium (Advanced or Physical AI)

  1. The "Physical AI" Edge Kit

  Since a full humanoid robot is expensive, students learn "Physical AI" by setting up the nervous system on a desk before deploying it to a robot. This kit covers Module 3 (Isaac ROS) and Module 4 (VLA).

  The Brain: NVIDIA Jetson Orin Nano (8GB) or Orin NX (16GB).

  Role: This is the industry standard for embodied AI. Students will deploy their ROS 2 nodes here to understand resource constraints vs. their powerful workstations.

  The Eyes (Vision): Intel RealSense D435i or D455.

  Role: Provides RGB (Color) and Depth (Distance) data. Essential for the VSLAM and Perception modules.

  The Inner Ear (Balance): Generic USB IMU (BNO055) (Often built into the RealSense D435i or Jetson boards, but a separate module helps teach IMU calibration).

  Voice Interface: A simple USB Microphone/Speaker array (e.g., ReSpeaker) for the "Voice-to-Action" Whisper integration.


  Each higher tier MAY include all capabilities of the previous tier.


  2. The Robot Lab

  For the "Physical" part of the course, you have three tiers of options depending on budget.

  Option A: The "Proxy" Approach (Recommended for Budget)

  Use a quadruped (dog) or a robotic arm as a proxy. The software principles (ROS 2, VSLAM, Isaac Sim) transfer 90% effectively to humanoids.

  Robot: Unitree Go2 Edu etc.
  Pros: Highly durable, excellent ROS 2 support, affordable enough to have multiple units.
  Cons: Not a biped (humanoid).

  Option B: The "Miniature Humanoid" Approach

  Small, table-top humanoids.

  Robot: Unitree H1, Unitree G1 or Robotis OP3 (older, but stable) etc.
  Budget Alternative: Hiwonder TonyPi Pro.
  Warning: The cheap kits (Hiwonder) usually run on Raspberry Pi, which cannot run NVIDIA Isaac ROS efficiently. You would use these only for kinematics (walking) and use the Jetson kits for AI.

  Option C: The "Premium" Lab (Sim-to-Real specific)

  If the goal is to actually deploy the Capstone to a real humanoid:

  Robot: Unitree G1 Humanoid etc.
  Why: It is one of the few commercially available humanoids that can actually walk dynamically and has an SDK open enough for students to inject their own ROS 2 controllers.

  4. Summary of Architecture
  To teach this successfully, your lab infrastructure should look like this:
  Component
  Hardware
  Function
  Sim Rig
  PC with RTX 4080 + Ubuntu 22.04
  Runs Isaac Sim, Gazebo, Unity, and trains LLM/VLA models.
  Edge Brain
  Jetson Orin Nano
  Runs the "Inference" stack. Students deploy their code here.
  Sensors
  RealSense Camera + Lidar
  Connected to the Jetson to feed real-world data to the AI.
  Actuator
  Unitree Go2 or G1 (Shared)
  Receives motor commands from the Jetson.


  If you do not have access to RTX-enabled workstations, we must restructure the course to rely entirely on cloud-based instances (like AWS RoboMaker or NVIDIA's cloud delivery for Omniverse), though this introduces significant latency and cost complexity.

  Building a "Physical AI" lab is a significant investment. You will have to choose between building a physical On-Premise Lab at Home (High CapEx) versus running a Cloud-Native Lab (High OpEx).

  Option 2 High OpEx: The "Ether" Lab (Cloud-Native)

  Best for: Rapid deployment, or students with weak laptops.

  1. Cloud Workstations (AWS/Azure) Instead of buying PCs, you rent instances.
  Instance Type: AWS g5.2xlarge (A10G GPU, 24GB VRAM) or g6e.xlarge.
  Software: NVIDIA Isaac Sim on Omniverse Cloud (requires specific AMI).
  Usage: 10 hours/week × 12 weeks = 120 hours.
  Storage (EBS volumes for saving environments):
  Total Cloud Bill: ~$205 per quarter.

  2. Local "Bridge" Hardware You cannot eliminate hardware entirely for "Physical AI." You still need the edge devices to deploy the code physically.
  Edge AI Kits: You still need the Jetson Kit for the physical deployment phase.
  Cost: (One-time purchase).
  Robot: You still need one physical robot for the final demo.
  (Unitree Go2 Standard etc).
  The Economy Jetson Student Kit
  Best for: Learning ROS 2, Basic Computer Vision, and Sim-to-Real control.
  Component
  Model
  Price (Approx.)
  Notes
  The Brain
  NVIDIA Jetson Orin Nano Super Dev Kit (8GB)
  New official MSRP. Capable of 40 TOPS.
  The Eyes
  Intel RealSense D435i
  Includes IMU (essential for SLAM). Do not buy the D435 (non-i).
  The Ears
  ReSpeaker USB Mic Array v2.0
  Far-field microphone for voice commands (Module 4).
  Wi-Fi
  (Included in Dev Kit)
  The new "Super" kit includes the Wi-Fi module pre-installed.
  Power/Misc
  SD Card (128GB) + Jumper Wires
  High-endurance microSD card required for the OS.

  3. The Latency Trap (Hidden Cost)
  Simulating in the cloud works well, but controlling a real robot from a cloud instance is dangerous due to latency.
  Solution: Students train in the Cloud, download the model (weights), and flash it to the local Jetson kit.

  4. CAPABILITY-BASED LANGUAGE
  - Describe hardware in terms of CAPABILITIES, not purchasing instructions.
  - Brand or model examples MAY be mentioned only as non-binding references.
  - Avoid prescriptive language such as "must buy", "required purchase", or "mandatory ownership".

  5. PRICE PROHIBITION (STRICT)
  - NO prices, cost estimates, budgets, or monetary values are allowed.
  - NO CapEx vs OpEx comparisons.
  - NO purchasing advice or cost optimization strategies.

  6. PHYSICAL AI BOUNDARY
  - Physical robots, edge devices, and sensors may be described ONLY as:
    - Optional
    - Advanced
    - For sim-to-real experimentation
  - The constitution MUST clearly state that "Physical AI" is an extension, not a requirement.

  7. CLOUD EQUIVALENCE GUARANTEE
  - The constitution MUST explicitly guarantee that:
    - Cloud-based GPU workstations are valid substitutes for local high-performance machines.
    - Simulation-first workflows are fully supported.
  - Latency or performance tradeoffs may be acknowledged, but never framed as blockers.

  8. NO SETUP OR TUTORIAL CONTENT
  - The constitution MUST prohibit:
    - Installation steps
    - Configuration guides
    - OS setup instructions
    - Driver or firmware guidance
  These belong outside the Hardware section.

  9. FUTURE-PROOFING
  - Hardware descriptions MUST be resilient to technological change.
  - Avoid anchoring learning outcomes to specific GPU generations or vendor roadmaps.

  ========================
  SCOPE ANCHORS (CONTEXT)
  ========================

  This book involves:
  - Physics-based simulation (Gazebo, Isaac Sim)
  - Visual perception (SLAM, computer vision)
  - Vision–Language–Action systems using LLMs
  - A simulated humanoid capstone project
  - Optional Physical AI via edge devices and robots

  The Hardware section exists to help readers understand execution environments — not to restrict them.

  ========================
  OUTPUT REQUIREMENTS
  ========================

  - Output ONLY valid `constitution.md` content.
  file path: physical-ai-humanoid-robotics-book/specs/constitution.md
  - Use clear constitutional language (rules, guarantees, prohibitions).
  - Do NOT include implementation details.
  - Do NOT include prices.
  - Do NOT include tables of products.
  - Do NOT reference assessments or grading logic.

  The constitution must make it impossible for future specifications, plans, or tasks to accidentally turn hardware into a requirement.

response_text: |
  Successfully created the Hardware Requirements constitution file at physical-ai-humanoid-robotics-book/specs/hardware-requirements/constitution.md. The constitution includes:

  1. Non-normative status ensuring hardware is informational only
  2. SDD isolation to prevent hardware requirements from influencing curriculum
  3. Tiered structure with Minimum, Recommended, and Optional/Premium tiers
  4. Capability-based descriptions instead of purchasing instructions
  5. Cloud equivalence guarantee ensuring simulation-first workflows are supported
  6. Prohibition of prices, cost estimates, and purchasing advice
  7. Clear boundaries for Physical AI as optional extension

  The constitution ensures that all book content remains completable using simulation and cloud-based environments, with no learner required to own, purchase, or access any specific hardware.

outcome: |
  Hardware Requirements constitution created with proper tiered structure and non-normative status, ensuring all book content remains accessible through simulation and cloud environments.

evaluation: |
  The constitution successfully meets all requirements by maintaining non-normative status, ensuring simulation-first workflows are supported, and preventing hardware ownership from becoming a requirement for course completion.
---