---
id: 001
title: Module 1 Constitution: ROS2 Foundations
stage: constitution
date: 2025-12-06
surface: agent
model: claude-sonnet-4-5-20250929
feature: module-1
branch: master
user: unknown
command: /sp.constitution
labels: ["robotics", "curriculum", "ros2", "constitution"]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - physical-ai-humanoid-robotics-book/specs/module-1/constitution.md
tests: null
---

## Prompt

```text
You are an expert curriculum designer, robotics engineer, and textbook author.

BOOK CONTEXT:
This book is titled: "Physical AI & Humanoid Robotics".
Audience: Advanced undergraduate / early graduate students.
Reader background: Python programming, basic linear algebra, basic robotics concepts.
Course duration: 13 weeks.
Goal: Make students capable of building, simulating, and deploying humanoid robots using ROS2, Gazebo, Unity/NVIDIA Isaac, and Vision-Language-Action AI systems.

MODULE 1 CONTEXT:
Module 1 Theme: "The Robotic Nervous System – ROS2 Foundations for Physical AI"
This module introduces the core software nervous system that connects perception, intelligence, and actuation in humanoid robots.

YOUR TASK:
Create a production-ready `constitution.md` file for **Module 1**, and also define the **four core lessons of this module**.

CONTENT REQUIREMENTS:

1. Begin with a powerful 1–2 paragraph **vision statement** explaining why ROS2 is the nervous system of physical AI and humanoid robots.

2. Provide **8–12 precise learning objectives** written as measurable outcomes
   (example style: “Students will be able to design…”, “Students will be able to implement…”).

3. Include a section titled:
   ## Why This Module Matters for Physical AI
   Explain real-world relevance: humanoid robots, autonomy stacks, perception-to-action pipelines, industry usage.

4. Include a section titled:
   ## Hardware–Software Mindset
   Explain how software architecture directly shapes humanoid motion, perception, decision-making, and safety.

5. Include a section titled:
   ## What Students Will Build by the End of This Module
   List **4–6 tangible outcomes** (systems, simulations, ROS2 architectures, data pipelines, control systems).

6. Include a section titled:
   ## Mental Models to Master
   List **5–7 deep conceptual shifts** students must internalize about physical AI and robotic software systems.

7. Include a section titled:
   ## Module 1 Lesson Structure
   Define exactly **four lessons**.
   For each lesson, provide:
   - Lesson title
   - 3–4 bullet learning goals
   - 1 paragraph summary of what the student will understand or build

   The four lessons should logically follow this progression:
   Focus: Middleware for robot control
   - Lesson 1 → ROS 2 and the Physical AI nervous system
   - Lesson 2 → ROS 2 Nodes, Topics, Services, and robot communication
   - Lesson 3 → Robot description (URDF/Xacro) and embodiment
   - Lesson 4 → Bridging Python-based Agents to ROS2 controllers using rclpy and simulation readiness
Success criteria:
- Explains ROS 2 core concepts clearly.
- Demonstrates creating and running ROS 2 nodes and packages.
- Shows communication via topics and services.
- Includes python integration examples using rclpy.
- Includes URDF examples for humanoid robots.

Format: Markdown, with diagrams and runnable code snippets.

FORMAT RULES:
- Output must be **pure Markdown**
- Use proper Markdown headers (##, ###)
- Use bullet points where appropriate
- No emojis
- No marketing tone
- Maintain **academic + engineering professionalism**
- Do NOT include implementation steps, code, commands, or tools yet
- This file is ONLY about vision, learning goals, conceptual structure, and lesson framing

OUTPUT:
Return only the complete `constitution.md` content for Module 1.
```

## Response snapshot

The `physical-ai-humanoid-robotics-book/specs/module-1/constitution.md` file was created, detailing the vision, learning objectives, relevance to physical AI, hardware-software mindset, tangible student outcomes, mental models to master, and a structured outline of the four lessons for Module 1. Each lesson includes a title, learning goals, and a summary, adhering to the specified progression and academic tone.

## Outcome

- ✅ Impact: Created module-specific constitution, defining vision, objectives, and lesson structure.
- 🧪 Tests: N/A
- 📁 Files: New file `physical-ai-humanoid-robotics-book/specs/module-1/constitution.md` created.
- 🔁 Next prompts: Next step would be to create the spec for Module 1.
- 🧠 Reflection: Successfully generated constitution for Module 1 based on detailed requirements.

## Evaluation notes (flywheel)

- Failure modes observed: Script `create-phr.sh` not found, fell back to agent-native PHR creation.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
