# Chapter 3 – Robot Description (URDF/Xacro) and Embodiment

## Chapter Description

Chapter 3 focuses on creating URDF (Unified Robot Description Format) models for humanoid robot kinematics and using Xacro to parameterize robot descriptions. Students will learn to define the physical form and kinematic structure of a humanoid robot, visualize robot models in RViz and Gazebo, and validate kinematic chain definitions. This chapter bridges the gap between abstract robot concepts and physical embodiment, enabling students to describe a robot's body parts and their interconnections, which is essential for simulation, visualization, motion planning, and control.

## Learning Objectives

- Understand the purpose and structure of URDF for robot description
- Create URDF models to define a robot's links, joints, and physical properties
- Utilize Xacro for modular and parametric URDF generation
- Visualize URDF models in ROS2 tools
- Validate kinematic chain definitions for correctness
- Integrate URDF models with Robot State Publisher for visualization

## Lessons Breakdown

### Lesson 3.1 – Basic URDF Robot Description

- **Objective**: Create basic robot URDF with links and joints, defining the fundamental structure of a humanoid robot
- **Scope**: Define base link and fundamental robot structure in URDF, add joints and connected links to form kinematic chain, create complete URDF file for humanoid robot
- **Expected Outcome**: Students will produce a basic URDF file that represents the skeleton of a humanoid robot with proper kinematic chains
- **Tools**: URDF, ROS2, Robot State Publisher

### Lesson 3.2 – Xacro Parameterization

- **Objective**: Convert URDF to parameterized Xacro files using macros for modular robot description
- **Scope**: Convert URDF to parameterized Xacro files, create Xacro macros for modular robot description, add parameter definitions for robot components, generate complete URDF from Xacro files
- **Expected Outcome**: Students will create parameterized Xacro files that allow for reusable and configurable robot descriptions
- **Tools**: Xacro, URDF, ROS2

### Lesson 3.3 – Visualization and Validation

- **Objective**: Visualize robot models in RViz and validate kinematic properties to ensure correctness
- **Scope**: Test URDF with Robot State Publisher, visualize robot in RViz and verify kinematics, add visual and collision properties to robot links, create visualization launch files
- **Expected Outcome**: Students will validate their robot descriptions and be able to visualize the robot in ROS2 tools
- **Tools**: RViz, Robot State Publisher, URDF, Xacro

## Chapter Dependencies

- **Relation to Chapter 2**: This chapter builds upon the communication patterns learned in Chapter 2 by providing the robot description that will be used by nodes to understand the robot's physical structure. The URDF model will be published via Robot State Publisher, which will be consumed by other nodes for visualization and planning.
- **Preparation for Chapter 4**: This chapter prepares students for Chapter 4 by providing the complete robot model that will be used in simulation environments. The URDF/Xacro files created here will be essential for connecting Python-based agents with simulated robot controllers in Gazebo.