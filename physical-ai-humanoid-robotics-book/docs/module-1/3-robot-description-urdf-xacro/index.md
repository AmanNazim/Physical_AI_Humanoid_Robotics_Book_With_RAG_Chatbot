---
title: Chapter 3 Introduction - Robot Description (URDF/Xacro) and Embodiment
---

# Chapter 3 Introduction – Robot Description (URDF/Xacro) and Embodiment

## Overview

Welcome to Chapter 3 of Module 1: The Robotic Nervous System. In this chapter, we will explore the fundamental concepts of robot description using URDF (Unified Robot Description Format) and Xacro (XML Macros). This chapter is crucial for understanding how to represent the physical structure of humanoid robots in a way that can be processed by ROS2 systems.

## Chapter Context and Importance

In the previous chapters, we learned about ROS2 as the communication middleware and how nodes, topics, services, and parameters enable distributed robotic systems. Now, we need to understand how to describe the physical robot itself – its links, joints, and overall structure. This is where URDF and Xacro come into play.

URDF (Unified Robot Description Format) is an XML-based format that describes robot models. It defines the physical structure of a robot, including links (rigid parts), joints (connections between links), and other properties such as visual and collision geometry. Xacro (XML Macros) is an XML macro language that allows us to create more complex and reusable URDF files through parameterization and macro definitions.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the purpose and structure of URDF for robot description
- Create URDF models to define a robot's links, joints, and physical properties
- Utilize Xacro for modular and parametric URDF generation
- Visualize URDF models in ROS2 tools
- Validate kinematic chain definitions for correctness
- Integrate URDF models with Robot State Publisher for visualization

## Chapter Structure

This chapter is organized into three lessons that build upon each other:

1. **Lesson 3.1 – Basic URDF Robot Description**: You'll learn to create basic robot URDF with links and joints, defining the fundamental structure of a humanoid robot.

2. **Lesson 3.2 – Xacro Parameterization**: You'll convert URDF to parameterized Xacro files using macros for modular robot description.

3. **Lesson 3.3 – Visualization and Validation**: You'll visualize robot models in RViz and validate kinematic properties to ensure correctness.

## Prerequisites

Before starting this chapter, you should have:
- A working ROS2 environment (covered in Chapter 1)
- Understanding of ROS2 communication patterns (covered in Chapter 2)
- Basic knowledge of XML syntax (helpful but not required)

## Why This Matters for Physical AI

The ability to accurately describe a robot's physical structure is fundamental to Physical AI. Without proper robot description:
- Simulation environments cannot accurately model robot behavior
- Motion planning algorithms cannot understand the robot's kinematic constraints
- Controllers cannot properly command the robot's joints
- Visualization tools cannot display the robot correctly

In Physical AI and humanoid robotics, the robot's embodiment – its physical form – directly impacts its capabilities and behaviors. URDF and Xacro provide the bridge between abstract robot concepts and concrete physical models that can be used in real applications.

## What You'll Build

By the end of this chapter, you will have created:
- A complete URDF file representing a simplified humanoid robot
- Parameterized Xacro files that allow for reusable and configurable robot descriptions
- Visualization configurations that allow you to see your robot in RViz
- Validation scripts to ensure your robot model is kinematically correct

## Tools and Technologies

In this chapter, we will work with:
- URDF (Unified Robot Description Format)
- Xacro (XML Macros)
- ROS2 Robot State Publisher
- RViz for visualization

## Chapter Dependencies

This chapter builds upon the communication patterns learned in Chapter 2 by providing the robot description that will be used by nodes to understand the robot's physical structure. The URDF model will be published via Robot State Publisher, which will be consumed by other nodes for visualization and planning.

This chapter also prepares you for Chapter 4, where you will connect Python-based agents with simulated robot controllers in Gazebo, using the complete robot models you'll develop here.