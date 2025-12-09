---
title: Lesson 1.1 - Introduction to ROS2 Architecture
---

# Lesson 1.1 – Introduction to ROS2 Architecture

## Learning Objective
Understand what ROS2 is and its role as a communication middleware in robotic systems, compare ROS2 with ROS1 and understand the evolution, describe the DDS (Data Distribution Service) communication model

## Conceptual Scope
This lesson introduces the fundamental concepts of ROS2 architecture. ROS2 is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. We'll use intuitive analogies like the human nervous system to explain how ROS2 enables distributed robotic systems. The lesson covers the evolution from ROS1 to ROS2, focusing on improvements in security, real-time performance, and multi-robot systems.

This lesson directly supports the module specification requirement for "ROS2 middleware implementation for robot communication" and "Node-based architecture for distributed robot control." It also addresses the book-level goal of "Understand and implement fundamental ROS2 communication patterns step-by-step" by establishing the foundational understanding of ROS2 architecture.

The concept of ROS2 as a "nervous system" is particularly apt because, like the biological nervous system, it allows for distributed processing while maintaining coordination. Different "nodes" (software components) can run on different computers, yet communicate seamlessly as if they were part of a single system. This architecture is fundamental to the implementation of the core ROS2 entities specified in the module specification, including nodes, topics, services, and parameters.

The core innovation of ROS2 is its use of DDS (Data Distribution Service) as the underlying communication middleware, which provides a standardized way for different software components to exchange data. This addresses the module specification requirement for "Node-based architecture for distributed robot control" and "Topic-based pub/sub communication patterns."

Key concepts include:
- Distributed communication patterns
- Node-based architecture
- The role of DDS in enabling reliable communication
- Why ROS2 is essential for Physical AI systems

## Expected Learning Outcome
By the end of this lesson, you will understand what ROS2 is and its role as a communication middleware in robotic systems, and you'll be able to compare ROS2 with ROS1 and understand the evolution.

## Tools / References
- ROS2
- DDS (Data Distribution Service)