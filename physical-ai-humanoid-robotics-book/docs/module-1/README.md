# Module 1: The Robotic Nervous System – ROS2 Foundations for Physical AI

## Overview

Welcome to the foundational module of Physical AI and humanoid robotics! This module introduces you to ROS2 (Robot Operating System 2), the communication framework that serves as the "nervous system" for robotic systems. Think of ROS2 as the infrastructure that allows different parts of a robot to communicate with each other - just like how your nervous system allows different parts of your body to coordinate.

In this module, we'll take a step-by-step approach to understanding how robots communicate internally. You'll learn how to set up the ROS2 environment, create communication pathways between different robot components, describe robot structure, and connect AI algorithms to robot controllers. No prior robotics knowledge is required - we'll build concepts from the ground up with intuitive examples.

This module is designed specifically for beginner to intermediate students and focuses on practical, hands-on learning. You'll start with simple concepts and gradually build toward more sophisticated implementations, creating a complete communication framework for a simulated humanoid robot.

## Learning Objectives

Upon completion of this module, beginner to intermediate students will be able to:

- Explain the core architectural components of ROS2 and their roles in a robotic system
- Create and test basic ROS2 nodes, topics, services, and parameters for inter-process communication
- Use `rclpy` to connect Python-based AI agents and control algorithms with ROS2
- Create and understand basic Unified Robot Description Format (URDF) files for humanoid robot embodiment
- Configure ROS2 workspaces and build systems for development
- Simulate basic robot behaviors within a Gazebo or similar environment using ROS2 interfaces
- Identify and fix common ROS2 communication issues in simple robotic setups
- Understand the advantages of a distributed middleware like ROS2 for physical AI applications
- Recognize the significance of robust software architecture in ensuring robot safety and reliability

## Core Technologies and System Architecture

This module covers the fundamental technologies that form the backbone of modern robotic systems:

- **ROS 2 (Robot Operating System 2)**: The communication middleware that enables distributed robotic systems through its DDS-based architecture
- **Node-based Architecture**: Distributed system design with isolated processes encapsulating robot functionality
- **Communication Patterns**: Topic-based pub/sub, service-based request/response, and action-based goal-oriented communication
- **Parameter Management**: Configuration system for robot parameters and settings
- **URDF/Xacro**: Unified Robot Description Format for defining robot kinematics, geometry, and sensor placement
- **Python Integration**: Using `rclpy` to connect Python-based AI agents and control algorithms with ROS2
- **Simulation Readiness**: Abstraction layers for Gazebo/Isaac/Unity compatibility

The logical software architecture of a humanoid robot ROS2 system follows a distributed node-based pattern with three primary layers:

### Perception Layer
- Sensor nodes publish raw and processed data
- Camera, IMU, joint encoders, force/torque sensors
- Data flows to processing nodes for interpretation

### Cognition Layer
- Processing nodes interpret sensor data
- Decision-making algorithms operate on processed information
- Planning nodes generate action commands

### Actuation Layer
- Control nodes execute motor commands
- Joint controllers manage physical movement
- Feedback systems monitor execution status

### Data Flow Pattern
Data flows from perception → cognition → actuation through standardized ROS2 topics. Each layer communicates asynchronously via message passing, enabling modularity and fault tolerance. Inter-module boundaries are defined by message interface contracts that future AI/VLA modules must adhere to for compatibility.

## Hardware/Software Requirements

Students will need to prepare their development environment with the following requirements:

- **Operating System**: Ubuntu 22.04 LTS (recommended) or compatible Linux system
- **ROS 2 Distribution**: Humble Hawksbill or later version
- **Python**: Version 3.8 or higher
- **Development Tools**: colcon build system, Git, basic development libraries
- **Memory**: 8GB RAM minimum recommended for simulation work
- **Simulation Environment**: Gazebo for robot simulation and testing

## What You Will Build

By the end of this module, students will have created:

- A functional ROS2 communication graph for a simulated humanoid robot
- Basic ROS2 packages for sensor data publishing and motor command subscription
- Basic URDF/Xacro models representing simplified humanoid robot kinematics
- Python-based ROS2 nodes that interface with a simulated robot's controllers using `rclpy`
- A basic simulation environment in Gazebo demonstrating ROS2 control of a humanoid robot
- A modular software architecture for elementary perception-to-action tasks

This module emphasizes hands-on learning with beginner-friendly examples, fostering a mindset where architectural choices are made with physical embodiment and real-world interaction in mind.