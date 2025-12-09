# Specification: Chapter 1 - ROS 2 and the Physical AI Nervous System

## Chapter Description

This chapter introduces beginner to intermediate students to the foundational concepts of ROS2 (Robot Operating System 2) architecture and communication patterns. Students will learn how ROS2 serves as the "nervous system" for robotic systems, enabling different components to communicate effectively. The chapter covers environment setup, basic workspace creation, and implementation of fundamental communication patterns like publisher-subscriber models. This establishes the essential foundation for all future robotic communication in the course.

## Lesson Breakdown

### Lesson 1.1: Introduction to ROS2 Architecture
- **Duration**: 1 hour
- **Topic**: Understanding ROS2 as the communication middleware for humanoid robots
- **Focus**: Core concepts of ROS2 and its evolution from ROS1, DDS communication fundamentals

### Lesson 1.2: Environment Setup and Workspace Creation
- **Duration**: 2 hours
- **Topic**: Installing ROS2 and creating the development environment
- **Focus**: ROS2 Humble Hawksbill installation, workspace structure setup, colcon build system configuration

### Lesson 1.3: Basic Publisher/Subscriber Implementation
- **Duration**: 2 hours
- **Topic**: Creating first ROS2 communication patterns
- **Focus**: Writing "Hello World" publisher and subscriber nodes in Python, launching communication graph

### Lesson 1.4: ROS2 Command Line Tools
- **Duration**: 1.5 hours
- **Topic**: Using ROS2 CLI tools to examine communication
- **Focus**: Using ros2 topic, ros2 node, ros2 service commands, understanding ROS_DOMAIN_ID and network isolation

## Learning Objectives

Upon completion of this chapter, students will be able to:

### Lesson 1.1 Objectives:
- Explain what ROS2 is and its role as a communication middleware in robotic systems
- Understand the concept of ROS2 as a "nervous system" for robots
- Compare ROS2 with ROS1 and understand the evolution
- Describe the DDS (Data Distribution Service) communication model

### Lesson 1.2 Objectives:
- Install ROS2 Humble Hawksbill on Ubuntu 22.04 environment
- Create and configure a ROS2 workspace with proper directory structure
- Set up the development environment with colcon build system
- Verify ROS2 installation with basic commands

### Lesson 1.3 Objectives:
- Write and execute a basic publisher node in Python
- Write and execute a basic subscriber node in Python
- Launch and test a ROS2 communication graph
- Understand the message flow between publisher and subscriber nodes

### Lesson 1.4 Objectives:
- Use ROS2 command-line tools to examine communication patterns
- Understand node status and communication topology
- Work with services and examine service communication
- Understand ROS_DOMAIN_ID and network isolation concepts

## Detailed Lesson Content

### Lesson 1.1: Introduction to ROS2 Architecture

This lesson introduces students to the fundamental concepts of ROS2 architecture. Students will learn that ROS2 is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. We'll use intuitive analogies like the human nervous system to explain how ROS2 enables distributed robotic systems. The lesson will cover the evolution from ROS1 to ROS2, focusing on improvements in security, real-time performance, and multi-robot systems.

Key concepts include:
- Distributed communication patterns
- Node-based architecture
- The role of DDS in enabling reliable communication
- Why ROS2 is essential for Physical AI systems

### Lesson 1.2: Environment Setup and Workspace Creation

This hands-on lesson guides students through setting up their ROS2 development environment. Students will install ROS2 Humble Hawksbill, create their first workspace, and configure the build system. The lesson emphasizes best practices for workspace organization and includes troubleshooting tips for common installation issues. Students will create their first package.xml and setup.py files, establishing a proper development workflow.

Key activities include:
- Installing ROS2 on Ubuntu 22.04
- Creating workspace directory structure in ~/ros2_ws/src
- Setting up colcon build system
- Creating basic package configuration files

### Lesson 1.3: Basic Publisher/Subscriber Implementation

This practical lesson teaches students to implement the most fundamental ROS2 communication pattern: publisher-subscriber. Students will write their first ROS2 nodes in Python, creating a publisher that sends messages and a subscriber that receives them. The lesson emphasizes understanding message flow and the asynchronous nature of topic-based communication.

Key activities include:
- Writing a "Hello World" publisher node
- Writing a "Hello World" subscriber node
- Launching the communication graph
- Testing communication between nodes
- Documenting the communication pattern

### Lesson 1.4: ROS2 Command Line Tools

This lesson focuses on using ROS2's powerful command-line tools to examine and debug communication patterns. Students will learn to use ros2 topic, ros2 node, and ros2 service commands to inspect running systems. The lesson covers network isolation concepts and how ROS_DOMAIN_ID enables multiple ROS2 systems to operate on the same network without interference.

Key activities include:
- Using ros2 topic commands to examine communication
- Using ros2 node commands to examine node status
- Using ros2 service commands to examine services
- Understanding ROS_DOMAIN_ID and network isolation

## Expected Outputs per Lesson

### Lesson 1.1:
- Written summary of ROS2 architecture concepts
- Diagram showing ROS2 as a nervous system analogy
- Notes comparing ROS1 and ROS2 features

### Lesson 1.2:
- Successfully installed ROS2 Humble Hawksbill environment
- Created ROS2 workspace with src directory structure
- Configured colcon build system
- Verified ROS2 installation with basic commands
- Created basic package.xml and setup.py files

### Lesson 1.3:
- Functional publisher node in Python
- Functional subscriber node in Python
- Working communication graph between nodes
- Documented communication pattern and message flow
- Test results showing successful message transmission

### Lesson 1.4:
- Documentation of command usage and outputs
- Results from ros2 topic, ros2 node, and ros2 service commands
- Understanding of ROS_DOMAIN_ID demonstrated through testing
- Network isolation concepts validated through practical exercises

## Required Hardware/Software

### Software Requirements:
- **Operating System**: Ubuntu 22.04 LTS (recommended) or compatible Linux system
- **ROS 2 Distribution**: Humble Hawksbill or later version
- **Python**: Version 3.8 or higher
- **Development Tools**: colcon build system, Git, basic development libraries
- **Text Editor**: VSCode with ROS2 extensions recommended or any code editor

### Hardware Requirements:
- **Processor**: Multi-core processor (Intel i5 or equivalent recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for simulation work)
- **Storage**: 20GB free disk space for ROS2 installation and workspace
- **Network**: Internet connection for package installation and updates

## Dependencies on Module 1 Specs

This chapter directly implements the following elements from the Module 1 specification:

1. **ROS2 middleware implementation**: Establishes the foundational communication infrastructure as specified in Module 1

2. **Node-based architecture**: Introduces distributed system design with isolated processes as outlined in Module 1 specification

3. **Topic-based pub/sub communication**: Implements the first communication pattern (aligns with book-level goal: "Understand and implement fundamental ROS2 communication patterns")

4. **Environment setup requirements**: Implements the workspace and development environment setup specified in Module 1

5. **Python-based ROS2 interfaces**: Introduces rclpy concepts that will be expanded in later chapters (aligns with book-level goal: "Connect Python-based agents with ROS2 controllers using rclpy")

This chapter serves as the foundation for all subsequent chapters in Module 1 and directly supports the Module 1 objectives of introducing students to ROS2 as the communication middleware for humanoid robots.