---
title: Lesson 1.2 - Environment Setup and Workspace Creation
---

# Lesson 1.2 – Environment Setup and Workspace Creation

## Learning Objective
Install ROS2 Humble Hawksbill on Ubuntu 22.04 environment, create and configure a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, verify ROS2 installation with basic commands

## Conceptual Scope
This hands-on lesson guides you through setting up your ROS2 development environment. You'll install ROS2 Humble Hawksbill, create your first workspace, and configure the build system. The lesson emphasizes best practices for workspace organization and includes troubleshooting tips for common installation issues. You'll create your first package.xml and setup.py files, establishing a proper development workflow.

This lesson implements the module specification requirement for "environment setup for future development" and supports the "Python-based ROS2 control interfaces using rclpy" objective. It directly addresses the module completion criteria of "All ROS2 communication patterns (topics, services, actions) must be implemented and tested" by establishing the foundational development environment.

The ROS2 workspace follows a standard structure with a `src` directory where you'll place your packages. The `colcon` build system is used to compile your ROS2 packages. This setup is crucial as it establishes the foundation for all future development in this module and aligns with the module specification requirement for "Simulation-ready abstractions for Gazebo/Isaac/Unity compatibility."

Key activities include:
- Installing ROS2 on Ubuntu 22.04
- Creating workspace directory structure in ~/ros2_ws/src
- Setting up colcon build system
- Creating basic package configuration files

This lesson also supports the book-level technical competency of "Create basic ROS2 nodes, topics, services, and parameters for inter-process communication" by establishing the proper development environment needed to create these components.

## Expected Learning Outcome
By the end of this lesson, you will have successfully installed ROS2 Humble Hawksbill on Ubuntu 22.04 environment, created and configured a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, and verified ROS2 installation with basic commands.

## Tools / References
- ROS2 Humble Hawksbill
- colcon build system
- Ubuntu 22.04