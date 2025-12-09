---
title: Chapter 1 – ROS 2 and the Physical AI Nervous System
---

# Chapter 1 – ROS 2 and the Physical AI Nervous System

## Introduction

This chapter introduces beginner to intermediate students to the foundational concepts of ROS2 (Robot Operating System 2) architecture and communication patterns. You will learn how ROS2 serves as the "nervous system" for robotic systems, enabling different components to communicate effectively. The chapter covers environment setup, basic workspace creation, and implementation of fundamental communication patterns like publisher-subscriber models. This establishes the essential foundation for all future robotic communication in the course.

ROS2 is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. Think of ROS2 as the infrastructure that allows different parts of a robot to communicate with each other - just like how your nervous system allows different parts of your body to coordinate.

In this chapter, you'll take a step-by-step approach to understanding how robots communicate internally. You'll learn how to set up the ROS2 environment, create communication pathways between different robot components, and implement the most fundamental communication pattern: publisher-subscriber. No prior robotics knowledge is required - we'll build concepts from the ground up with intuitive examples.

This chapter is designed specifically for beginner to intermediate students and focuses on practical, hands-on learning. You'll start with simple concepts and gradually build toward more sophisticated implementations, creating your first ROS2 communication graph.

## Lessons

### Lesson 1.1 – Introduction to ROS2 Architecture
- **Objective**: Understand what ROS2 is and its role as a communication middleware in robotic systems, compare ROS2 with ROS1 and understand the evolution, describe the DDS (Data Distribution Service) communication model
- **Scope**: This lesson introduces the fundamental concepts of ROS2 architecture. ROS2 is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. We'll use intuitive analogies like the human nervous system to explain how ROS2 enables distributed robotic systems. The lesson covers the evolution from ROS1 to ROS2, focusing on improvements in security, real-time performance, and multi-robot systems.
- **Outcome**: By the end of this lesson, you will understand what ROS2 is and its role as a communication middleware in robotic systems, and you'll be able to compare ROS2 with ROS1 and understand the evolution.
- **Tools**: ROS2, DDS (Data Distribution Service)

### Lesson 1.2 – Environment Setup and Workspace Creation
- **Objective**: Install ROS2 Humble Hawksbill on Ubuntu 22.04 environment, create and configure a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, verify ROS2 installation with basic commands
- **Scope**: This hands-on lesson guides you through setting up your ROS2 development environment. You'll install ROS2 Humble Hawksbill, create your first workspace, and configure the build system. The lesson emphasizes best practices for workspace organization and includes troubleshooting tips for common installation issues. You'll create your first package.xml and setup.py files, establishing a proper development workflow.
- **Outcome**: By the end of this lesson, you will have successfully installed ROS2 Humble Hawksbill on Ubuntu 22.04 environment, created and configured a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, and verified ROS2 installation with basic commands.
- **Tools**: ROS2 Humble Hawksbill, colcon build system, Ubuntu 22.04

### Lesson 1.3 – Basic Publisher/Subscriber Implementation
- **Objective**: Write and execute a basic publisher node in Python, write and execute a basic subscriber node in Python, launch and test a ROS2 communication graph, understand the message flow between publisher and subscriber nodes
- **Scope**: This practical lesson teaches you to implement the most fundamental ROS2 communication pattern: publisher-subscriber. You'll write your first ROS2 nodes in Python, creating a publisher that sends messages and a subscriber that receives them. The lesson emphasizes understanding message flow and the asynchronous nature of topic-based communication.
- **Outcome**: By the end of this lesson, you will have written and executed a basic publisher node in Python, written and executed a basic subscriber node in Python, launched and tested a ROS2 communication graph, and understood the message flow between publisher and subscriber nodes.
- **Tools**: Python 3.8+, rclpy, ROS2

### Lesson 1.4 – ROS2 Command Line Tools
- **Objective**: Use ROS2 command-line tools to examine communication patterns, understand node status and communication topology, work with services and examine service communication, understand ROS_DOMAIN_ID and network isolation concepts
- **Scope**: This lesson focuses on using ROS2's powerful command-line tools to examine and debug communication patterns. You'll learn to use `ros2 topic`, `ros2 node`, and `ros2 service` commands to inspect running systems. The lesson covers network isolation concepts and how ROS_DOMAIN_ID enables multiple ROS2 systems to operate on the same network without interference.
- **Outcome**: By the end of this lesson, you will have used ROS2 command-line tools to examine communication patterns, understood node status and communication topology, worked with services and examined service communication, and understood ROS_DOMAIN_ID and network isolation concepts.
- **Tools**: ROS2 command-line tools (`ros2 topic`, `ros2 node`, `ros2 service`)