---
title: Lesson 1.3 - Basic Publisher/Subscriber Implementation
---

# Lesson 1.3 – Basic Publisher/Subscriber Implementation

## Learning Objective
Write and execute a basic publisher node in Python, write and execute a basic subscriber node in Python, launch and test a ROS2 communication graph, understand the message flow between publisher and subscriber nodes

## Conceptual Scope
This practical lesson teaches you to implement the most fundamental ROS2 communication pattern: publisher-subscriber. You'll write your first ROS2 nodes in Python, creating a publisher that sends messages and a subscriber that receives them. The lesson emphasizes understanding message flow and the asynchronous nature of topic-based communication.

This lesson directly implements the module specification requirement for "Topic-based pub/sub communication patterns" and "Create basic ROS2 nodes, topics, services, and parameters for inter-process communication." It also addresses the book-level goal of "Understand and implement fundamental ROS2 communication patterns step-by-step" by providing hands-on experience with the most basic communication pattern.

The publisher-subscriber pattern is the backbone of ROS2 communication. A publisher node sends messages to a topic, and any number of subscriber nodes can receive those messages. This creates a decoupled system where publishers don't need to know about subscribers and vice versa. This decoupling is what allows for flexible, modular robot architectures.

This lesson also supports the module specification requirement for "Python-based ROS2 control interfaces using rclpy" by providing practical experience using the rclpy library to create nodes. It directly addresses the module completion criteria of "All ROS2 communication patterns (topics, services, actions) must be implemented and tested" by implementing the fundamental topic-based communication pattern.

Key activities include:
- Writing a "Hello World" publisher node
- Writing a "Hello World" subscriber node
- Launching the communication graph
- Testing communication between nodes
- Documenting the communication pattern

## Expected Learning Outcome
By the end of this lesson, you will have written and executed a basic publisher node in Python, written and executed a basic subscriber node in Python, launched and tested a ROS2 communication graph, and understood the message flow between publisher and subscriber nodes.

## Tools / References
- Python 3.8+
- rclpy
- ROS2