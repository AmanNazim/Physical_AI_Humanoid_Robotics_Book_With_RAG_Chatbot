---
title: Lesson 1.1 - Introduction to ROS2 Architecture
---

# Lesson 1.1 – Introduction to ROS2 Architecture

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand what ROS2 is and its role as a communication middleware in robotic systems
- Compare ROS2 with ROS1 and understand the evolution
- Describe the DDS (Data Distribution Service) communication model
- Explain the node-based architecture in ROS2
- Understand the Physical AI context of ROS2 distributed processing

## Concept Overview and Scope

This lesson introduces the fundamental concepts of ROS2 architecture. ROS2 is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. We'll use intuitive analogies like the human nervous system to explain how ROS2 enables distributed robotic systems. The lesson covers the evolution from ROS1 to ROS2, focusing on improvements in security, real-time performance, and multi-robot systems.

ROS2 serves as the essential communication middleware that enables embodied intelligence in Physical AI systems. It provides the distributed architecture needed for seamless coordination between perception, cognition, and actuation layers - exactly what Physical AI systems require.

## Introduction to ROS2 Architecture

ROS2 (Robot Operating System 2) is not just a framework but a complete communication infrastructure that allows different parts of a robot to coordinate with each other. Think of ROS2 as the "nervous system" of a robot - it enables distributed processing while maintaining coordination, just like how your biological nervous system allows different parts of your body to work together seamlessly.

In the context of Physical AI and humanoid robotics, ROS2 serves as the essential communication middleware that enables embodied intelligence. Physical AI systems require seamless coordination between perception, cognition, and actuation layers - exactly what ROS2's distributed architecture provides. The perception layer processes sensor data, the cognition layer makes decisions, and the actuation layer executes movements, all communicating through ROS2 topics, services, and parameters.

## The Evolution from ROS1 to ROS2

ROS1, the original Robot Operating System, was developed to provide a flexible framework for robotics software development. However, as robotics applications grew more complex and diverse, several limitations became apparent:

### Key Limitations of ROS1:
- **Single Master Architecture**: A single point of failure that could bring down the entire system
- **Security Concerns**: No built-in security mechanisms for multi-robot systems
- **Real-time Performance**: Limited support for real-time applications
- **Multi-robot Support**: Difficult to coordinate multiple robots effectively
- **Middleware Dependencies**: Tightly coupled to specific communication protocols

### ROS2 Improvements:
ROS2 addressed these limitations through a complete architectural overhaul, introducing several key improvements:

- **Distributed Architecture**: No single point of failure, with each node capable of discovering others
- **Security Features**: Built-in authentication, authorization, and encryption capabilities
- **Real-time Support**: Better support for real-time applications and deterministic behavior
- **Multi-robot Coordination**: Enhanced support for multi-robot systems and fleet management
- **Middleware Abstraction**: Pluggable middleware through the ROS Middleware (RMW) interface

## Understanding DDS (Data Distribution Service)

The core innovation of ROS2 is its use of DDS (Data Distribution Service) as the underlying communication middleware. DDS is an industry-standard middleware for real-time, distributed systems that provides a data-centric approach to communication.

### DDS Architecture Components:
- **Domain**: A logical network partition where participants can discover and communicate
- **Participant**: An application that participates in a DDS domain
- **Publisher**: An entity that sends data to topics
- **Subscriber**: An entity that receives data from topics
- **Topic**: A named data object that defines the type and name of data being exchanged
- **DataWriter**: The interface between a Publisher and the DDS network
- **DataReader**: The interface between a Subscriber and the DDS network

### DDS Quality of Service (QoS) Policies:
DDS provides Quality of Service (QoS) policies that allow fine-tuning of communication behavior:

- **Reliability**: Guarantees delivery (RELIABLE) or best-effort (BEST_EFFORT)
- **Durability**: Whether data persists for late-joining subscribers (TRANSIENT_LOCAL or VOLATILE)
- **History**: How many samples to keep in the history queue
- **Deadline**: Maximum time between sample updates
- **Liveliness**: How to detect if a participant is still active


## Node-based Architecture in ROS2

ROS2 uses a node-based architecture where each node is an independent process that performs computation and communicates with other nodes.

### Node Characteristics:
- **Isolation**: Each node runs in its own process, providing fault isolation
- **Communication**: Nodes communicate through topics, services, and actions
- **Lifecycle**: Nodes have defined lifecycle states (unconfigured, inactive, active, finalized)
- **Parameters**: Nodes can have configurable parameters that can be changed at runtime

### Node Communication Patterns:
1. **Topics (Publish/Subscribe)**: Asynchronous, one-to-many communication
2. **Services (Request/Response)**: Synchronous, one-to-one communication
3. **Actions**: Goal-oriented communication with feedback and result

This architecture supports the three-layer system:
- **Perception Layer**: Sensor nodes publish raw and processed data
- **Cognition Layer**: Processing nodes interpret sensor data and make decisions
- **Actuation Layer**: Control nodes execute motor commands

## Practical Application and Physical AI Context

In Physical AI systems, the ROS2 architecture enables the distributed processing required for embodied intelligence. When a humanoid robot needs to process sensor data, make decisions, and execute movements, these tasks can be distributed across multiple nodes running on different processors or even different physical computers.

For example:
- A camera sensor node processes visual data and publishes image topics
- A perception node subscribes to these topics and performs object recognition
- A decision-making node receives object information and determines appropriate actions
- A control node executes motor commands based on the decisions

This distributed approach allows for:
- **Scalability**: Adding more computational resources as needed
- **Fault Tolerance**: If one node fails, others can continue operating
- **Modularity**: Components can be developed and tested independently
- **Real-time Performance**: Critical tasks can be prioritized appropriately

## Lesson Summary

In this lesson, you have learned:

- **ROS2 Architecture**: How ROS2 serves as the communication middleware for robotic systems, functioning as the "nervous system" of a robot
- **Evolution from ROS1 to ROS2**: The key improvements and limitations addressed in ROS2, including distributed architecture, security features, real-time support, and multi-robot coordination
- **DDS Communication Model**: The Data Distribution Service as the underlying communication middleware, including its architecture components and Quality of Service policies
- **Node-based Architecture**: How nodes function as independent processes that communicate through topics, services, and actions
- **Physical AI Context**: How ROS2 enables distributed processing for embodied intelligence in humanoid robotics

You now have the essential understanding of ROS2 architecture needed to proceed with environment setup and practical implementation in subsequent lessons.

## Tools / References
- ROS2
- DDS (Data Distribution Service)