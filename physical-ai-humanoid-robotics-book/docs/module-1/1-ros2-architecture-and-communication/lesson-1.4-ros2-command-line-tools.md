---
title: Lesson 1.4 - ROS2 Command Line Tools
---

# Lesson 1.4 – ROS2 Command Line Tools

## Learning Objectives

By the end of this lesson, you will be able to:
- Use ROS2 command-line tools to examine communication patterns
- Understand node status and communication topology
- Work with services and examine service communication
- Understand ROS_DOMAIN_ID and network isolation concepts
- Use ROS2 CLI tools for system inspection and debugging

## Concept Overview and Scope

This lesson focuses on using ROS2's powerful command-line tools to examine and debug communication patterns. You'll learn to use `ros2 topic`, `ros2 node`, `ros2 service` commands to inspect running systems. The lesson covers network isolation concepts and how ROS_DOMAIN_ID enables multiple ROS2 systems to operate on the same network without interference.

These command-line tools are essential for effective development and debugging of Physical AI systems, providing visibility into the communication topology and system behavior.

## Introduction to ROS2 Command-Line Tools

ROS2 provides a comprehensive set of command-line tools that are essential for understanding, debugging, and managing ROS2 systems. These tools allow you to inspect the structure of your ROS2 graph, monitor message traffic, diagnose communication issues, and manage system components. Understanding these tools is crucial for effective development and debugging of ROS2-based robotic systems.

The ROS2 command-line interface (CLI) tools follow a consistent naming convention: `ros2 <command> <subcommand> [options]`. This structure makes it easy to remember and use the various tools available.

## Node Management and Inspection Tools

### ros2 node Commands

The `ros2 node` command group provides tools for managing and inspecting nodes in your ROS2 system.

#### Listing Active Nodes
To see all currently running nodes:

```bash
ros2 node list
```

This command shows all nodes that are currently active in the ROS2 system. If you have the publisher and subscriber nodes from the previous lesson running, you'll see:
```
/minimal_publisher
/minimal_subscriber
```

#### Getting Node Information
To get detailed information about a specific node:

```bash
ros2 node info /minimal_publisher
```

This command provides information about the node's publishers, subscribers, services, and parameters.

#### Node Lifecycle Management
For nodes that implement the lifecycle pattern, you can manage their state:

```bash
# List lifecycle nodes
ros2 lifecycle list <node_name>

# Change lifecycle state
ros2 lifecycle set <node_name> <state>
```

## Topic Inspection and Management Tools

### ros2 topic Commands

The `ros2 topic` command group is essential for inspecting and managing topics in your ROS2 system.

#### Listing Active Topics
To see all currently active topics:

```bash
ros2 topic list
```

This will show topics that have at least one publisher or subscriber. With the publisher/subscriber example running, you'll see:
```
/parameter_events
/rosout
/topic
```

#### Getting Topic Information
To get detailed information about a specific topic:

```bash
ros2 topic info /topic
```

This command shows the number of publishers and subscribers for the topic, as well as the message type.

#### Examining Topic Messages
To see messages being published to a topic in real-time:

```bash
ros2 topic echo /topic
```

This command subscribes to the topic and displays messages as they are received, similar to the subscriber node functionality but without creating a persistent subscriber.

#### Publishing Messages Manually
To manually publish a message to a topic:

```bash
ros2 topic pub /topic std_msgs/msg/String "data: 'Hello from command line'"
```

This allows you to send messages without running a dedicated publisher node.

#### Monitoring Topic Statistics
To monitor the rate at which messages are published to a topic:

```bash
ros2 topic hz /topic
```

This command calculates and displays the frequency of messages on the topic.

#### Getting Topic Type Information
To get the message type of a topic:

```bash
ros2 topic type /topic
```

This returns the fully qualified message type, such as `std_msgs/msg/String`.

## Service Inspection and Management Tools

### ros2 service Commands

The `ros2 service` command group provides tools for managing and inspecting services in your ROS2 system.

#### Listing Active Services
To see all currently active services:

```bash
ros2 service list
```

This shows all services that are currently available in the ROS2 system.

#### Getting Service Information
To get detailed information about a specific service:

```bash
ros2 service info <service_name>
```

This command shows the service type and which nodes are providing and using the service.

#### Calling Services
To call a service with specific request parameters:

```bash
ros2 service call <service_name> <service_type> <request_data>
```

For example:
```bash
ros2 service call /set_parameters rcl_interfaces/srv/SetParameters "{parameters: [{name: 'my_param', value: {string_value: 'new_value'}}]}"
```

#### Getting Service Type Information
To get the service type of a specific service:

```bash
ros2 service type <service_name>
```

## Parameter Management Tools

### ros2 param Commands

The `ros2 param` command group provides tools for managing parameters in your ROS2 system.

#### Listing Node Parameters
To list all parameters for a specific node:

```bash
ros2 param list <node_name>
```

#### Getting Parameter Values
To get the value of a specific parameter:

```bash
ros2 param get <node_name> <parameter_name>
```

#### Setting Parameter Values
To set a parameter value:

```bash
ros2 param set <node_name> <parameter_name> <value>
```

#### Loading Parameters from Files
To load parameters from a YAML file:

```bash
ros2 param load <node_name> <parameter_file.yaml>
```

## Action Management Tools

### ros2 action Commands

The `ros2 action` command group provides tools for managing actions in your ROS2 system.

#### Listing Active Actions
To see all currently active actions:

```bash
ros2 action list
```

#### Getting Action Information
To get detailed information about a specific action:

```bash
ros2 action info <action_name>
```

## Advanced Network Isolation Concepts

### ROS_DOMAIN_ID

ROS_DOMAIN_ID is a crucial concept for network isolation in ROS2. It allows multiple ROS2 systems to operate on the same network without interference.

#### Understanding ROS_DOMAIN_ID
- Default value: 0
- Range: 0 to 232 (theoretically up to 2^23 - 1)
- Nodes with different domain IDs cannot communicate with each other
- This enables multiple independent ROS2 systems on the same network

#### Setting ROS_DOMAIN_ID
To set the domain ID for a single command:

```bash
ROS_DOMAIN_ID=1 ros2 run publisher_subscriber_example publisher_node
```

To set it for your entire terminal session:

```bash
export ROS_DOMAIN_ID=1
ros2 run publisher_subscriber_example publisher_node
```

#### Practical Example of Domain Isolation
Terminal 1 (Domain 0):
```bash
export ROS_DOMAIN_ID=0
ros2 run publisher_subscriber_example publisher_node
```

Terminal 2 (Domain 1):
```bash
export ROS_DOMAIN_ID=1
ros2 run publisher_subscriber_example subscriber_node
```

In this example, the publisher and subscriber will not communicate because they are on different domains.

## Practical Exercise: Complete System Inspection

Let's use the command-line tools to inspect the communication graph we created in the previous lesson.

### Step 1: Start Your Publisher and Subscriber Nodes

Terminal 1:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_example publisher_node
```

Terminal 2:
```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_example subscriber_node
```

### Step 2: Inspect the System

In a third terminal, run these commands:

```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Get information about the communication topic
ros2 topic info /topic

# Get information about the publisher node
ros2 node info /minimal_publisher

# Get information about the subscriber node
ros2 node info /minimal_subscriber

# Monitor message rate on the topic
ros2 topic hz /topic

# See messages being published in real-time
ros2 topic echo /topic
```

### Step 3: Analyzing the Communication Topology

The command-line tools allow you to visualize the communication topology without any additional software. You can see:
- Which nodes exist in the system
- What topics they publish to and subscribe from
- How many publishers and subscribers exist for each topic
- The message types being used
- The frequency of message publication

## Physical AI Context and Application

### Service-Based Communication in Physical AI Systems

In Physical AI applications, services provide a synchronous request/response communication pattern that complements the asynchronous topic-based communication. Services are essential for:

- **Configuration Requests**: Setting robot parameters or configuration values
- **Synchronous Operations**: Operations that require a guaranteed response before proceeding
- **Control Commands**: Commands that need confirmation of execution
- **Data Queries**: Requesting specific information from nodes

This service-based communication pattern is fundamental to the three-layer system architecture:
- **Perception Layer**: Services for requesting sensor calibration or configuration
- **Cognition Layer**: Services for requesting processing tasks or status information
- **Actuation Layer**: Services for requesting motor calibration or status updates

## Troubleshooting Common Issues with CLI Tools

### Issue 1: Commands Not Found
If ROS2 commands are not recognized, ensure your environment is properly sourced:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

### Issue 2: No Nodes or Topics Visible
If `ros2 node list` or `ros2 topic list` return empty results:
- Ensure nodes are actually running
- Check that you're not in a different ROS_DOMAIN_ID
- Verify that the nodes have properly initialized

### Issue 3: Permission Issues
If you encounter permission errors, ensure your user is properly set up for ROS2 development and that all necessary directories have appropriate permissions.

## Best Practices for Using ROS2 CLI Tools

### 1. Regular System Monitoring
Make it a habit to check your ROS2 system status regularly:
```bash
ros2 node list && ros2 topic list
```

### 2. Use Descriptive Node Names
When creating nodes, use descriptive names that make it easy to identify their function when using `ros2 node list`.

### 3. Monitor Message Rates
Use `ros2 topic hz` to ensure your topics are publishing at expected rates, which is crucial for real-time robot control.

### 4. Verify Message Content
Use `ros2 topic echo` to verify that messages contain the expected content, especially when debugging communication issues.

## Advanced Usage: Scripting with ROS2 CLI Tools

The ROS2 CLI tools can be combined with shell scripting for more complex operations:

```bash
# Get all nodes and count them
node_count=$(ros2 node list | wc -l)
echo "Number of active nodes: $node_count"

# Find all topics of a specific type
ros2 topic list -t | grep "sensor_msgs"

# Monitor multiple topics simultaneously
ros2 topic echo /topic1 & ros2 topic echo /topic2 &
```

## Lesson Summary

In this lesson, you have learned:

- **Node Management**: How to list, inspect, and manage ROS2 nodes using `ros2 node` commands
- **Topic Inspection**: How to examine topics, monitor message rates, and view message content using `ros2 topic` commands
- **Service Management**: How to list, inspect, and call services using `ros2 service` commands
- **Parameter Management**: How to manage node parameters using `ros2 param` commands
- **Network Isolation**: How to use ROS_DOMAIN_ID to isolate multiple ROS2 systems on the same network
- **System Inspection**: How to use CLI tools to understand the complete communication topology of your ROS2 system

You now have the essential tools needed to inspect and manage ROS2 systems effectively.

## Tools / References
- ROS2 command-line tools (`ros2 topic`, `ros2 node`, `ros2 service`)