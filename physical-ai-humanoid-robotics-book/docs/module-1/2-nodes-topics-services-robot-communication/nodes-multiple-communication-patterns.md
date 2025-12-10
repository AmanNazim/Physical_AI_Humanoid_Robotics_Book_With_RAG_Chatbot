---
sidebar_position: 2
---

# Nodes with Multiple Communication Patterns

## Lesson Overview

In this lesson, you will learn to create nodes that implement multiple communication patterns simultaneously, specifically nodes that both publish and subscribe to different topics within the same node process. This approach creates more sophisticated communication architectures that are essential for complex robotic systems. You will understand how to manage different message types and timing requirements within a single node while maintaining proper lifecycle management and callback execution guarantees.

## Learning Objectives

After completing this lesson, you will be able to:
- Design nodes that can both publish and subscribe to different topics within the same node
- Implement proper node lifecycle management with multiple communication flows
- Understand callback execution guarantees in multi-communication nodes
- Manage different message types and timing requirements within a single node process

## Required Tools and Technologies

- ROS2 Humble Hawksbill
- rclpy (Python client library)
- colcon build system
- Standard ROS2 message types (sensor_msgs, std_msgs)

## Understanding Multi-Communication Nodes

In basic ROS2 implementations, nodes typically either publish data or subscribe to data. However, complex robotic systems often require nodes that participate in multiple communication flows simultaneously. For example, a sensor processing node might subscribe to raw sensor data while publishing processed information to other nodes. This pattern enables more sophisticated robot architectures where nodes can serve multiple roles in the communication network.

Multi-communication nodes offer several advantages:
- Reduced system complexity by consolidating related functionality
- Lower latency between related operations within the same process
- Simplified parameter and state management across communication patterns
- More efficient resource utilization

However, they also introduce complexity in terms of callback management, timing coordination, and lifecycle handling.

## Creating a Node with Both Publishers and Subscribers

Let's implement a node that demonstrates multiple communication patterns. This example will create a node that subscribes to sensor data and publishes processed information based on that data.

First, create a new package for this lesson:

```bash
mkdir -p ~/ros2_ws/src/advanced_communication_tutorials
cd ~/ros2_ws/src/advanced_communication_tutorials
```

Create the package structure:

```bash
mkdir -p advanced_communication_tutorials/launch
mkdir -p advanced_communication_tutorials/config
```

Create the package.xml file:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>advanced_communication_tutorials</name>
  <version>0.0.0</version>
  <description>Advanced ROS2 Communication Tutorials</description>
  <maintainer email="student@todo.todo">student</maintainer>
  <license>Apache-2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Create the setup.py file:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'advanced_communication_tutorials'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='Advanced ROS2 Communication Tutorials',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multi_communication_node = advanced_communication_tutorials.multi_communication_node:main',
            'sensor_publisher = advanced_communication_tutorials.sensor_publisher:main',
        ],
    },
)
```

Now create the main Python module directory:

```bash
mkdir -p advanced_communication_tutorials/advanced_communication_tutorials
```

Create the multi-communication node implementation (`advanced_communication_tutorials/advanced_communication_tutorials/multi_communication_node.py`):

```python
#!/usr/bin/env python3

"""
Node that demonstrates multiple communication patterns simultaneously.
This node subscribes to sensor data and publishes processed information.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class MultiCommunicationNode(Node):
    """
    A ROS2 node that demonstrates multiple communication patterns.
    It subscribes to sensor data and publishes processed information.
    """

    def __init__(self):
        super().__init__('multi_communication_node')

        # Create QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publisher for processed sensor data
        self.processed_data_publisher = self.create_publisher(
            String,
            'processed_sensor_data',
            qos_profile
        )

        # Publisher for calculated values
        self.calculated_value_publisher = self.create_publisher(
            Float32,
            'calculated_value',
            qos_profile
        )

        # Subscriber for raw sensor data
        self.sensor_subscriber = self.create_subscription(
            JointState,
            'raw_sensor_data',
            self.sensor_callback,
            qos_profile
        )

        # Timer for periodic publishing (independent of subscription)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.timer_counter = 0

        self.get_logger().info('Multi-Communication Node initialized')

    def sensor_callback(self, msg):
        """
        Callback function for sensor data subscription.
        Processes incoming sensor data and publishes results.
        """
        self.get_logger().info(f'Received sensor data with {len(msg.name)} joints')

        # Process the sensor data
        processed_info = self.process_sensor_data(msg)

        # Publish processed data
        processed_msg = String()
        processed_msg.data = processed_info
        self.processed_data_publisher.publish(processed_msg)

        # Calculate and publish derived values
        calculated_value = self.calculate_from_sensor_data(msg)
        calculated_msg = Float32()
        calculated_msg.data = calculated_value
        self.calculated_value_publisher.publish(calculated_msg)

        self.get_logger().info(f'Published processed data: {processed_info}')
        self.get_logger().info(f'Published calculated value: {calculated_value}')

    def process_sensor_data(self, joint_state_msg):
        """
        Process the incoming joint state data to extract meaningful information.
        """
        if len(joint_state_msg.position) > 0:
            avg_position = sum(joint_state_msg.position) / len(joint_state_msg.position)
            max_position = max(joint_state_msg.position)
            min_position = min(joint_state_msg.position)

            return f"Processed: avg={avg_position:.2f}, max={max_position:.2f}, min={min_position:.2f}"
        else:
            return "No position data available"

    def calculate_from_sensor_data(self, joint_state_msg):
        """
        Calculate a derived value from sensor data.
        """
        if len(joint_state_msg.position) > 0:
            # Calculate sum of absolute positions as an example
            return sum(abs(pos) for pos in joint_state_msg.position)
        else:
            return 0.0

    def timer_callback(self):
        """
        Timer callback that publishes data independently of subscriptions.
        """
        self.timer_counter += 1
        timer_msg = String()
        timer_msg.data = f'Timer message #{self.timer_counter} - Node is active'
        self.processed_data_publisher.publish(timer_msg)
        self.get_logger().info(f'Timer published: {timer_msg.data}')


def main(args=None):
    rclpy.init(args=args)

    node = MultiCommunicationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Create a simple sensor publisher for testing (`advanced_communication_tutorials/advanced_communication_tutorials/sensor_publisher.py`):

```python
#!/usr/bin/env python3

"""
Simple sensor publisher for testing multi-communication nodes.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time


class SensorPublisher(Node):
    """
    A simple node that publishes mock sensor data for testing.
    """

    def __init__(self):
        super().__init__('sensor_publisher')

        self.publisher = self.create_publisher(JointState, 'raw_sensor_data', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Initialize joint names and create a pattern
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5']
        self.i = 0

        self.get_logger().info('Sensor Publisher initialized')

    def timer_callback(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.name = self.joint_names
        msg.position = []
        msg.velocity = []
        msg.effort = []

        # Create a pattern of values that change over time
        for j, joint_name in enumerate(self.joint_names):
            position = math.sin(self.i * 0.1 + j) * 1.5
            velocity = math.cos(self.i * 0.1 + j) * 0.5
            effort = math.sin(self.i * 0.2 + j) * 0.1

            msg.position.append(position)
            msg.velocity.append(velocity)
            msg.effort.append(effort)

        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: positions={msg.position}')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    publisher = SensorPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        publisher.get_logger().info('Publisher interrupted by user')
    finally:
        publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Proper Node Lifecycle Management

When implementing nodes with multiple communication patterns, proper lifecycle management becomes crucial. The ROS2 lifecycle ensures that nodes initialize, execute, and shutdown cleanly. Here's how to properly manage a multi-communication node:

1. **Initialization Phase**: All publishers, subscribers, timers, and services are created during the node initialization. Resources are allocated and initial configuration is loaded.

2. **Execution Phase**: The node enters the spinning state where it processes callbacks from publishers, subscribers, timers, and services. Proper callback execution guarantees must be maintained.

3. **Shutdown Phase**: All resources are properly cleaned up, connections are closed, and any necessary state is persisted.

The multi-communication node we created follows these principles by properly initializing all communication components in the constructor, handling callbacks appropriately during execution, and ensuring proper cleanup during shutdown.

## Callback Execution Guarantees

In multi-communication nodes, you need to understand callback execution guarantees:

- **Thread Safety**: ROS2 callbacks execute in a thread-safe manner within the same node, but you should still be careful when sharing data between callbacks.
- **Execution Order**: There is no guaranteed order between different callback types (subscriber, timer, service). Your code should not rely on specific execution order.
- **Timing Constraints**: Callbacks must complete within reasonable timeframes to avoid blocking other callbacks in the same node.

In our example, we ensure thread safety by not sharing mutable state between callbacks and keeping callback execution time minimal.

## Managing Different Message Types

Multi-communication nodes often handle different message types simultaneously. In our example, we handle:
- `sensor_msgs/JointState` for sensor data
- `std_msgs/String` for processed information
- `std_msgs/Float32` for calculated values

Each message type has its own purpose and timing requirements. The node processes each type appropriately based on the application's needs.

## Testing the Multi-Communication Node

To test the multi-communication node, build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select advanced_communication_tutorials
source install/setup.bash
```

Run the sensor publisher in one terminal:

```bash
ros2 run advanced_communication_tutorials sensor_publisher
```

Run the multi-communication node in another terminal:

```bash
ros2 run advanced_communication_tutorials multi_communication_node
```

Monitor the topics to see the communication in action:

```bash
# Monitor processed sensor data
ros2 topic echo /processed_sensor_data

# Monitor calculated values
ros2 topic echo /calculated_value
```

## Best Practices for Multi-Communication Nodes

When designing nodes with multiple communication patterns, consider these best practices:

1. **Clear Purpose**: Ensure the node has a clear, unified purpose that justifies combining multiple communication patterns.

2. **Resource Management**: Properly manage resources and ensure all publishers, subscribers, and timers are created and destroyed correctly.

3. **Error Handling**: Implement proper error handling for each communication pattern to maintain system reliability.

4. **Performance Monitoring**: Monitor the performance of multi-communication nodes as they can become bottlenecks if not designed properly.

5. **Separation of Concerns**: Keep the different communication responsibilities distinct within the node to maintain code clarity.

## Summary

In this lesson, you learned how to create nodes that implement multiple communication patterns simultaneously. You implemented a node that both subscribes to sensor data and publishes processed information, while also maintaining a timer for independent publishing. You learned about proper lifecycle management, callback execution guarantees, and how to manage different message types within a single node. This foundation prepares you for more advanced communication patterns in the following lessons.