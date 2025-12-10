---
title: Lesson 4.1 - Python-based ROS2 Nodes with rclpy
---

# Lesson 4.1 – Python-based ROS2 Nodes with rclpy

## Learning Objectives

By the end of this lesson, you will be able to:
- Create Python nodes for AI agent integration using rclpy
- Process sensor data in Python nodes using rclpy
- Implement high-level decision-making logic in Python
- Integrate Python nodes with ROS2 communication patterns
- Connect Python-based AI agents and control algorithms with ROS2

## Concept Overview and Scope

This lesson focuses on creating Python nodes that serve as the bridge between high-level AI algorithms and the ROS2 communication infrastructure. We'll use `rclpy`, the Python client library for ROS2, to create nodes that can process sensor data, implement decision-making logic, and communicate with other ROS2 nodes.

`rclpy` provides a Python interface to the ROS2 middleware, allowing you to create nodes, publishers, subscribers, services, and other ROS2 components directly in Python. This is particularly valuable for AI development, as Python is the dominant language for machine learning and AI research.

## Understanding rclpy

`rclpy` is the Python client library for ROS2 that provides a Python API for all ROS2 concepts. It's built on top of the ROS Client Library (rcl) and the underlying middleware interface (rmw).

### Key Components of rclpy:

1. **Node**: The basic execution unit in ROS2
2. **Publisher**: For sending messages to topics
3. **Subscriber**: For receiving messages from topics
4. **Service Server**: For providing services
5. **Service Client**: For calling services
6. **Action Server/Client**: For goal-oriented communication
7. **Parameters**: For runtime configuration

### Basic rclpy Node Structure

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize publishers, subscribers, services, etc.

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating a Python Node with rclpy

Let's create a simple Python node that processes sensor data using rclpy:

### Basic Node Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class SensorProcessorNode(Node):
    def __init__(self):
        super().__init__('sensor_processor_node')

        # Create a subscription to receive joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Create a publisher for processed data
        self.publisher = self.create_publisher(
            Float64MultiArray,
            'processed_sensor_data',
            10)

        self.get_logger().info('Sensor Processor Node has been started')

    def listener_callback(self, msg):
        # Process the received joint state data
        self.get_logger().info(f'Received joint positions: {msg.position}')

        # Example processing: calculate some derived values
        processed_data = Float64MultiArray()
        processed_data.data = [pos * 2 for pos in msg.position]  # Example processing

        # Publish the processed data
        self.publisher.publish(processed_data)
        self.get_logger().info(f'Published processed data: {processed_data.data}')

def main(args=None):
    rclpy.init(args=args)

    sensor_processor = SensorProcessorNode()

    try:
        rclpy.spin(sensor_processor)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Implementing Decision-Making Logic

In this section, we'll explore how to implement high-level decision-making logic in Python nodes. This is where AI algorithms can be integrated into the ROS2 system.

### Example: Simple Decision-Making Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math

class DecisionMakerNode(Node):
    def __init__(self):
        super().__init__('decision_maker_node')

        # Create subscription for sensor data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

        # Create publisher for velocity commands
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Create a timer for decision-making loop
        self.timer = self.create_timer(0.1, self.decision_loop)

        self.latest_scan = None
        self.get_logger().info('Decision Maker Node has been started')

    def scan_callback(self, msg):
        # Store the latest scan data
        self.latest_scan = msg

    def decision_loop(self):
        if self.latest_scan is None:
            return

        # Simple obstacle avoidance logic
        min_distance = min(self.latest_scan.ranges)

        cmd = Twist()

        if min_distance < 1.0:  # Obstacle detected within 1 meter
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn right
        else:
            cmd.linear.x = 0.5   # Move forward
            cmd.angular.z = 0.0

        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)

    decision_maker = DecisionMakerNode()

    try:
        rclpy.spin(decision_maker)
    except KeyboardInterrupt:
        pass
    finally:
        decision_maker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integrating with ROS2 Communication Patterns

In this section, we'll explore how to integrate Python nodes with various ROS2 communication patterns:

### Publishers and Subscribers

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState

class CommunicationNode(Node):
    def __init__(self):
        super().__init__('communication_node')

        # Publisher
        self.publisher = self.create_publisher(String, 'ai_commands', 10)

        # Multiple subscribers
        self.joint_subscriber = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)
        self.imu_subscriber = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)

        # Timer for periodic publishing
        self.timer = self.create_timer(1.0, self.publish_command)

        self.command_counter = 0

    def joint_callback(self, msg):
        # Process joint state data
        self.get_logger().info(f'Joint positions: {msg.position}')

    def imu_callback(self, msg):
        # Process IMU data
        self.get_logger().info(f'Orientation: {msg.orientation}')

    def publish_command(self):
        msg = String()
        msg.data = f'Command #{self.command_counter}'
        self.publisher.publish(msg)
        self.command_counter += 1
```

### Services

```python
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from example_interfaces.srv import Trigger

class ServiceNode(Node):
    def __init__(self):
        super().__init__('service_node')

        # Create service server
        self.srv = self.create_service(
            SetBool,
            'set_behavior_mode',
            self.behavior_mode_callback)

        # Create service client
        self.client = self.create_client(Trigger, 'reset_system')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

    def behavior_mode_callback(self, request, response):
        if request.data:
            self.get_logger().info('Enabling autonomous mode')
            response.success = True
            response.message = 'Autonomous mode enabled'
        else:
            self.get_logger().info('Disabling autonomous mode')
            response.success = True
            response.message = 'Autonomous mode disabled'

        return response
```

### Parameters

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)
        self.declare_parameter('ai_confidence_threshold', 0.8)

        # Access parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.confidence_threshold = self.get_parameter('ai_confidence_threshold').value

        # Set up parameter callback for runtime changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.type_ == Parameter.Type.PARAMETER_DOUBLE:
                self.max_velocity = param.value
                self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
        return SetParametersResult(successful=True)
```

## Best Practices for Python-ROS2 Integration

### 1. Error Handling and Logging

```python
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        try:
            self.declare_parameter('critical_param', 'default_value')
        except Exception as e:
            self.get_logger().error(f'Failed to declare parameter: {e}')

    def safe_publish(self, publisher, msg):
        try:
            publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish message: {e}')
```

### 2. Threading Considerations

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from threading import Lock

class ThreadSafeNode(Node):
    def __init__(self):
        super().__init__('thread_safe_node')
        self.data_lock = Lock()
        self.shared_data = None

    def data_callback(self, msg):
        with self.data_lock:
            self.shared_data = msg
```

### 3. Resource Management

```python
class ResourceManagedNode(Node):
    def __init__(self):
        super().__init__('resource_managed_node')
        # Initialize resources

    def destroy_node(self):
        # Clean up resources before destroying node
        self.cleanup_resources()
        super().destroy_node()

    def cleanup_resources(self):
        # Close files, disconnect from external systems, etc.
        pass
```

## Step-by-Step Exercise

Let's create a complete Python node that integrates AI decision-making with ROS2:

1. Create a new file called `ai_decision_maker.py`
2. Implement a node that subscribes to sensor data (e.g., LaserScan)
3. Implement simple decision-making logic (e.g., obstacle avoidance)
4. Publish commands to control the robot
5. Add parameters for tuning the behavior

## Summary

In this lesson, you learned:
- How to create Python nodes using rclpy
- How to process sensor data in Python nodes
- How to implement decision-making logic in Python
- How to integrate Python nodes with ROS2 communication patterns
- Best practices for Python-ROS2 integration

These skills form the foundation for connecting AI algorithms with robotic systems, enabling intelligent agents to interact with the physical world through ROS2.

## Next Steps

In the next lesson, we'll explore how to interface these Python nodes with simulation environments, specifically Gazebo, to create a complete system where AI agents can control simulated robots.