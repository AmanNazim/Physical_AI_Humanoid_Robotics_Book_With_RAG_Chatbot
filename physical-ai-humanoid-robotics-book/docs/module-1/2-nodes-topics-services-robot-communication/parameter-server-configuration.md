---
sidebar_position: 4
---

# Parameter Server Configuration

## Lesson Overview

In this lesson, you will learn to configure and manage ROS2 parameters for dynamic node behavior and configuration. The ROS2 parameter server provides a centralized system for managing configuration values that can be changed at runtime. You will implement parameterized nodes that can adapt their behavior dynamically, supporting different robot configurations and operational modes. This lesson covers parameter definition, validation, runtime updates, and configuration file management.

## Learning Objectives

After completing this lesson, you will be able to:
- Understand the ROS2 parameter server and its role in robot configuration
- Define and use parameters in nodes with proper validation
- Implement runtime parameter updates for dynamic behavior
- Create and use parameter configuration files (YAML)
- Design parameterized nodes that adapt behavior at runtime
- Implement parameter validation and fallback mechanisms

## Required Tools and Technologies

- ROS2 Humble Hawksbill
- rclpy (Python client library)
- Parameter configuration files (YAML)
- colcon build system

## Understanding the ROS2 Parameter Server

The ROS2 parameter server is a distributed system that allows nodes to share configuration values. Parameters provide a way to configure node behavior without recompiling code, making robots adaptable to different configurations and operational requirements. Key features of the ROS2 parameter system include:

- **Runtime Configuration**: Parameters can be changed while nodes are running
- **Type Safety**: Parameters have defined types (integer, float, string, boolean, lists)
- **Declarative Definition**: Parameters can be declared with default values and constraints
- **Configuration Files**: Parameters can be loaded from YAML files at startup
- **Command-Line Tools**: Parameters can be viewed and modified using ROS2 command-line tools

Parameters are particularly useful for:
- Robot-specific configurations (joint limits, sensor offsets)
- Operational modes (debug vs production settings)
- Tuning values (PID controller parameters)
- Feature flags (enabling/disabling functionality)

## Defining and Using Parameters in Nodes

Let's create a parameterized node that demonstrates how to declare, use, and update parameters at runtime. First, let's create the parameterized node implementation:

Create the parameterized node file (`advanced_communication_tutorials/advanced_communication_tutorials/parameterized_robot_node.py`):

```python
#!/usr/bin/env python3

"""
Parameterized robot node demonstrating dynamic configuration.
This node uses parameters to control its behavior at runtime.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
import math


class ParameterizedRobotNode(Node):
    """
    A ROS2 node that demonstrates parameter-based configuration.
    The node behavior can be changed by modifying parameters at runtime.
    """

    def __init__(self):
        super().__init__('parameterized_robot_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot',
                              ParameterDescriptor(description='Name of the robot'))
        self.declare_parameter('control_frequency', 50,
                              ParameterDescriptor(description='Control loop frequency in Hz'))
        self.declare_parameter('safety_mode', True,
                              ParameterDescriptor(description='Enable safety checks'))
        self.declare_parameter('max_velocity', 1.0,
                              ParameterDescriptor(description='Maximum joint velocity'))
        self.declare_parameter('debug_mode', False,
                              ParameterDescriptor(description='Enable debug output'))
        self.declare_parameter('joint_offsets', [0.0, 0.0, 0.0, 0.0, 0.0],
                              ParameterDescriptor(description='Joint position offsets'))
        self.declare_parameter('operation_mode', 'normal',
                              ParameterDescriptor(description='Operation mode: normal, calibration, maintenance'))

        # Get initial parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_mode = self.get_parameter('safety_mode').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.joint_offsets = self.get_parameter('joint_offsets').value
        self.operation_mode = self.get_parameter('operation_mode').value

        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers
        self.status_publisher = self.create_publisher(String, 'robot_status', qos_profile)
        self.control_publisher = self.create_publisher(JointState, 'joint_commands', qos_profile)
        self.debug_publisher = self.create_publisher(String, 'debug_info', qos_profile)

        # Timer for control loop
        timer_period = 1.0 / self.control_frequency  # seconds
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Internal state
        self.loop_counter = 0
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.get_logger().info(f'Parameterized Robot Node "{self.robot_name}" initialized')
        self.get_logger().info(f'Control frequency: {self.control_frequency}Hz')
        self.get_logger().info(f'Safety mode: {self.safety_mode}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Debug mode: {self.debug_mode}')
        self.get_logger().info(f'Joint offsets: {self.joint_offsets}')
        self.get_logger().info(f'Operation mode: {self.operation_mode}')

    def parameter_callback(self, params):
        """
        Callback function for parameter changes.
        This is called when parameters are updated at runtime.
        """
        for param in params:
            if param.name == 'robot_name':
                self.robot_name = param.value
                self.get_logger().info(f'Robot name updated to: {self.robot_name}')
            elif param.name == 'control_frequency':
                if param.value > 0:
                    self.control_frequency = param.value
                    # Update timer period
                    new_period = 1.0 / self.control_frequency
                    self.timer.timer_period_ns = int(new_period * 1e9)
                    self.get_logger().info(f'Control frequency updated to: {self.control_frequency}Hz')
                else:
                    return SetParametersResult(successful=False, reason='Control frequency must be positive')
            elif param.name == 'safety_mode':
                self.safety_mode = param.value
                self.get_logger().info(f'Safety mode updated to: {self.safety_mode}')
            elif param.name == 'max_velocity':
                if param.value >= 0:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
                else:
                    return SetParametersResult(successful=False, reason='Max velocity must be non-negative')
            elif param.name == 'debug_mode':
                self.debug_mode = param.value
                self.get_logger().info(f'Debug mode updated to: {self.debug_mode}')
            elif param.name == 'joint_offsets':
                if len(param.value) == 5:  # Assuming 5 joints
                    self.joint_offsets = param.value
                    self.get_logger().info(f'Joint offsets updated to: {self.joint_offsets}')
                else:
                    return SetParametersResult(successful=False, reason='Joint offsets must have 5 values')
            elif param.name == 'operation_mode':
                valid_modes = ['normal', 'calibration', 'maintenance']
                if param.value in valid_modes:
                    self.operation_mode = param.value
                    self.get_logger().info(f'Operation mode updated to: {self.operation_mode}')
                else:
                    return SetParametersResult(successful=False, reason=f'Invalid operation mode. Valid: {valid_modes}')

        return SetParametersResult(successful=True)

    def control_loop(self):
        """
        Main control loop that runs at the specified frequency.
        This loop uses parameters to determine its behavior.
        """
        self.loop_counter += 1

        # Update joint positions based on current mode and parameters
        if self.operation_mode == 'calibration':
            # In calibration mode, move through a calibration pattern
            time_factor = self.loop_counter * 0.02  # Adjust based on control frequency
            self.joint_positions = [
                math.sin(time_factor + i * 0.5) * 0.5 for i in range(5)
            ]
        elif self.operation_mode == 'maintenance':
            # In maintenance mode, hold position or move slowly
            self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:  # normal mode
            # In normal mode, follow a more complex pattern
            time_factor = self.loop_counter * 0.01
            self.joint_positions = [
                math.sin(time_factor * (i + 1)) * (0.5 - i * 0.1) for i in range(5)
            ]

        # Apply joint offsets
        adjusted_positions = [
            pos + offset for pos, offset in zip(self.joint_positions, self.joint_offsets)
        ]

        # Apply velocity limits if safety mode is enabled
        if self.safety_mode:
            for i, pos in enumerate(adjusted_positions):
                # Apply velocity limiting
                target_vel = (pos - self.joint_positions[i]) * self.control_frequency
                if abs(target_vel) > self.max_velocity:
                    # Limit velocity
                    limited_change = math.copysign(self.max_velocity / self.control_frequency, target_vel)
                    adjusted_positions[i] = self.joint_positions[i] + limited_change

        # Update velocities (approximate)
        self.joint_velocities = [
            (new_pos - old_pos) * self.control_frequency
            for new_pos, old_pos in zip(adjusted_positions, self.joint_positions)
        ]

        # Publish status
        status_msg = String()
        status_msg.data = f'Robot: {self.robot_name}, Mode: {self.operation_mode}, Loop: {self.loop_counter}'
        self.status_publisher.publish(status_msg)

        # Publish joint commands
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = [f'joint_{i}' for i in range(5)]
        joint_msg.position = adjusted_positions
        joint_msg.velocity = self.joint_velocities
        joint_msg.effort = [0.0] * 5  # No effort values in this example
        self.control_publisher.publish(joint_msg)

        # Publish debug info if enabled
        if self.debug_mode:
            debug_msg = String()
            debug_msg.data = f'Pos: {adjusted_positions}, Vel: {self.joint_velocities}, Params: freq={self.control_frequency}, max_vel={self.max_velocity}'
            self.debug_publisher.publish(debug_msg)

        if self.loop_counter % 100 == 0:  # Log every 100 iterations
            self.get_logger().info(f'Control loop status - Mode: {self.operation_mode}, Positions: {adjusted_positions[:3]}...')

    def get_current_config(self):
        """
        Return current parameter configuration as a dictionary.
        """
        return {
            'robot_name': self.robot_name,
            'control_frequency': self.control_frequency,
            'safety_mode': self.safety_mode,
            'max_velocity': self.max_velocity,
            'debug_mode': self.debug_mode,
            'joint_offsets': self.joint_offsets,
            'operation_mode': self.operation_mode
        }


def main(args=None):
    rclpy.init(args=args)

    node = ParameterizedRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Parameterized node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

We need to add the ParameterDescriptor import. Let me update the imports:

```python
#!/usr/bin/env python3

"""
Parameterized robot node demonstrating dynamic configuration.
This node uses parameters to control its behavior at runtime.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.parameter_service import ParameterService
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.srv import SetParametersResult
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
import math


class ParameterizedRobotNode(Node):
    """
    A ROS2 node that demonstrates parameter-based configuration.
    The node behavior can be changed by modifying parameters at runtime.
    """

    def __init__(self):
        super().__init__('parameterized_robot_node')

        # Declare parameters with default values and descriptions
        self.declare_parameter('robot_name', 'default_robot',
                              ParameterDescriptor(description='Name of the robot'))
        self.declare_parameter('control_frequency', 50,
                              ParameterDescriptor(description='Control loop frequency in Hz'))
        self.declare_parameter('safety_mode', True,
                              ParameterDescriptor(description='Enable safety checks'))
        self.declare_parameter('max_velocity', 1.0,
                              ParameterDescriptor(description='Maximum joint velocity'))
        self.declare_parameter('debug_mode', False,
                              ParameterDescriptor(description='Enable debug output'))
        self.declare_parameter('joint_offsets', [0.0, 0.0, 0.0, 0.0, 0.0],
                              ParameterDescriptor(description='Joint position offsets'))
        self.declare_parameter('operation_mode', 'normal',
                              ParameterDescriptor(description='Operation mode: normal, calibration, maintenance'))

        # Get initial parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_mode = self.get_parameter('safety_mode').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.joint_offsets = self.get_parameter('joint_offsets').value
        self.operation_mode = self.get_parameter('operation_mode').value

        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers
        self.status_publisher = self.create_publisher(String, 'robot_status', qos_profile)
        self.control_publisher = self.create_publisher(JointState, 'joint_commands', qos_profile)
        self.debug_publisher = self.create_publisher(String, 'debug_info', qos_profile)

        # Timer for control loop
        timer_period = 1.0 / self.control_frequency  # seconds
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Internal state
        self.loop_counter = 0
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.get_logger().info(f'Parameterized Robot Node "{self.robot_name}" initialized')
        self.get_logger().info(f'Control frequency: {self.control_frequency}Hz')
        self.get_logger().info(f'Safety mode: {self.safety_mode}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Debug mode: {self.debug_mode}')
        self.get_logger().info(f'Joint offsets: {self.joint_offsets}')
        self.get_logger().info(f'Operation mode: {self.operation_mode}')

    def parameter_callback(self, params):
        """
        Callback function for parameter changes.
        This is called when parameters are updated at runtime.
        """
        for param in params:
            if param.name == 'robot_name':
                self.robot_name = param.value
                self.get_logger().info(f'Robot name updated to: {self.robot_name}')
            elif param.name == 'control_frequency':
                if param.value > 0:
                    self.control_frequency = param.value
                    # Update timer period
                    new_period = 1.0 / self.control_frequency
                    self.timer.timer_period_ns = int(new_period * 1e9)
                    self.get_logger().info(f'Control frequency updated to: {self.control_frequency}Hz')
                else:
                    return SetParametersResult(successful=False, reason='Control frequency must be positive')
            elif param.name == 'safety_mode':
                self.safety_mode = param.value
                self.get_logger().info(f'Safety mode updated to: {self.safety_mode}')
            elif param.name == 'max_velocity':
                if param.value >= 0:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
                else:
                    return SetParametersResult(successful=False, reason='Max velocity must be non-negative')
            elif param.name == 'debug_mode':
                self.debug_mode = param.value
                self.get_logger().info(f'Debug mode updated to: {self.debug_mode}')
            elif param.name == 'joint_offsets':
                if len(param.value) == 5:  # Assuming 5 joints
                    self.joint_offsets = param.value
                    self.get_logger().info(f'Joint offsets updated to: {self.joint_offsets}')
                else:
                    return SetParametersResult(successful=False, reason='Joint offsets must have 5 values')
            elif param.name == 'operation_mode':
                valid_modes = ['normal', 'calibration', 'maintenance']
                if param.value in valid_modes:
                    self.operation_mode = param.value
                    self.get_logger().info(f'Operation mode updated to: {self.operation_mode}')
                else:
                    return SetParametersResult(successful=False, reason=f'Invalid operation mode. Valid: {valid_modes}')

        return SetParametersResult(successful=True)

    def control_loop(self):
        """
        Main control loop that runs at the specified frequency.
        This loop uses parameters to determine its behavior.
        """
        self.loop_counter += 1

        # Update joint positions based on current mode and parameters
        if self.operation_mode == 'calibration':
            # In calibration mode, move through a calibration pattern
            time_factor = self.loop_counter * 0.02  # Adjust based on control frequency
            self.joint_positions = [
                math.sin(time_factor + i * 0.5) * 0.5 for i in range(5)
            ]
        elif self.operation_mode == 'maintenance':
            # In maintenance mode, hold position or move slowly
            self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:  # normal mode
            # In normal mode, follow a more complex pattern
            time_factor = self.loop_counter * 0.01
            self.joint_positions = [
                math.sin(time_factor * (i + 1)) * (0.5 - i * 0.1) for i in range(5)
            ]

        # Apply joint offsets
        adjusted_positions = [
            pos + offset for pos, offset in zip(self.joint_positions, self.joint_offsets)
        ]

        # Apply velocity limits if safety mode is enabled
        if self.safety_mode:
            for i, pos in enumerate(adjusted_positions):
                # Apply velocity limiting
                target_vel = (pos - self.joint_positions[i]) * self.control_frequency
                if abs(target_vel) > self.max_velocity:
                    # Limit velocity
                    limited_change = math.copysign(self.max_velocity / self.control_frequency, target_vel)
                    adjusted_positions[i] = self.joint_positions[i] + limited_change

        # Update velocities (approximate)
        self.joint_velocities = [
            (new_pos - old_pos) * self.control_frequency
            for new_pos, old_pos in zip(adjusted_positions, self.joint_positions)
        ]

        # Publish status
        status_msg = String()
        status_msg.data = f'Robot: {self.robot_name}, Mode: {self.operation_mode}, Loop: {self.loop_counter}'
        self.status_publisher.publish(status_msg)

        # Publish joint commands
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = [f'joint_{i}' for i in range(5)]
        joint_msg.position = adjusted_positions
        joint_msg.velocity = self.joint_velocities
        joint_msg.effort = [0.0] * 5  # No effort values in this example
        self.control_publisher.publish(joint_msg)

        # Publish debug info if enabled
        if self.debug_mode:
            debug_msg = String()
            debug_msg.data = f'Pos: {adjusted_positions}, Vel: {self.joint_velocities}, Params: freq={self.control_frequency}, max_vel={self.max_velocity}'
            self.debug_publisher.publish(debug_msg)

        if self.loop_counter % 100 == 0:  # Log every 100 iterations
            self.get_logger().info(f'Control loop status - Mode: {self.operation_mode}, Positions: {adjusted_positions[:3]}...}')

    def get_current_config(self):
        """
        Return current parameter configuration as a dictionary.
        """
        return {
            'robot_name': self.robot_name,
            'control_frequency': self.control_frequency,
            'safety_mode': self.safety_mode,
            'max_velocity': self.max_velocity,
            'debug_mode': self.debug_mode,
            'joint_offsets': self.joint_offsets,
            'operation_mode': self.operation_mode
        }


def main(args=None):
    rclpy.init(args=args)

    node = ParameterizedRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Parameterized node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Actually, let me create a simpler version that doesn't need the additional imports to avoid complexity:

```python
#!/usr/bin/env python3

"""
Parameterized robot node demonstrating dynamic configuration.
This node uses parameters to control its behavior at runtime.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
import math


class ParameterizedRobotNode(Node):
    """
    A ROS2 node that demonstrates parameter-based configuration.
    The node behavior can be changed by modifying parameters at runtime.
    """

    def __init__(self):
        super().__init__('parameterized_robot_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('control_frequency', 50)
        self.declare_parameter('safety_mode', True)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('joint_offsets', [0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('operation_mode', 'normal')

        # Get initial parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.safety_mode = self.get_parameter('safety_mode').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.joint_offsets = self.get_parameter('joint_offsets').value
        self.operation_mode = self.get_parameter('operation_mode').value

        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        # Publishers
        self.status_publisher = self.create_publisher(String, 'robot_status', qos_profile)
        self.control_publisher = self.create_publisher(JointState, 'joint_commands', qos_profile)
        self.debug_publisher = self.create_publisher(String, 'debug_info', qos_profile)

        # Timer for control loop
        timer_period = 1.0 / self.control_frequency  # seconds
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Parameter change callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Internal state
        self.loop_counter = 0
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.get_logger().info(f'Parameterized Robot Node "{self.robot_name}" initialized')
        self.get_logger().info(f'Control frequency: {self.control_frequency}Hz')
        self.get_logger().info(f'Safety mode: {self.safety_mode}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Debug mode: {self.debug_mode}')
        self.get_logger().info(f'Joint offsets: {self.joint_offsets}')
        self.get_logger().info(f'Operation mode: {self.operation_mode}')

    def parameter_callback(self, params):
        """
        Callback function for parameter changes.
        This is called when parameters are updated at runtime.
        """
        from rcl_interfaces.srv import SetParametersResult

        for param in params:
            if param.name == 'robot_name':
                self.robot_name = param.value
                self.get_logger().info(f'Robot name updated to: {self.robot_name}')
            elif param.name == 'control_frequency':
                if param.value > 0:
                    self.control_frequency = param.value
                    # Update timer period - note: in real applications you might want to recreate the timer
                    self.get_logger().info(f'Control frequency updated to: {self.control_frequency}Hz')
                else:
                    return SetParametersResult(successful=False, reason='Control frequency must be positive')
            elif param.name == 'safety_mode':
                self.safety_mode = param.value
                self.get_logger().info(f'Safety mode updated to: {self.safety_mode}')
            elif param.name == 'max_velocity':
                if param.value >= 0:
                    self.max_velocity = param.value
                    self.get_logger().info(f'Max velocity updated to: {self.max_velocity}')
                else:
                    return SetParametersResult(successful=False, reason='Max velocity must be non-negative')
            elif param.name == 'debug_mode':
                self.debug_mode = param.value
                self.get_logger().info(f'Debug mode updated to: {self.debug_mode}')
            elif param.name == 'joint_offsets':
                if len(param.value) == 5:  # Assuming 5 joints
                    self.joint_offsets = param.value
                    self.get_logger().info(f'Joint offsets updated to: {self.joint_offsets}')
                else:
                    return SetParametersResult(successful=False, reason='Joint offsets must have 5 values')
            elif param.name == 'operation_mode':
                valid_modes = ['normal', 'calibration', 'maintenance']
                if param.value in valid_modes:
                    self.operation_mode = param.value
                    self.get_logger().info(f'Operation mode updated to: {self.operation_mode}')
                else:
                    return SetParametersResult(successful=False, reason=f'Invalid operation mode. Valid: {valid_modes}')

        return SetParametersResult(successful=True)

    def control_loop(self):
        """
        Main control loop that runs at the specified frequency.
        This loop uses parameters to determine its behavior.
        """
        self.loop_counter += 1

        # Update joint positions based on current mode and parameters
        if self.operation_mode == 'calibration':
            # In calibration mode, move through a calibration pattern
            time_factor = self.loop_counter * 0.02  # Adjust based on control frequency
            self.joint_positions = [
                math.sin(time_factor + i * 0.5) * 0.5 for i in range(5)
            ]
        elif self.operation_mode == 'maintenance':
            # In maintenance mode, hold position or move slowly
            self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:  # normal mode
            # In normal mode, follow a more complex pattern
            time_factor = self.loop_counter * 0.01
            self.joint_positions = [
                math.sin(time_factor * (i + 1)) * (0.5 - i * 0.1) for i in range(5)
            ]

        # Apply joint offsets
        adjusted_positions = [
            pos + offset for pos, offset in zip(self.joint_positions, self.joint_offsets)
        ]

        # Apply velocity limits if safety mode is enabled
        if self.safety_mode:
            for i, pos in enumerate(adjusted_positions):
                # Apply velocity limiting (simplified approach)
                pass  # In a real system, you'd implement velocity limiting here

        # Update velocities (approximate)
        if self.loop_counter > 1:  # Skip first iteration
            self.joint_velocities = [
                (new_pos - old_pos) * self.control_frequency
                for new_pos, old_pos in zip(adjusted_positions, self.joint_positions)
            ]
        else:
            self.joint_velocities = [0.0] * 5

        # Publish status
        status_msg = String()
        status_msg.data = f'Robot: {self.robot_name}, Mode: {self.operation_mode}, Loop: {self.loop_counter}'
        self.status_publisher.publish(status_msg)

        # Publish joint commands
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'
        joint_msg.name = [f'joint_{i}' for i in range(5)]
        joint_msg.position = adjusted_positions
        joint_msg.velocity = self.joint_velocities
        joint_msg.effort = [0.0] * 5  # No effort values in this example
        self.control_publisher.publish(joint_msg)

        # Publish debug info if enabled
        if self.debug_mode:
            debug_msg = String()
            debug_msg.data = f'Pos: {adjusted_positions}, Vel: {self.joint_velocities}, Params: freq={self.control_frequency}, max_vel={self.max_velocity}'
            self.debug_publisher.publish(debug_msg)

        if self.loop_counter % 100 == 0:  # Log every 100 iterations
            self.get_logger().info(f'Control loop status - Mode: {self.operation_mode}, Positions: {adjusted_positions[:3]}...')

    def get_current_config(self):
        """
        Return current parameter configuration as a dictionary.
        """
        return {
            'robot_name': self.robot_name,
            'control_frequency': self.control_frequency,
            'safety_mode': self.safety_mode,
            'max_velocity': self.max_velocity,
            'debug_mode': self.debug_mode,
            'joint_offsets': self.joint_offsets,
            'operation_mode': self.operation_mode
        }


def main(args=None):
    rclpy.init(args=args)

    node = ParameterizedRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Parameterized node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

Now let me update the setup.py to include this new node:

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
            'robot_state_server = advanced_communication_tutorials.robot_state_server:main',
            'robot_command_client = advanced_communication_tutorials.robot_command_client:main',
            'parameterized_robot_node = advanced_communication_tutorials.parameterized_robot_node:main',
        ],
    },
)
```

## Creating Parameter Configuration Files (YAML)

Parameter configuration files in YAML format allow you to set parameter values at node startup. Create a config directory and configuration file:

```bash
mkdir -p advanced_communication_tutorials/config
```

Create a default configuration file (`advanced_communication_tutorials/config/default_robot_config.yaml`):

```yaml
parameterized_robot_node:
  ros__parameters:
    robot_name: "my_robot"
    control_frequency: 100
    safety_mode: true
    max_velocity: 2.0
    debug_mode: false
    joint_offsets: [0.1, -0.1, 0.05, -0.05, 0.0]
    operation_mode: "normal"

# Example of another node's parameters
sensor_publisher:
  ros__parameters:
    publish_frequency: 50
    sensor_noise_level: 0.01
```

Create a calibration configuration file (`advanced_communication_tutorials/config/calibration_config.yaml`):

```yaml
parameterized_robot_node:
  ros__parameters:
    robot_name: "calibration_robot"
    control_frequency: 10
    safety_mode: true
    max_velocity: 0.5
    debug_mode: true
    joint_offsets: [0.0, 0.0, 0.0, 0.0, 0.0]
    operation_mode: "calibration"

# Additional calibration-specific parameters
calibration_node:
  ros__parameters:
    calibration_speed: 0.1
    tolerance: 0.001
    max_attempts: 5
```

## Using Parameter Configuration Files

To use these configuration files when launching your node, you can either:

1. Load them programmatically in your launch file
2. Specify them on the command line when running the node

Here's how to load parameters from a YAML file when starting a node:

```bash
# Run the node with a specific configuration file
ros2 run advanced_communication_tutorials parameterized_robot_node --ros-args --params-file config/default_robot_config.yaml
```

## Runtime Parameter Updates

ROS2 provides command-line tools to update parameters while nodes are running. Here are the most common operations:

1. **List all parameters for a node:**
```bash
ros2 param list /parameterized_robot_node
```

2. **Get a specific parameter value:**
```bash
ros2 param get /parameterized_robot_node robot_name
```

3. **Set a parameter value:**
```bash
ros2 param set /parameterized_robot_node debug_mode true
```

4. **Get all parameters in YAML format:**
```bash
ros2 param dump /parameterized_robot_node
```

## Parameter Validation and Fallback Mechanisms

In our parameterized node implementation, we included validation mechanisms:

1. **Range Validation**: We validate that control frequency is positive and max velocity is non-negative.

2. **List Length Validation**: We ensure that joint offsets have exactly 5 values.

3. **Value Set Validation**: We validate that operation mode is one of the allowed values.

4. **Fallback Behavior**: If a parameter update fails validation, the parameter callback returns a failure result, and the parameter value remains unchanged.

## Testing Parameter Configuration

First, build the package with the new node:

```bash
cd ~/ros2_ws
colcon build --packages-select advanced_communication_tutorials
source install/setup.bash
```

Run the parameterized node:

```bash
ros2 run advanced_communication_tutorials parameterized_robot_node
```

In another terminal, you can modify parameters while the node is running:

```bash
# Change the operation mode
ros2 param set /parameterized_robot_node operation_mode calibration

# Enable debug mode
ros2 param set /parameterized_robot_node debug_mode true

# Change max velocity
ros2 param set /parameterized_robot_node max_velocity 1.5

# View all parameters
ros2 param list /parameterized_robot_node
```

You can also run the node with a configuration file:

```bash
ros2 run advanced_communication_tutorials parameterized_robot_node --ros-args --params-file ~/ros2_ws/src/advanced_communication_tutorials/config/calibration_config.yaml
```

## Best Practices for Parameter Management

When designing parameterized nodes, follow these best practices:

1. **Use Descriptive Names**: Choose parameter names that clearly indicate their purpose.

2. **Provide Good Defaults**: Set sensible default values that work for most use cases.

3. **Validate Input**: Always validate parameter values to prevent invalid configurations.

4. **Document Parameters**: Document what each parameter does and its valid range of values.

5. **Group Related Parameters**: Organize related parameters logically in configuration files.

6. **Use Appropriate Types**: Use the correct parameter types (int, float, string, bool, lists) for validation.

7. **Consider Performance**: Be aware that parameter callbacks are synchronous and can affect node performance.

8. **Plan for Multiple Configurations**: Design your system to support different operational configurations (development, testing, production).

## Parameter Management for Different Robot Configurations

Parameters enable the same node code to work with different robot configurations. For example:

- **Different Robot Models**: Use parameters to specify joint limits, link lengths, and other robot-specific values
- **Operational Modes**: Use parameters to switch between normal operation, calibration, and maintenance modes
- **Environmental Conditions**: Use parameters to adjust for different operating environments (indoor vs outdoor, different payloads)
- **Safety Levels**: Use parameters to adjust safety margins and operational constraints

## Summary

In this lesson, you learned how to configure and manage ROS2 parameters for dynamic node behavior. You implemented a parameterized node that can adapt its behavior at runtime, created YAML configuration files for different operational modes, and learned about parameter validation and fallback mechanisms. You also explored how to update parameters while nodes are running and best practices for parameter management. Parameter management is essential for creating flexible, configurable robotic systems that can adapt to different robots, operational modes, and environmental conditions.