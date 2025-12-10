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