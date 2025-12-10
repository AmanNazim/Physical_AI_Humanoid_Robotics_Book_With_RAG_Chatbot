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