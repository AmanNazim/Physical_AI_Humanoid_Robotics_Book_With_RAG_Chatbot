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