#!/usr/bin/env python3

"""
Service server for robot state queries.
Implements RobotStateQuery service to provide current robot state information.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import math


class RobotStateServer(Node):
    """
    A ROS2 service server that provides robot state information.
    """

    def __init__(self):
        super().__init__('robot_state_server')

        # Define service interfaces (in a real implementation, you would import these from generated files)
        # For this example, we'll simulate the service functionality

        # Store some simulated robot state
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0]  # Simulated joint positions
        self.robot_status = "IDLE"  # Robot operational status
        self.is_calibrated = True  # Calibration status

        # Subscribe to actual joint states (if available) to update our simulation
        self.joint_state_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Robot State Server initialized and ready')

    def joint_state_callback(self, msg):
        """
        Update internal state based on received joint states.
        """
        if len(msg.position) > 0:
            self.joint_positions = list(msg.position)[:5]  # Take first 5 joints
            self.get_logger().debug(f'Updated joint positions: {self.joint_positions}')

    def handle_state_query(self, query_type, target_frame=""):
        """
        Simulate handling robot state query requests.
        """
        self.get_logger().info(f'Received state query: {query_type}')

        if query_type == 'position':
            success = True
            message = "Position query successful"
            values = self.joint_positions
            names = [f'joint_{i}' for i in range(len(self.joint_positions))]
        elif query_type == 'status':
            success = True
            message = f"Robot status: {self.robot_status}"
            values = [1.0 if self.is_calibrated else 0.0]  # Calibration status
            names = ['is_calibrated']
        elif query_type == 'configuration':
            success = True
            message = "Configuration query successful"
            values = [len(self.joint_positions), 5.0, 10.0]  # [joint_count, max_speed, max_torque]
            names = ['joint_count', 'max_speed', 'max_torque']
        else:
            success = False
            message = f"Unknown query type: {query_type}"
            values = []
            names = []

        self.get_logger().info(f'Responding to query: success={success}, message={message}')

        # Create a mock response object structure
        response = type('MockResponse', (), {})()
        response.success = success
        response.message = message
        response.values = values
        response.names = names
        return response

    def handle_robot_command(self, command_type, parameters=None, target_frame=""):
        """
        Simulate handling robot command requests.
        """
        if parameters is None:
            parameters = []

        self.get_logger().info(f'Received command: {command_type}')

        # Simulate command execution
        execution_time = 0.0

        if command_type == 'move':
            if len(parameters) > 0:
                # Simulate moving to new positions
                self.joint_positions = list(parameters)[:len(self.joint_positions)]
                self.robot_status = "MOVING"
                execution_time = 1.0  # Simulated execution time
                success = True
                message = f"Move command executed, new positions: {self.joint_positions}"
            else:
                success = False
                message = "Move command requires parameters"

        elif command_type == 'stop':
            self.robot_status = "STOPPED"
            execution_time = 0.1  # Quick stop
            success = True
            message = "Robot stopped"

        elif command_type == 'calibrate':
            self.is_calibrated = True
            self.robot_status = "CALIBRATED"
            execution_time = 5.0  # Calibration takes longer
            success = True
            message = "Calibration completed"

        else:
            success = False
            message = f"Unknown command type: {command_type}"

        # Create a mock response object structure
        response = type('MockResponse', (), {})()
        response.success = success
        response.message = message
        response.execution_time = execution_time
        self.get_logger().info(f'Responding to command: success={success}, message={response.message}')
        return response


def main(args=None):
    rclpy.init(args=args)

    server = RobotStateServer()

    try:
        rclpy.spin(server)
    except KeyboardInterrupt:
        server.get_logger().info('Service server interrupted by user')
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()