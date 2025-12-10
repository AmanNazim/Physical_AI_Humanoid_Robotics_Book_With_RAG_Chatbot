#!/usr/bin/env python3

"""
Service client for robot commands.
Implements client-side interaction with robot state services.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class RobotCommandClient(Node):
    """
    A ROS2 service client that sends commands to the robot state server.
    """

    def __init__(self):
        super().__init__('robot_command_client')

        # In a real implementation, we would create clients for services
        # For this example, we'll simulate the service interaction
        self.get_logger().info('Service client initialized (simulated)')

    def query_robot_state(self, query_type, target_frame=""):
        """
        Simulate sending a state query to the robot and returning a response.
        """
        self.get_logger().info(f'Simulating state query: {query_type}')

        # Simulate the service call
        if query_type == 'position':
            success = True
            message = "Position query successful"
            values = [0.5, -0.3, 1.2, 0.0, -0.8]  # Simulated positions
            names = ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4']
        elif query_type == 'status':
            success = True
            message = "Robot status: IDLE"
            values = [1.0]  # Calibrated
            names = ['is_calibrated']
        elif query_type == 'configuration':
            success = True
            message = "Configuration query successful"
            values = [5.0, 5.0, 10.0]  # [joint_count, max_speed, max_torque]
            names = ['joint_count', 'max_speed', 'max_torque']
        else:
            success = False
            message = f"Unknown query type: {query_type}"
            values = []
            names = []

        # Create a mock response object structure
        response = type('MockResponse', (), {})()
        response.success = success
        response.message = message
        response.values = values
        response.names = names

        self.get_logger().info(f'State query response: success={response.success}')
        self.get_logger().info(f'Message: {response.message}')
        self.get_logger().info(f'Values: {response.values}')
        self.get_logger().info(f'Names: {response.names}')
        return response

    def send_robot_command(self, command_type, parameters=None, target_frame=""):
        """
        Simulate sending a command to the robot and returning a response.
        """
        if parameters is None:
            parameters = []

        self.get_logger().info(f'Simulating command: {command_type} with parameters: {parameters}')

        # Simulate the command execution
        if command_type == 'move':
            success = True
            message = f"Move command executed with parameters: {parameters}"
            execution_time = 1.0
        elif command_type == 'stop':
            success = True
            message = "Robot stopped"
            execution_time = 0.1
        elif command_type == 'calibrate':
            success = True
            message = "Calibration completed"
            execution_time = 5.0
        else:
            success = False
            message = f"Unknown command type: {command_type}"
            execution_time = 0.0

        # Create a mock response object structure
        response = type('MockResponse', (), {})()
        response.success = success
        response.message = message
        response.execution_time = execution_time

        self.get_logger().info(f'Command response: success={response.success}')
        self.get_logger().info(f'Message: {response.message}')
        self.get_logger().info(f'Execution time: {response.execution_time}s')
        return response

    def run_demo_sequence(self):
        """
        Run a demonstration sequence of service calls.
        """
        self.get_logger().info('Starting service communication demo...')

        # Query initial robot position
        self.get_logger().info('\n--- Querying Initial Robot Position ---')
        self.query_robot_state('position')

        # Query robot status
        self.get_logger().info('\n--- Querying Robot Status ---')
        self.query_robot_state('status')

        # Query robot configuration
        self.get_logger().info('\n--- Querying Robot Configuration ---')
        self.query_robot_state('configuration')

        # Send a move command
        self.get_logger().info('\n--- Sending Move Command ---')
        move_params = [1.0, 0.5, -0.5, 0.0, 0.3]  # Example joint positions
        self.send_robot_command('move', move_params)

        # Query position after move
        self.get_logger().info('\n--- Querying Position After Move ---')
        self.query_robot_state('position')

        # Send a stop command
        self.get_logger().info('\n--- Sending Stop Command ---')
        self.send_robot_command('stop')

        # Send a calibration command
        self.get_logger().info('\n--- Sending Calibration Command ---')
        self.send_robot_command('calibrate')

        # Query status after calibration
        self.get_logger().info('\n--- Querying Status After Calibration ---')
        self.query_robot_state('status')

        self.get_logger().info('\n--- Demo Sequence Complete ---')


def main(args=None):
    rclpy.init(args=args)

    client = RobotCommandClient()

    try:
        # Run the demo sequence
        client.run_demo_sequence()
    except KeyboardInterrupt:
        client.get_logger().info('Client interrupted by user')
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()