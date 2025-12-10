---
sidebar_position: 3
---

# Service-based Communication

## Lesson Overview

In this lesson, you will learn to implement service-server and service-client communication patterns for synchronous operations in ROS2. Services provide request/response communication that is essential for operations requiring guaranteed completion, state queries, and configuration changes. You will understand when to use services versus topics, implement reliable service communication, and handle synchronous operations within the ROS2 framework.

## Learning Objectives

After completing this lesson, you will be able to:
- Implement service-server and service-client communication patterns
- Understand when to use services vs topics for different communication needs
- Handle timeout mechanisms and error responses in service communication
- Design proper service interfaces for robot state queries and configuration
- Implement reliable service communication with proper error handling

## Required Tools and Technologies

- ROS2 Humble Hawksbill
- rclpy (Python client library)
- Service definition files (.srv)
- colcon build system

## Understanding Service Communication

Service communication in ROS2 follows a request/response pattern where a client sends a request to a server and waits for a response. This synchronous communication is fundamentally different from the asynchronous topic-based communication you learned in previous lessons. Services are ideal for:

- State queries (e.g., "What is the current robot position?")
- Configuration changes (e.g., "Set the robot to safe mode")
- Operations requiring guaranteed completion (e.g., "Execute calibration sequence")
- Synchronous data processing (e.g., "Transform these coordinates")

The key characteristics of service communication are:
- Synchronous: The client waits for the response
- Request/response: One request generates one response
- Point-to-point: Direct communication between client and server
- Blocking: The client is blocked until the response is received

## When to Use Services vs Topics

Understanding when to use services versus topics is crucial for effective ROS2 design:

**Use Services When:**
- You need guaranteed completion of an operation
- The operation is synchronous by nature
- You need to query current state
- You're performing configuration changes
- You need to return results from computation
- The operation is not frequent (services are not for streaming data)

**Use Topics When:**
- You need asynchronous communication
- You're streaming data continuously
- Multiple subscribers need the same information
- You're implementing publish/subscribe patterns
- Real-time performance is critical (topics have lower latency)
- You're broadcasting information to multiple nodes

## Creating Custom Service Definitions

Let's create a custom service definition for robot state queries. First, create the service definition file in your package:

Create the services directory in your package:

```bash
mkdir -p advanced_communication_tutorials/advanced_communication_tutorials/srv
```

Create a service definition file for robot state queries (`advanced_communication_tutorials/advanced_communication_tutorials/srv/RobotStateQuery.srv`):

```
# Request: Query type and parameters
string query_type  # Type of query: "position", "status", "configuration", etc.
string target_frame  # Optional target frame for transforms
---
# Response: Query result and success status
bool success  # Whether the query was successful
string message  # Additional information or error message
float64[] values  # Numeric values (positions, angles, etc.)
string[] names  # Names corresponding to the values
```

Create another service for robot commands (`advanced_communication_tutorials/advanced_communication_tutorials/srv/RobotCommand.srv`):

```
# Request: Command type and parameters
string command_type  # Type of command: "move", "stop", "calibrate", etc.
float64[] parameters  # Command parameters (positions, velocities, etc.)
string target_frame  # Optional target frame
---
# Response: Command execution result
bool success  # Whether the command was accepted/started
string message  # Additional information or error message
float64 execution_time  # Time taken to execute the command
```

Update the package.xml to include the service definitions:

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
  <depend>geometry_msgs</depend>
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

Update the setup.py to include service definitions:

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
        ],
    },
)
```

## Implementing a Service Server

Now let's implement a service server that can handle robot state queries. Create the service server file (`advanced_communication_tutorials/advanced_communication_tutorials/robot_state_server.py`):

```python
#!/usr/bin/env python3

"""
Service server for robot state queries.
Implements RobotStateQuery service to provide current robot state information.
"""

import rclpy
from rclpy.node import Node
from advanced_communication_tutorials.srv import RobotStateQuery, RobotCommand
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import math


class RobotStateServer(Node):
    """
    A ROS2 service server that provides robot state information.
    """

    def __init__(self):
        super().__init__('robot_state_server')

        # Create service servers
        self.state_query_service = self.create_service(
            RobotStateQuery,
            'get_robot_state',
            self.handle_state_query
        )

        self.command_service = self.create_service(
            RobotCommand,
            'execute_robot_command',
            self.handle_robot_command
        )

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

    def handle_state_query(self, request, response):
        """
        Handle robot state query requests.
        """
        self.get_logger().info(f'Received state query: {request.query_type}')

        if request.query_type == 'position':
            response.success = True
            response.message = "Position query successful"
            response.values = self.joint_positions
            response.names = [f'joint_{i}' for i in range(len(self.joint_positions))]

        elif request.query_type == 'status':
            response.success = True
            response.message = f"Robot status: {self.robot_status}"
            response.values = [1.0 if self.is_calibrated else 0.0]  # Calibration status
            response.names = ['is_calibrated']

        elif request.query_type == 'configuration':
            response.success = True
            response.message = "Configuration query successful"
            response.values = [len(self.joint_positions), 5.0, 10.0]  # [joint_count, max_speed, max_torque]
            response.names = ['joint_count', 'max_speed', 'max_torque']

        else:
            response.success = False
            response.message = f"Unknown query type: {request.query_type}"
            response.values = []
            response.names = []

        self.get_logger().info(f'Responding to query: success={response.success}, message={response.message}')
        return response

    def handle_robot_command(self, request, response):
        """
        Handle robot command requests.
        """
        self.get_logger().info(f'Received command: {request.command_type}')

        # Simulate command execution
        execution_time = 0.0

        if request.command_type == 'move':
            if len(request.parameters) > 0:
                # Simulate moving to new positions
                self.joint_positions = list(request.parameters)[:len(self.joint_positions)]
                self.robot_status = "MOVING"
                execution_time = 1.0  # Simulated execution time
                response.success = True
                response.message = f"Move command executed, new positions: {self.joint_positions}"
            else:
                response.success = False
                response.message = "Move command requires parameters"

        elif request.command_type == 'stop':
            self.robot_status = "STOPPED"
            execution_time = 0.1  # Quick stop
            response.success = True
            response.message = "Robot stopped"

        elif request.command_type == 'calibrate':
            self.is_calibrated = True
            self.robot_status = "CALIBRATED"
            execution_time = 5.0  # Calibration takes longer
            response.success = True
            response.message = "Calibration completed"

        else:
            response.success = False
            response.message = f"Unknown command type: {request.command_type}"

        response.execution_time = execution_time
        self.get_logger().info(f'Responding to command: success={response.success}, message={response.message}')
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
```

## Implementing a Service Client

Now let's implement a service client that can interact with our service server. Create the service client file (`advanced_communication_tutorials/advanced_communication_tutorials/robot_command_client.py`):

```python
#!/usr/bin/env python3

"""
Service client for robot commands.
Implements client-side interaction with robot state services.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from advanced_communication_tutorials.srv import RobotStateQuery, RobotCommand
from std_msgs.msg import String
import time


class RobotCommandClient(Node):
    """
    A ROS2 service client that sends commands to the robot state server.
    """

    def __init__(self):
        super().__init__('robot_command_client')

        # Create clients for the services
        self.state_client = self.create_client(RobotStateQuery, 'get_robot_state')
        self.command_client = self.create_client(RobotCommand, 'execute_robot_command')

        # Wait for services to be available
        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('State service not available, waiting again...')

        while not self.command_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Command service not available, waiting again...')

        self.get_logger().info('Service clients initialized and connected')

    def query_robot_state(self, query_type, target_frame=""):
        """
        Send a state query to the robot and return the response.
        """
        request = RobotStateQuery.Request()
        request.query_type = query_type
        request.target_frame = target_frame

        self.get_logger().info(f'Sending state query: {query_type}')

        # Make the service call with timeout
        future = self.state_client.call_async(request)

        # Wait for response with timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'State query response: success={response.success}')
            self.get_logger().info(f'Message: {response.message}')
            self.get_logger().info(f'Values: {response.values}')
            self.get_logger().info(f'Names: {response.names}')
            return response
        else:
            self.get_logger().error('Exception while calling service: %r' % future.exception())
            return None

    def send_robot_command(self, command_type, parameters=None, target_frame=""):
        """
        Send a command to the robot and return the response.
        """
        if parameters is None:
            parameters = []

        request = RobotCommand.Request()
        request.command_type = command_type
        request.parameters = parameters
        request.target_frame = target_frame

        self.get_logger().info(f'Sending command: {command_type} with parameters: {parameters}')

        # Make the service call with timeout
        future = self.command_client.call_async(request)

        # Wait for response with timeout
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Command response: success={response.success}')
            self.get_logger().info(f'Message: {response.message}')
            self.get_logger().info(f'Execution time: {response.execution_time}s')
            return response
        else:
            self.get_logger().error('Exception while calling service: %r' % future.exception())
            return None

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
```

## Timeout Handling and Error Responses

Service communication must include proper timeout handling to prevent clients from hanging indefinitely. In our implementation, we use `rclpy.spin_until_future_complete()` with a timeout parameter to ensure that service calls don't block forever.

Error handling in services involves:
- Checking if the service is available before making calls
- Setting appropriate timeout values based on expected service response time
- Handling exceptions that may occur during service calls
- Providing meaningful error messages in service responses

## Testing the Service Communication

To test the service communication, first build the package with the new service definitions:

```bash
cd ~/ros2_ws
colcon build --packages-select advanced_communication_tutorials
source install/setup.bash
```

Run the service server in one terminal:

```bash
ros2 run advanced_communication_tutorials robot_state_server
```

In another terminal, run the service client:

```bash
ros2 run advanced_communication_tutorials robot_command_client
```

You can also test services manually using the command line:

```bash
# Query robot state
ros2 service call /get_robot_state advanced_communication_tutorials/srv/RobotStateQuery "{'query_type': 'position'}"

# Send a move command
ros2 service call /execute_robot_command advanced_communication_tutorials/srv/RobotCommand "{'command_type': 'move', 'parameters': [1.0, 0.5, -0.5]}"
```

## Service Interface Design Best Practices

When designing service interfaces for robot systems, consider these best practices:

1. **Clear Request/Response Structure**: Define clear input parameters and output responses that match the operation's purpose.

2. **Error Handling**: Always include success/failure indicators and descriptive error messages in responses.

3. **Appropriate Timeout Values**: Set timeout values based on the expected execution time of the service.

4. **State Consistency**: Ensure services maintain consistent robot state and don't leave the system in an inconsistent state.

5. **Validation**: Validate input parameters before executing service operations.

6. **Documentation**: Clearly document what each service does, its parameters, and expected behavior.

## Comparison: Services vs Topics

| Aspect | Services | Topics |
|--------|----------|--------|
| Communication Type | Synchronous (request/response) | Asynchronous (publish/subscribe) |
| Blocking | Client blocks until response | Non-blocking |
| Use Case | State queries, commands, configuration | Data streaming, broadcasting |
| Latency | Higher (due to round-trip) | Lower |
| Reliability | Guaranteed response | Best-effort delivery |
| Frequency | Not suitable for high-frequency | Ideal for high-frequency |

## Summary

In this lesson, you learned how to implement service-based communication in ROS2. You created custom service definitions, implemented both service servers and clients, and learned about proper timeout handling and error responses. You also understood when to use services versus topics based on the communication requirements. Service communication is essential for synchronous operations in robotic systems, particularly for state queries, commands, and configuration changes that require guaranteed completion.