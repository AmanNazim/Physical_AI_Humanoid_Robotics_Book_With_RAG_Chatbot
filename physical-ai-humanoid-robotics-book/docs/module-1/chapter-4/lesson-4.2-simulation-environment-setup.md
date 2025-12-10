---
title: Lesson 4.2 - Simulation Environment Setup
---

# Lesson 4.2 – Simulation Environment Setup

## Learning Objectives

By the end of this lesson, you will be able to:
- Interface Python nodes with Gazebo simulation controllers
- Build perception-to-action pipelines that work in simulation
- Test simulation in Gazebo environment
- Validate simulation-ready configurations
- Configure robots for basic simulation in Gazebo or similar environments

## Concept Overview and Scope

This lesson focuses on setting up and configuring simulation environments for humanoid robots using Gazebo, a powerful 3D simulation environment that provides accurate physics simulation, high-quality graphics, and convenient programmatic interfaces. We'll learn how to connect our Python-based AI agents to the simulated robot, enabling testing and validation of our perception-to-action pipelines in a safe, repeatable environment.

Simulation is crucial for robotics development as it allows us to test complex behaviors without risk of hardware damage, enables faster iteration cycles, and provides controlled environments for debugging.

## Understanding Gazebo and ROS2 Integration

Gazebo provides a realistic physics simulation environment that can be integrated with ROS2 through the Gazebo ROS packages. The integration works through:

1. **Gazebo Plugins**: These provide ROS2 interfaces to simulated sensors and actuators
2. **ROS2 Control**: For commanding simulated joints
3. **TF Transforms**: For robot state visualization
4. **Sensor Simulation**: For realistic sensor data generation

### Key Components of Gazebo-ROS2 Integration:

1. **gazebo_ros_pkgs**: Provides the bridge between Gazebo and ROS2
2. **ros_gz**: Modern bridge for Gazebo Garden/Harmonic
3. **Controller Manager**: Manages robot controllers in simulation
4. **Robot State Publisher**: Publishes robot state for visualization

## Setting Up Gazebo with Your Robot Model

To simulate your robot in Gazebo, you need to ensure your URDF model is properly configured for simulation. Here are the key elements:

### 1. Adding Gazebo-Specific Tags to URDF

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <!-- Include the robot's physical description -->
  <xacro:include filename="my_robot.urdf.xacro"/>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros2_control.so">
      <parameters>$(find my_robot_description)/config/my_robot_controllers.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Gazebo material definitions -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Sensor plugins -->
  <gazebo reference="camera_link">
    <sensor type="camera" name="camera_sensor">
      <update_rate>30</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>800</width>
          <height>600</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>100</far>
        </clip>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

### 2. Controller Configuration

Create a controller configuration file (`config/my_robot_controllers.yaml`):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    velocity_controller:
      type: velocity_controllers/JointGroupVelocityController

    position_controller:
      type: position_controllers/JointGroupPositionController

velocity_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3

position_controller:
  ros__parameters:
    joints:
      - joint1
      - joint2
      - joint3
```

## Launching Gazebo with Your Robot

Create a launch file to start Gazebo with your robot:

### Python Launch File

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Package names
    pkg_share = FindPackageShare("my_robot_description").find("my_robot_description")
    gazebo_pkg_share = FindPackageShare("gazebo_ros").find("gazebo_ros")

    # Arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Robot description
    robot_description_content = Command([
        'xacro ',
        PathJoinSubstitution([pkg_share, 'urdf', 'my_robot.urdf.xacro'])
    ])

    # Robot state publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description_content
        }]
    )

    # Spawn entity in Gazebo
    spawn_entity_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_pkg_share, 'launch', 'gazebo.launch.py')
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time if true'
        ),
        gazebo_launch,
        robot_state_publisher_node,
        spawn_entity_node
    ])
```

### XML Launch File

```xml
<launch>
  <!-- Arguments -->
  <arg name="use_sim_time" default="true"/>
  <arg name="world" default="empty"/>

  <!-- Start Gazebo -->
  <include file="$(find-pkg-share gazebo_ros)/launch/gazebo.launch.py">
    <arg name="world" value="$(var world)"/>
    <arg name="gui" value="true"/>
  </include>

  <!-- Robot State Publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="use_sim_time" value="$(var use_sim_time)"/>
    <param name="robot_description" value="$(command 'xacro $(find-pkg-share my_robot_description)/urdf/my_robot.urdf.xacro')"/>
  </node>

  <!-- Spawn robot in Gazebo -->
  <node pkg="gazebo_ros" exec="spawn_entity.py" args="-topic robot_description -entity my_robot -x 0 -y 0 -z 1.0" name="spawn_entity" output="screen"/>

  <!-- Load controllers -->
  <node pkg="controller_manager" exec="ros2_control_node" name="ros2_control_node" output="both">
    <param name="robot_description" value="$(command 'xacro $(find-pkg-share my_robot_description)/urdf/my_robot.urdf.xacro')"/>
    <param name="use_sim_time" value="$(var use_sim_time)"/>
  </node>
</launch>
```

## Interface Python Nodes with Gazebo Simulation

Now let's create Python nodes that can interface with the Gazebo simulation:

### 1. Basic Simulation Controller Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
import math

class SimulationControllerNode(Node):
    def __init__(self):
        super().__init__('simulation_controller')

        # Publisher for joint commands
        self.joint_cmd_publisher = self.create_publisher(
            Float64MultiArray,
            '/velocity_controller/commands',
            10
        )

        # Subscription to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        self.current_joint_positions = []
        self.get_logger().info('Simulation Controller Node has been started')

    def joint_state_callback(self, msg):
        self.current_joint_positions = list(msg.position)

    def control_loop(self):
        # Example: Simple sinusoidal motion for demonstration
        cmd_msg = Float64MultiArray()

        # Calculate commands based on current time
        t = self.get_clock().now().nanoseconds / 1e9

        # Example: Move joints in a coordinated pattern
        commands = []
        for i in range(len(self.current_joint_positions)):
            command = math.sin(t + i * 0.5) * 0.5  # Amplitude of 0.5 rad/s
            commands.append(command)

        cmd_msg.data = commands
        self.joint_cmd_publisher.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationControllerNode()

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

### 2. Perception Processing Node for Simulation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class SimulationPerceptionNode(Node):
    def __init__(self):
        super().__init__('simulation_perception')

        # Subscriptions
        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.camera_callback,
            10
        )

        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        self.latest_scan = None
        self.latest_image = None

        # Timer for processing loop
        self.process_timer = self.create_timer(0.1, self.process_data)

        self.get_logger().info('Simulation Perception Node has been started')

    def laser_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def process_data(self):
        if self.latest_scan is None:
            return

        # Simple obstacle avoidance based on laser scan
        cmd = Twist()

        # Get distances in front, left, and right sectors
        scan_ranges = self.latest_scan.ranges
        front_distances = scan_ranges[:len(scan_ranges)//8] + scan_ranges[-len(scan_ranges)//8:]
        left_distances = scan_ranges[len(scan_ranges)*3//8:len(scan_ranges)*5//8]
        right_distances = scan_ranges[len(scan_ranges)*5//8:len(scan_ranges)*7//8]

        min_front = min(front_distances) if front_distances else float('inf')
        min_left = min(left_distances) if left_distances else float('inf')
        min_right = min(right_distances) if right_distances else float('inf')

        # Simple navigation logic
        if min_front < 0.8:  # Obstacle ahead
            cmd.linear.x = 0.0
            if min_left > min_right:
                cmd.angular.z = 0.5  # Turn left
            else:
                cmd.angular.z = -0.5  # Turn right
        else:
            cmd.linear.x = 0.5  # Move forward

        self.cmd_vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationPerceptionNode()

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

## Building Perception-to-Action Pipelines

In simulation, we can build complete perception-to-action pipelines that mirror real-world behavior:

### Complete Pipeline Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

class PerceptionToActionPipeline(Node):
    def __init__(self):
        super().__init__('perception_to_action_pipeline')

        # Subscriptions for different sensor modalities
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, 'camera/image_raw', self.camera_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Publishers for different action modalities
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.behavior_status_pub = self.create_publisher(String, 'behavior_status', 10)

        # CV Bridge for image processing
        self.cv_bridge = CvBridge()

        # State variables
        self.latest_scan = None
        self.latest_image = None
        self.latest_joints = None
        self.current_behavior = "EXPLORING"

        # Timer for main processing loop
        self.pipeline_timer = self.create_timer(0.05, self.pipeline_loop)  # 20Hz

        self.get_logger().info('Perception-to-Action Pipeline has been started')

    def laser_callback(self, msg):
        self.latest_scan = msg

    def camera_callback(self, msg):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')

    def joint_callback(self, msg):
        self.latest_joints = msg

    def pipeline_loop(self):
        if self.latest_scan is None:
            return

        # Process sensor data to make decisions
        behavior_command = self.make_decision()

        # Execute the behavior
        self.execute_behavior(behavior_command)

    def make_decision(self):
        """Process sensor data and make behavioral decisions"""
        if self.latest_scan is None:
            return {"linear": 0.0, "angular": 0.0, "behavior": "WAITING"}

        # Analyze laser scan for obstacles
        scan_ranges = self.latest_scan.ranges
        front_distances = scan_ranges[:len(scan_ranges)//8] + scan_ranges[-len(scan_ranges)//8:]
        min_front = min(front_distances) if front_distances else float('inf')

        # Analyze image for specific features (simplified)
        if self.latest_image is not None:
            # Example: Check for red objects in image
            hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
            red_pixels = cv2.countNonZero(red_mask)

            if red_pixels > 1000:  # If significant red detected
                self.current_behavior = "APPROACH_RED"
                return {"linear": 0.3, "angular": 0.0, "behavior": "APPROACH_RED"}

        # Standard obstacle avoidance
        if min_front < 0.8:
            self.current_behavior = "AVOIDING_OBSTACLE"
            # Turn in the direction with more space
            left_distances = scan_ranges[len(scan_ranges)*3//8:len(scan_ranges)*5//8]
            right_distances = scan_ranges[len(scan_ranges)*5//8:len(scan_ranges)*7//8]
            min_left = min(left_distances) if left_distances else float('inf')
            min_right = min(right_distances) if right_distances else float('inf')

            if min_left > min_right:
                return {"linear": 0.0, "angular": 0.5, "behavior": "AVOIDING_OBSTACLE"}
            else:
                return {"linear": 0.0, "angular": -0.5, "behavior": "AVOIDING_OBSTACLE"}
        else:
            self.current_behavior = "EXPLORING"
            return {"linear": 0.5, "angular": 0.0, "behavior": "EXPLORING"}

    def execute_behavior(self, command):
        """Execute the decided behavior"""
        cmd = Twist()
        cmd.linear.x = command["linear"]
        cmd.angular.z = command["angular"]

        self.cmd_vel_pub.publish(cmd)

        # Publish behavior status
        status_msg = String()
        status_msg.data = command["behavior"]
        self.behavior_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionToActionPipeline()

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

## Validation Techniques for Simulation

Proper validation ensures your robot behaves correctly in simulation:

### 1. Sensor Data Validation

```python
def validate_sensor_data(self, sensor_msg):
    """Validate that sensor data is reasonable"""
    if sensor_msg is None:
        return False

    # Check for NaN or infinite values
    if hasattr(sensor_msg, 'ranges'):
        for range_val in sensor_msg.ranges:
            if math.isnan(range_val) or math.isinf(range_val):
                self.get_logger().warning(f'Invalid range value: {range_val}')
                return False

    return True
```

### 2. Control Command Validation

```python
def validate_control_command(self, cmd):
    """Validate control commands before sending to robot"""
    if abs(cmd.linear.x) > self.max_linear_velocity:
        cmd.linear.x = math.copysign(self.max_linear_velocity, cmd.linear.x)
        self.get_logger().warning(f'Linear velocity clamped to {self.max_linear_velocity}')

    if abs(cmd.angular.z) > self.max_angular_velocity:
        cmd.angular.z = math.copysign(self.max_angular_velocity, cmd.angular.z)
        self.get_logger().warning(f'Angular velocity clamped to {self.max_angular_velocity}')

    return cmd
```

## Common Simulation Issues and Solutions

### Issue 1: Robot Falls Through Ground
**Symptoms**: Robot falls through the ground or other static objects
**Solutions**:
1. Check that collision geometries are properly defined
2. Verify that the robot has proper mass and inertia properties
3. Ensure physics parameters are set correctly in Gazebo

### Issue 2: Joint Commands Not Working
**Symptoms**: Joint commands sent to the robot have no effect
**Solutions**:
1. Verify controller configuration files are correct
2. Check that controller manager is running
3. Ensure proper topic names and message types

### Issue 3: Sensor Data Not Publishing
**Symptoms**: Sensor topics are not publishing data
**Solutions**:
1. Check Gazebo sensor plugin configuration
2. Verify that sensors are properly attached to links
3. Confirm that physics simulation is running

### Issue 4: High CPU Usage
**Symptoms**: Simulation runs slowly or uses excessive CPU
**Solutions**:
1. Reduce physics update rate in Gazebo
2. Simplify collision geometries
3. Limit the number of sensors in simulation

## Step-by-Step Exercise

Create a complete simulation setup for your robot:

1. Add Gazebo plugins to your URDF file
2. Create a controller configuration file
3. Create a launch file to start Gazebo with your robot
4. Implement a Python node that controls your simulated robot
5. Test the simulation and validate the behavior

## Summary

In this lesson, you learned:
- How to configure your robot model for Gazebo simulation
- How to interface Python nodes with Gazebo simulation controllers
- How to build perception-to-action pipelines that work in simulation
- How to validate simulation-ready configurations
- Common issues and solutions in robot simulation

Simulation provides a safe, repeatable environment for testing and validating your robot's behavior before deploying to real hardware.

## Next Steps

In the final lesson of this module, we'll integrate all components into a complete system and perform end-to-end validation of our perception-to-action pipeline in both simulation and preparation for real hardware.