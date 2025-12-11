---
title: Lesson 3.3 - Visualization and Validation
---

# Lesson 3.3 – Visualization and Validation

## Learning Objectives

By the end of this lesson, you will be able to:
- Integrate URDF/Xacro models with Robot State Publisher
- Visualize robot models in RViz
- Add visual and collision properties to robot links
- Create launch files for visualization
- Validate URDF kinematic properties and transformations
- Troubleshoot common visualization issues

## Concept Overview and Scope

In this lesson, we'll focus on bringing our robot descriptions to life by visualizing them in ROS2 tools and validating their correctness. Visualization is crucial for understanding robot structure, debugging kinematic issues, and ensuring that our robot models are properly defined before moving to simulation and control.

We'll learn how to use Robot State Publisher to broadcast transforms and RViz to visualize our robot models in 3D space.

## Robot State Publisher Integration

Robot State Publisher is a ROS2 node that reads a URDF model and publishes the appropriate transforms to tf2 (the transform library) using the robot model and joint positions. This allows visualization tools like RViz to display the robot correctly.

### How Robot State Publisher Works

Robot State Publisher subscribes to the `/joint_states` topic to receive information about the current state of the robot's joints. It then calculates the transforms between all the links based on the URDF model and the joint states, and publishes these transforms to the `/tf` topic.

### Basic Robot State Publisher Launch

```xml
<launch>
  <!-- Load the robot description from Xacro -->
  <param name="robot_description"
         value="$(command 'xacro $(find-pkg-share my_robot_description)/urdf/robot.xacro')"/>

  <!-- Launch robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description"
           value="$(command 'xacro $(find-pkg-share my_robot_description)/urdf/robot.xacro')"/>
  </node>
</launch>
```

### Robot Description Parameter

The robot description is typically loaded as a parameter and contains the entire URDF/Xacro content. This parameter is used by Robot State Publisher and other nodes that need to understand the robot's structure.

## Setting Up RViz for Robot Visualization

RViz is ROS2's 3D visualization tool that allows us to visualize robots, sensor data, paths, and other information in a 3D environment.

### Basic RViz Configuration

To visualize a robot in RViz, you need to:

1. Add a RobotModel display
2. Set the Robot Description parameter to match your robot_description
3. Ensure TF (Transforms) are being published

### RViz Display Configuration

In RViz, you'll need to add these displays:
- **RobotModel**: Shows the 3D model of your robot
- **TF**: Shows the transform frames and their relationships
- **Grid**: Provides a reference frame for orientation

### Sample RViz Configuration File

```yaml
# config/robot_visualization.rviz
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views

Visualization Manager:
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Name: Grid
      Value: true
    - Class: rviz_default_plugins/RobotModel
      Name: RobotModel
      Description Topic:
        Value: /joint_states
      Robot Description: robot_description
      TF Prefix: ""
    - Class: rviz_default_plugins/TF
      Name: TF
      Frame Timeout: 15
      Show Arrows: true
      Show Names: false
      Value: true

  Views:
    Current:
      Class: rviz_default_plugins/Orbit
```

## Creating Visualization Launch Files

Launch files allow us to start multiple nodes together to visualize our robot:

### Complete Visualization Launch File

```xml
<launch>
  <!-- Declare arguments -->
  <arg name="model" default="$(find-pkg-share my_robot_description)/urdf/robot.xacro"/>
  <arg name="rvizconfig" default="$(find-pkg-share my_robot_description)/rviz/robot_visualization.rviz"/>

  <!-- Load the robot description -->
  <param name="robot_description"
         value="$(command 'xacro $(var model)')"/>

  <!-- Launch robot state publisher -->
  <node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot_state_publisher">
    <param name="robot_description"
           value="$(command 'xacro $(var model)')"/>
  </node>

  <!-- Launch joint state publisher (for interactive joint control) -->
  <node pkg="joint_state_publisher_gui" exec="joint_state_publisher_gui" name="joint_state_publisher_gui">
  </node>

  <!-- Launch RViz -->
  <node pkg="rviz2" exec="rviz2" name="rviz2" args="-d $(var rvizconfig)">
  </node>
</launch>
```

### Joint State Publisher

The Joint State Publisher provides a GUI to manually control joint positions, which is useful for visualizing how your robot moves:

- **joint_state_publisher**: Provides static joint states
- **joint_state_publisher_gui**: Provides an interactive GUI to control joint positions

## Adding Visual and Collision Properties

Proper visual and collision properties are essential for both visualization and simulation:

### Visual Properties

Visual properties define how the robot appears in visualization tools:

```xml
<link name="link_name">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Options: box, cylinder, sphere, mesh -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
</link>
```

### Collision Properties

Collision properties define the physical boundaries for simulation:

```xml
<link name="link_name">
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Often simpler than visual geometry for performance -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

Inertial properties are crucial for physics simulation:

```xml
<link name="link_name">
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

## Validation Techniques for URDF Models

Proper validation ensures your robot model is correct and ready for simulation:

### 1. Syntax Validation

Check that your URDF/Xacro is valid XML:

```bash
# Validate Xacro syntax
xacro my_robot.xacro

# Convert to URDF and check syntax
xacro my_robot.xacro > my_robot.urdf
check_urdf my_robot.urdf
```

### 2. Kinematic Validation

Verify that your kinematic chains are properly formed:

```bash
# Check the kinematic structure
check_urdf my_robot.urdf
```

This command will show you:
- The number of links and joints
- The kinematic tree structure
- Any issues with the model

### 3. Transform Validation

Use RViz to visually inspect transforms:

- Check that all links appear correctly positioned
- Verify that joint movements are as expected
- Ensure there are no disconnected links

### 4. Physics Validation

For simulation, verify inertial properties:

- Mass values should be realistic
- Inertia values should be physically plausible
- Center of mass should be reasonable

## Common Visualization Issues and Solutions

### Issue 1: Robot Not Appearing in RViz

**Symptoms**: Robot model doesn't show up in RViz

**Solutions**:
1. Check that Robot State Publisher is running
2. Verify the robot_description parameter is set correctly
3. Ensure the URDF/Xacro file path is correct
4. Check RViz RobotModel display settings

### Issue 2: Incorrect Joint Positions

**Symptoms**: Robot appears in unexpected poses

**Solutions**:
1. Check Joint State Publisher is publishing values
2. Verify joint names match between URDF and joint states
3. Ensure joint limits are properly defined

### Issue 3: Disconnected Links

**Symptoms**: Parts of the robot float independently

**Solutions**:
1. Verify all joints have proper parent-child relationships
2. Check for typos in link names
3. Ensure all required transforms are being published

### Issue 4: Performance Issues

**Symptoms**: RViz running slowly with complex models

**Solutions**:
1. Simplify collision geometry (keep visual geometry complex)
2. Reduce the number of meshes if using many
3. Use simpler shapes where possible

## Advanced Visualization Techniques

### 1. Multiple Robot Visualization

Visualize multiple robots with different TF prefixes:

```xml
<node pkg="robot_state_publisher" exec="robot_state_publisher" name="robot1_state_publisher">
  <param name="robot_description"
         value="$(command 'xacro $(find-pkg-share my_robot_description)/urdf/robot1.xacro')"/>
  <param name="tf_prefix" value="robot1"/>
</node>
```

### 2. Interactive Markers

Add interactive markers for manual control:

```xml
<node pkg="interactive_marker_tutorials" exec="basic_controls" name="basic_controls">
</node>
```

### 3. Sensor Visualization

Add sensor data visualization to your robot:

```xml
<!-- Add LaserScan display in RViz -->
- Class: rviz_default_plugins/LaserScan
  Topic: /scan
  Value: true
```

## Complete Example: Visualization Setup

Let's put together a complete visualization setup for a robot:

### 1. URDF/Xacro File (`my_robot.xacro`)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">
  <xacro:property name="PI" value="3.14159"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.2" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Rotating platform -->
  <link name="rotating_platform">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
  </link>

  <joint name="platform_joint" type="continuous">
    <parent link="base_link"/>
    <child link="rotating_platform"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
```

### 2. Launch File (`launch/visualize_robot.launch.py`)

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the package directory
    pkg_share = get_package_share_directory('my_robot_description')

    # Declare arguments
    model_path = os.path.join(pkg_share, 'urdf', 'my_robot.xacro')

    # Robot description
    robot_desc = Command(['xacro ', model_path])

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(robot_desc, value_type=str)}]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher'
    )

    # Joint state publisher GUI
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui'
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_share, 'rviz', 'robot_visualization.rviz')]
    )

    return LaunchDescription([
        robot_state_publisher,
        joint_state_publisher,
        joint_state_publisher_gui,
        rviz
    ])
```

## Step-by-Step Exercise

Create a complete visualization setup for the humanoid robot from previous lessons:

1. Take your URDF/Xacro model from Lesson 3.2
2. Create a launch file to visualize it in RViz
3. Launch the visualization and verify all parts appear correctly
4. Use the Joint State Publisher GUI to move the joints
5. Verify that the kinematic chains work properly

## Summary

In this lesson, you learned:
- How to integrate URDF/Xacro models with Robot State Publisher
- How to visualize robots in RViz with proper configuration
- How to create launch files for complete visualization setups
- How to add visual and collision properties for proper display
- Various validation techniques for URDF models
- How to troubleshoot common visualization issues

Visualization and validation are critical steps in robot development, ensuring that our robot descriptions are accurate and ready for simulation and control. These tools help us verify our designs before moving to more complex implementations.

## Next Steps

With your robot properly described, visualized, and validated, you're now ready to connect Python-based agents with ROS2 controllers using rclpy, which will be the focus of Chapter 4. Your robot model will serve as the foundation for simulation and control in the upcoming chapters.