# Lesson 1.3 – Robot Integration in Gazebo

## Learning Objectives

By the end of this lesson, you will be able to:
- Import and configure humanoid robots in Gazebo simulation from URDF models
- Convert URDF to SDF format for Gazebo compatibility
- Configure joint constraints and collision properties for humanoid robots
- Understand the relationship between URDF and SDF formats
- Test and validate robot integration in the Gazebo environment

## Introduction to Robot Integration

Now that you have Gazebo installed and can create custom environments, it's time to bring your humanoid robots into the simulation. This lesson focuses on integrating the robot models you created in Module 1 (in URDF format) into the Gazebo simulation environment.

### Why URDF to SDF Conversion?

Gazebo uses SDF (Simulation Description Format) as its native format, while ROS 2 typically uses URDF (Unified Robot Description Format). The conversion process ensures that your robot models can be properly interpreted by Gazebo's physics engine and visualization system.

## Understanding URDF to SDF Conversion

### Key Differences Between URDF and SDF

While both formats describe robot models, they have different focuses:
- **URDF**: Primarily for kinematic and geometric description
- **SDF**: Comprehensive format for simulation including physics properties

### Conversion Process

The conversion from URDF to SDF typically involves:
1. **Geometry conversion**: Translating visual and collision geometries
2. **Physics property addition**: Adding mass, inertia, and friction parameters
3. **Simulation-specific elements**: Adding Gazebo-specific plugins and properties

## Converting URDF to SDF Format

### Method 1: Using the gz sdf Command

The most straightforward way to convert URDF to SDF is using Gazebo's command-line tools:

```bash
# Convert a URDF file to SDF
gz sdf -p /path/to/your/robot.urdf > /path/to/output/robot.sdf
```

### Method 2: Using xacro to Generate URDF First

If your robot model uses xacro (XML macros), you'll need to generate the URDF first:

```bash
# Generate URDF from xacro
ros2 run xacro xacro /path/to/your/robot.xacro > /tmp/robot.urdf

# Then convert to SDF
gz sdf -p /tmp/robot.urdf > /path/to/output/robot.sdf
```

### Example Conversion Process

Let's convert a simple humanoid robot model:

1. **Create a simple URDF file** (robot.urdf):
```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Joint connecting base to torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.35"/>
  </joint>

  <!-- Add Gazebo-specific elements -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <gazebo reference="torso">
    <material>Gazebo/Red</material>
  </gazebo>
</robot>
```

2. **Convert to SDF**:
```bash
gz sdf -p robot.urdf > robot.sdf
```

## Adding Gazebo-Specific Elements to URDF

Before conversion, you can add Gazebo-specific elements to your URDF to ensure proper simulation behavior:

### Physics Properties

```xml
<gazebo reference="link_name">
  <mu1>0.2</mu1>  <!-- Friction coefficient -->
  <mu2>0.2</mu2>  <!-- Friction coefficient in the second direction -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100000.0</kd>  <!-- Contact damping -->
  <material>Gazebo/Blue</material>
</gazebo>
```

### Inertial Properties

```xml
<gazebo reference="link_name">
  <!-- Override inertial properties if needed -->
  <self_collide>false</self_collide>  <!-- Prevent self-collision -->
  <gravity>true</gravity>  <!-- Enable gravity for this link -->
  <max_contacts>10</max_contacts>  <!-- Maximum contacts for this link -->
</gazebo>
```

### Sensor Integration

```xml
<gazebo reference="sensor_link">
  <sensor name="camera_sensor" type="camera">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30.0</update_rate>
    <camera name="head">
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
      <frame_name>camera_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Configuring Joint Constraints and Properties

### Joint Types in Gazebo

Gazebo supports various joint types that correspond to URDF joint types:
- **Fixed**: No movement between links
- **Revolute**: Rotational movement around a single axis
- **Prismatic**: Linear movement along a single axis
- **Continuous**: Rotational movement without limits
- **Floating**: 6-DOF movement
- **Planar**: Movement on a plane

### Joint Limitations and Properties

```xml
<joint name="joint_name" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>

<!-- Gazebo-specific joint properties -->
<gazebo reference="joint_name">
  <provide_feedback>true</provide_feedback>
  <implicit_spring_damper>true</implicit_spring_damper>
</gazebo>
```

### Advanced Joint Configuration

For humanoid robots, precise joint configuration is crucial:

```xml
<gazebo reference="hip_joint">
  <provide_feedback>true</provide_feedback>
  <implicit_spring_damper>true</implicit_spring_damper>
  <axis>
    <xyz>0 0 1</xyz>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
      <effort>200</effort>
      <velocity>2</velocity>
    </limit>
    <dynamics>
      <damping>1.0</damping>
      <friction>0.1</friction>
    </dynamics>
  </axis>
</gazebo>
```

## Collision Properties Configuration

### Collision Detection Parameters

Proper collision configuration is essential for realistic physics simulation:

```xml
<gazebo reference="link_name">
  <collision>
    <max_contacts>10</max_contacts>
    <surface>
      <contact>
        <ode>
          <kp>1e+6</kp>  <!-- Contact stiffness -->
          <kd>1e+3</kd>  <!-- Contact damping -->
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
      <friction>
        <ode>
          <mu>0.5</mu>
          <mu2>0.5</mu2>
          <fdir1>0 0 1</fdir1>
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
      </friction>
    </surface>
  </collision>
</gazebo>
```

### Self-Collision Avoidance

For complex humanoid robots with many links, you may want to disable self-collision between certain pairs:

```xml
<gazebo>
  <static>false</static>
  <self_collide>false</self_collide>
  <enable_wind>false</enable_wind>
  <kinematic>false</kinematic>
  <gravity>true</gravity>
</gazebo>
```

## Spawning Robots in Gazebo

### Method 1: Using the spawn_model Service

Once you have your SDF file, you can spawn the robot in Gazebo:

```bash
# Spawn a model from an SDF file
ros2 run gazebo_ros spawn_entity.py -file /path/to/robot.sdf -entity my_robot -x 0 -y 0 -z 1
```

### Method 2: Launch File Integration

Create a launch file that starts Gazebo and spawns your robot:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'worlds',
                'my_world.world'
            ])
        }.items()
    )

    # Spawn robot
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'my_robot',
            '-file', PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'models',
                'robot.sdf'
            ]),
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_entity
    ])
```

### Method 3: Direct SDF Integration in World Files

You can also include your robot directly in world files:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="robot_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include your robot model directly -->
    <include>
      <uri>model://my_robot</uri>
      <pose>0 0 0.5 0 0 0</pose>
    </include>

    <!-- Other world elements -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

## Testing Robot Integration

### Basic Functionality Test

1. **Launch Gazebo with your environment**
2. **Spawn your robot model**
3. **Verify the robot appears correctly in the simulation**
4. **Check that physics properties are working (gravity, collisions)**

### Joint Movement Test

If your robot has movable joints, test them:

```bash
# List available topics
ros2 topic list | grep joint

# Check joint states
ros2 topic echo /joint_states

# If you have joint controllers, publish commands
ros2 topic pub /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "..."
```

### Sensor Data Verification

If your robot has sensors, verify they're publishing data:

```bash
# Check for sensor topics
ros2 topic list | grep -E "(camera|imu|lidar|scan)"

# View sensor data
ros2 topic echo /my_robot/camera/image_raw
ros2 topic echo /my_robot/imu/data
```

## Troubleshooting Common Integration Issues

### Issue: Robot Falls Through Ground
- **Cause**: Missing collision geometries or incorrect inertial properties
- **Solution**: Verify all links have proper collision elements and mass properties

### Issue: Robot Explodes or Behaves Erratically
- **Cause**: Incorrect inertial properties or joint limits
- **Solution**: Check mass, inertia, and joint parameters; ensure values are physically realistic

### Issue: Model Doesn't Appear
- **Cause**: Incorrect file path or malformed SDF
- **Solution**: Verify file path and check Gazebo console for error messages

### Issue: Joint Limits Not Working
- **Cause**: Joint limits defined in URDF but not properly transferred to SDF
- **Solution**: Add Gazebo-specific joint limit configuration in URDF

## Tools Required for This Lesson

- **URDF/Xacro**: Robot description format from Module 1
- **SDF**: Simulation Description Format for Gazebo
- **Gazebo**: Physics simulation environment
- **ROS 2 (Humble Hawksbill)**: Robot operating system for communication
- **Text Editor**: For editing URDF and SDF files
- **Command Line Tools**: For conversion and spawning operations

## Best Practices for Robot Integration

### 1. Verify Inertial Properties
- Ensure all links have realistic mass and inertia values
- Use the `inertial_calculator.py` tool if needed to calculate proper inertial properties

### 2. Test Incrementally
- Start with a simple model and gradually add complexity
- Test each joint and sensor individually before full integration

### 3. Use Proper Scaling
- Ensure all dimensions are in meters for consistency
- Verify that link sizes are appropriate for humanoid scale

### 4. Configure Physics Appropriately
- Set realistic friction and damping values
- Configure contact properties for stable simulation

## Summary

In this lesson, you've learned to integrate humanoid robots into the Gazebo simulation environment. You've mastered the conversion from URDF to SDF format, configured joint constraints and collision properties, and learned how to spawn and test robots in simulation.

The integration process you've completed creates a bridge between the robot description you created in Module 1 and the simulation environment established in this module. This integration is crucial for testing your robots in a safe, repeatable environment before physical deployment.

## Next Steps

With your Gazebo simulation environment established and your robots integrated, you're now ready to advance to Module 2 Chapter 2, where you'll implement sensor simulation systems (LiDAR, Depth Camera, IMU) in the Gazebo environment you've created. The physics parameters and understanding you've gained will be essential as you expand your simulation to include sophisticated sensor modeling.