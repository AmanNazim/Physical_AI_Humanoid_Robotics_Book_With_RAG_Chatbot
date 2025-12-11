---
title: Lesson 3.1 - Basic URDF Robot Description
---

# Lesson 3.1 – Basic URDF Robot Description

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the fundamental structure of URDF files
- Define base links and fundamental robot structure in URDF
- Add joints and connected links to form kinematic chains
- Create complete URDF files for humanoid robots
- Validate URDF syntax with XML parsers

## Concept Overview and Scope

URDF (Unified Robot Description Format) is an XML-based format that describes robot models. It defines the physical structure of a robot, including links (rigid parts), joints (connections between links), and other properties such as visual and collision geometry. URDF is essential for simulation, visualization, motion planning, and control of robotic systems.

In this lesson, we'll focus on the basic components of URDF and learn how to create the skeleton of a humanoid robot with proper kinematic chains.

## Understanding URDF Structure

URDF is an XML format that describes a robot as a collection of links connected by joints. The structure follows a tree-like hierarchy with a single base link (usually called "base_link") and branches that represent the robot's limbs.

### Basic URDF Components

A URDF file consists of several key elements:

1. **Links**: Represent rigid parts of the robot (e.g., torso, arms, legs)
2. **Joints**: Define how links connect and move relative to each other
3. **Visual**: Defines how the link appears in visualization
4. **Collision**: Defines the collision properties of the link
5. **Inertial**: Defines the mass and inertia properties of the link

### Basic URDF File Structure

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Define links -->
  <link name="base_link">
    <!-- Link properties -->
  </link>

  <!-- Define joints -->
  <joint name="joint_name" type="joint_type">
    <parent link="parent_link_name"/>
    <child link="child_link_name"/>
    <!-- Joint properties -->
  </joint>
</robot>
```

## Creating the Base Link

The base link is the root of the robot's kinematic tree. All other links connect to it directly or indirectly through joints.

### Simple Base Link Example

```xml
<?xml version="1.0"?>
<robot name="simple_robot">
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
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Key Components of a Link:

1. **Visual**: Defines how the link appears in visualization tools
   - Geometry: Shape (box, cylinder, sphere, mesh)
   - Material: Color and appearance properties

2. **Collision**: Defines collision detection properties
   - Usually simpler geometry than visual for performance

3. **Inertial**: Physical properties for simulation
   - Mass: The mass of the link
   - Inertia: Moment of inertia values

## Adding Joints to Connect Links

Joints define the relationship between two links and specify how they can move relative to each other.

### Joint Types

ROS2 supports several joint types:

1. **Revolute**: Rotational joint with limited range
2. **Continuous**: Rotational joint without limits
3. **Prismatic**: Linear sliding joint with limits
4. **Fixed**: No movement (rigid connection)
5. **Floating**: 6 DOF (not commonly used)
6. **Planar**: Motion on a plane (not commonly used)

### Joint Definition Example

```xml
<joint name="joint_name" type="revolute">
  <parent link="base_link"/>
  <child link="upper_arm"/>
  <origin xyz="0.0 0.0 0.5" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>
```

### Joint Components:

1. **Parent/Child**: Links that the joint connects
2. **Origin**: Position and orientation of the joint relative to the parent
   - xyz: Position offset (x, y, z)
   - rpy: Rotation (roll, pitch, yaw) in radians
3. **Axis**: Direction of joint motion (for revolute/prismatic)
4. **Limit**: Joint limits (for revolute/prismatic joints)

## Building a Simple Humanoid Robot

Let's create a simplified humanoid robot with a torso, head, two arms, and two legs.

### Complete URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="fixed">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.45"/>
  </joint>

  <!-- Left Arm -->
  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.0 0.2 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="1"/>
  </joint>
</robot>
```

## Kinematic Chains and Robot Structure

A kinematic chain is a series of rigid bodies (links) connected by joints. In humanoid robots, we typically have multiple kinematic chains:

1. **Torso Chain**: Base to head
2. **Arm Chains**: Torso to hands
3. **Leg Chains**: Torso to feet

### Important Considerations for Kinematic Chains:

1. **Tree Structure**: URDF must form a tree (no loops)
2. **Fixed Joints**: Used for multiple visual/collision elements per link
3. **Joint Limits**: Critical for realistic simulation
4. **Inertial Properties**: Needed for physics simulation

## Best Practices for URDF Creation

1. **Start Simple**: Begin with basic shapes, add complexity gradually
2. **Use Consistent Units**: Stick to meters for length, kilograms for mass
3. **Validate Early**: Check URDF syntax frequently
4. **Plan Your Structure**: Sketch the robot before coding
5. **Use Descriptive Names**: Make link and joint names clear and consistent

## Validation Techniques

Before using your URDF file, validate it:

1. **XML Validation**: Ensure proper XML syntax
2. **URDF Validation**: Use ROS2 tools to check URDF structure
3. **Kinematic Validation**: Verify joints form proper chains

### Checking URDF Validity

You can use ROS2 tools to validate your URDF:

```bash
# Check if the URDF file is valid
check_urdf /path/to/your/robot.urdf
```

## Step-by-Step Exercise

Let's create a simple mobile robot with a base, caster wheel, and two drive wheels:

1. Create a new file called `mobile_robot.urdf`
2. Define a base link as a rectangular box
3. Add a caster wheel in the front (fixed joint)
4. Add two drive wheels on the sides (continuous joints)
5. Validate your URDF

## Summary

In this lesson, you learned:
- The fundamental structure of URDF files
- How to define links with visual, collision, and inertial properties
- How to connect links using joints with different types
- How to create a simple humanoid robot structure
- Best practices for URDF creation and validation

URDF forms the foundation for robot simulation, visualization, and control. Understanding these basics is crucial for creating robots that can be properly simulated and controlled in ROS2 systems.

## Next Steps

In the next lesson, we'll explore Xacro, which allows us to create parameterized and modular URDF files, making it easier to create complex robots and modify designs efficiently.