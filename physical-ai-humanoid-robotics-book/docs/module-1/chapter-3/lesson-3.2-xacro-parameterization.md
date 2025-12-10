---
title: Lesson 3.2 - Xacro Parameterization
---

# Lesson 3.2 – Xacro Parameterization

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand the benefits of using Xacro over pure URDF
- Convert URDF to parameterized Xacro files using macros
- Create reusable Xacro macros for robot components
- Define and manage parameters in Xacro files
- Organize complex robot descriptions using modular Xacro files

## Concept Overview and Scope

Xacro (XML Macros) is an XML macro language that extends URDF by adding features like parameters, macros, and mathematical expressions. Xacro allows us to create more complex and reusable robot descriptions by enabling parameterization, macro definitions, and modular design patterns.

In this lesson, we'll learn how to transform static URDF files into dynamic, parameterized Xacro files that can be easily modified and reused across different robot designs.

## Understanding Xacro Benefits

While URDF is powerful for describing robot structures, it has limitations:
- No parameterization: Values must be hardcoded
- No reusability: Similar components need to be redefined
- No calculations: Mathematical expressions not supported
- Verbose syntax: Complex robots result in large files

Xacro addresses these limitations by providing:
- Parameters: Define values once and reuse them
- Macros: Create reusable components
- Expressions: Perform calculations within the file
- Inclusion: Split complex descriptions across multiple files

## Basic Xacro Syntax

Xacro files use the `.xacro` extension and must include the Xacro namespace:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="robot_name">
  <!-- Xacro content -->
</robot>
```

### Defining Properties (Parameters)

Properties in Xacro work like variables:

```xml
<xacro:property name="wheel_radius" value="0.1"/>
<xacro:property name="wheel_width" value="0.05"/>
<xacro:property name="PI" value="3.14159"/>
```

### Using Properties

Once defined, properties can be used throughout the file:

```xml
<link name="wheel">
  <visual>
    <geometry>
      <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
    </geometry>
  </visual>
</link>
```

### Mathematical Expressions

Xacro supports mathematical expressions within `${}`:

```xml
<xacro:property name="wheel_circumference" value="${2 * PI * wheel_radius}"/>
<xacro:property name="half_wheel_width" value="${wheel_width / 2}"/>
```

## Creating Xacro Macros

Macros are reusable components that can accept parameters:

```xml
<xacro:macro name="simple_wheel" params="prefix parent_link x_pos y_pos z_pos radius width color:=blue">
  <link name="${prefix}_wheel">
    <visual>
      <geometry>
        <cylinder radius="${radius}" length="${width}"/>
      </geometry>
      <material name="${color}">
        <color rgba="0 0 1 1" if="${color == 'blue'}"/>
        <color rgba="1 0 0 1" if="${color == 'red'}"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="${radius}" length="${width}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="${prefix}_wheel_joint" type="continuous">
    <parent link="${parent_link}"/>
    <child link="${prefix}_wheel"/>
    <origin xyz="${x_pos} ${y_pos} ${z_pos}" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</xacro:macro>
```

### Using Macros

Once defined, macros can be instantiated multiple times:

```xml
<xacro:simple_wheel prefix="front_left" parent_link="base_link"
                   x_pos="0.2" y_pos="0.15" z_pos="0"
                   radius="0.05" width="0.02" color="blue"/>
<xacro:simple_wheel prefix="front_right" parent_link="base_link"
                   x_pos="0.2" y_pos="-0.15" z_pos="0"
                   radius="0.05" width="0.02" color="blue"/>
```

## Advanced Xacro Features

### Conditionals

Xacro supports conditional statements:

```xml
<xacro:if value="${has_lidar}">
  <link name="lidar_mount">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.02"/>
      </geometry>
    </visual>
  </link>
</xacro:if>
```

### Loops

Xacro can create repetitive structures:

```xml
<xacro:macro name="spine_links" params="count spacing">
  <xacro:property name="link_height" value="${count * spacing}"/>
  <xacro:for each="i" in="${range(1, count+1)}">
    <link name="spine_$(i)">
      <visual>
        <geometry>
          <box size="0.1 0.1 ${spacing}"/>
        </geometry>
      </visual>
    </link>
  </xacro:for>
</xacro:macro>
```

### File Inclusion

Complex robots can be split across multiple files:

```xml
<xacro:include filename="$(find my_robot_description)/urdf/arm.xacro"/>
<xacro:include filename="$(find my_robot_description)/urdf/sensors.xacro"/>
```

## Converting URDF to Xacro: Complete Example

Let's convert the simple humanoid robot from Lesson 3.1 to Xacro:

### Parameter Definitions

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_humanoid">

  <!-- Robot Parameters -->
  <xacro:property name="PI" value="3.14159"/>
  <xacro:property name="torso_height" value="0.6"/>
  <xacro:property name="torso_width" value="0.3"/>
  <xacro:property name="torso_depth" value="0.3"/>
  <xacro:property name="head_radius" value="0.1"/>
  <xacro:property name="arm_length" value="0.3"/>
  <xacro:property name="arm_radius" value="0.05"/>

  <!-- Material Definitions -->
  <material name="light_gray">
    <color rgba="0.7 0.7 0.7 1"/>
  </material>
  <material name="white">
    <color rgba="1 1 1 1"/>
  </material>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
</robot>
```

### Base Link Macro

```xml
  <!-- Base Link Macro -->
  <xacro:macro name="base_link_macro">
    <link name="base_link">
      <visual>
        <geometry>
          <box size="${torso_width} ${torso_depth} ${torso_height}"/>
        </geometry>
        <material name="light_gray"/>
      </visual>
      <collision>
        <geometry>
          <box size="${torso_width} ${torso_depth} ${torso_height}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="5.0"/>
        <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
      </inertial>
    </link>
  </xacro:macro>
```

### Head Macro

```xml
  <!-- Head Macro -->
  <xacro:macro name="head_macro">
    <link name="head">
      <visual>
        <geometry>
          <sphere radius="${head_radius}"/>
        </geometry>
        <material name="white"/>
      </visual>
      <collision>
        <geometry>
          <sphere radius="${head_radius}"/>
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
      <origin xyz="0.0 0.0 ${torso_height/2 + head_radius}"/>
    </joint>
  </xacro:macro>
```

### Arm Macro

```xml
  <!-- Arm Macro -->
  <xacro:macro name="arm_macro" params="side position">
    <link name="${side}_upper_arm">
      <visual>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
        <material name="blue"/>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${arm_length}" radius="${arm_radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="base_link"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0.0 ${position * torso_depth/2} ${torso_height/2 - arm_radius}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-PI/2}" upper="${PI/2}" effort="10" velocity="1"/>
    </joint>
  </xacro:macro>
```

### Complete Robot Assembly

```xml
  <!-- Instantiate the robot components -->
  <xacro:base_link_macro/>
  <xacro:head_macro/>
  <xacro:arm_macro side="left" position="1"/>
  <xacro:arm_macro side="right" position="-1"/>
</robot>
```

## Modular Design with Multiple Xacro Files

For complex robots, it's beneficial to split the description across multiple files:

### Main Robot File (`robot.xacro`)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot">

  <!-- Include component definitions -->
  <xacro:include filename="common_properties.xacro"/>
  <xacro:include filename="base.xacro"/>
  <xacro:include filename="arm.xacro"/>
  <xacro:include filename="leg.xacro"/>

  <!-- Instantiate components -->
  <xacro:robot_base/>
  <xacro:left_arm prefix="l"/>
  <xacro:right_arm prefix="r"/>
  <xacro:left_leg prefix="l"/>
  <xacro:right_leg prefix="r"/>
</robot>
```

### Common Properties File (`common_properties.xacro`)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="PI" value="3.14159"/>

  <!-- Material definitions -->
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</robot>
```

## Best Practices for Xacro Development

1. **Organize Parameters**: Group related parameters at the top of the file
2. **Use Descriptive Names**: Make parameter and macro names self-explanatory
3. **Modular Design**: Split complex robots across multiple files
4. **Default Values**: Provide sensible defaults for macro parameters
5. **Documentation**: Comment complex expressions and macros
6. **Validation**: Test Xacro files by converting to URDF

### Validating Xacro Files

You can convert Xacro to URDF to validate the output:

```bash
# Convert Xacro to URDF
xacro input_file.xacro > output_file.urdf

# Then validate the resulting URDF
check_urdf output_file.urdf
```

## Step-by-Step Exercise

Create a parameterized differential drive robot:

1. Create a `differential_drive.xacro` file
2. Define parameters for wheel radius, base dimensions, etc.
3. Create a macro for wheels that accepts position parameters
4. Create macros for the robot base and mounting points
5. Assemble the complete robot using your macros
6. Convert to URDF and validate

## Summary

In this lesson, you learned:
- The benefits of Xacro over pure URDF
- How to define and use properties (parameters) in Xacro
- How to create and use macros for reusable components
- Advanced features like conditionals, loops, and file inclusion
- Best practices for organizing complex Xacro files
- How to validate Xacro files

Xacro enables the creation of flexible, reusable robot descriptions that can be easily modified and adapted for different robot configurations. This is essential for creating scalable robot models in ROS2 systems.

## Next Steps

In the next lesson, we'll explore how to visualize and validate our URDF/Xacro models using ROS2 tools like RViz and Robot State Publisher, ensuring our robot descriptions are correct and ready for simulation.