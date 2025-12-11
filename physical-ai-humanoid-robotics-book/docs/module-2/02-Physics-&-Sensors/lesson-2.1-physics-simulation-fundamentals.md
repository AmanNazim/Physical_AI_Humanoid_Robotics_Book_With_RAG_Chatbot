---
title: Lesson 2.1 – Physics Simulation Fundamentals
sidebar_position: 3
---

# Lesson 2.1 – Physics Simulation Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

- Configure physics parameters for realistic simulation including gravity, friction, collision detection, and material properties
- Understand physics engines and their application to humanoid robotics
- Test physics behavior with different parameter settings
- Validate physics simulation accuracy against real-world expectations
- Work with different physics engines (ODE, Bullet, DART) in Gazebo
- Create and modify physics configuration files with realistic parameters

## Introduction to Physics Simulation

Physics simulation is the cornerstone of any realistic robotic simulation environment. In the context of humanoid robotics, accurate physics simulation ensures that the robot behaves naturally when interacting with its environment. This includes realistic movement, balance, collision responses, and environmental interactions.

Gazebo provides three different physics engines that can be used for simulating physical interactions:

1. **ODE (Open Dynamics Engine)**: The default physics engine, widely used and stable
2. **Bullet**: Offers advanced collision detection and constraint solving
3. **DART (Dynamic Animation and Robotics Toolkit)**: Provides robust and accurate physics simulation

Each engine has its strengths and is suitable for different types of simulations. Understanding these differences will help you choose the right engine for your specific humanoid robotics applications.

## Physics Parameters Configuration

### Gravity Configuration

Gravity is a fundamental parameter that affects all objects in the simulation. By default, Gazebo uses Earth's gravity value of 9.8 m/s², but this can be adjusted for different scenarios (e.g., moon simulation with 1.62 m/s²).

To configure gravity, you'll need to modify the physics configuration file. Create or update the `physics.world` file in your simulation directory:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="default">
    <!-- Physics engine configuration -->
    <physics type="ode" enabled="true">
      <!-- Gravity vector: x, y, z components in m/s^2 -->
      <gravity>0 0 -9.8</gravity>

      <!-- ODE-specific parameters -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Rest of your world configuration -->
  </world>
</sdf>
```

### Friction Parameters

Friction determines how objects interact when they come into contact with surfaces. There are two main types of friction coefficients:

- **Static Friction**: Resistance to initial motion
- **Dynamic Friction**: Resistance during sliding motion

These parameters are typically defined in the material properties of your robot's links:

```xml
<!-- Example of friction configuration in a URDF/SDF model -->
<link name="foot_link">
  <collision name="foot_collision">
    <geometry>
      <box>
        <size>0.1 0.08 0.02</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.7</mu>       <!-- Static friction coefficient -->
          <mu2>0.7</mu2>     <!-- Dynamic friction coefficient -->
          <slip1>0.0</slip1> <!-- Slip coefficient 1 -->
          <slip2>0.0</slip2> <!-- Slip coefficient 2 -->
        </ode>
      </friction>
    </surface>
  </collision>
</link>
```

### Collision Detection Settings

Collision detection is critical for preventing objects from passing through each other. The parameters control the precision and performance of collision detection:

```xml
<collision name="collision_name">
  <geometry>
    <!-- Geometry definition -->
  </geometry>
  <surface>
    <contact>
      <ode>
        <soft_cfm>0.0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1e+13</kp>        <!-- Contact stiffness -->
        <kd>1.0</kd>          <!-- Damping coefficient -->
        <max_vel>100.0</max_vel>
        <min_depth>0.001</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

### Material Properties

Material properties define how surfaces interact with each other. These include bounciness (restitution coefficient) and other surface characteristics:

```xml
<surface>
  <bounce>
    <restitution_coefficient>0.1</restitution_coefficient>
    <threshold>100000.0</threshold>
  </bounce>
</surface>
```

## Setting Up Physics Configuration in Gazebo

### Step 1: Create a Custom Physics Configuration File

First, let's create a physics configuration file that will be used in your simulation:

```bash
# Create a directory for physics configurations
mkdir -p ~/humanoid_robot_ws/src/humanoid_robot_simulation/config/physics

# Create the physics configuration file
touch ~/humanoid_robot_ws/src/humanoid_robot_simulation/config/physics/humanoid_physics.sdf
```

### Step 2: Configure Physics Parameters

Add the following content to your `humanoid_physics.sdf` file:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_world">
    <physics type="ode" enabled="true" default="true">
      <!-- Standard Earth gravity -->
      <gravity>0 0 -9.8</gravity>

      <!-- ODE Solver Configuration -->
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>              <!-- Number of iterations -->
          <sor>1.3</sor>                  <!-- Successive over-relaxation parameter -->
        </solver>

        <!-- Constraints Configuration -->
        <constraints>
          <cfm>0.0</cfm>                  <!-- Constraint Force Mixing parameter -->
          <erp>0.2</erp>                  <!-- Error Reduction Parameter -->
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Optional: Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Optional: Include sky -->
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

### Step 3: Launch Gazebo with Custom Physics

To launch Gazebo with your custom physics configuration, create a launch file:

```xml
<!-- In your launch directory, create gazebo_physics.launch -->
<launch>
  <arg name="world_name" default="$(find humanoid_robot_simulation)/config/physics/humanoid_physics.sdf"/>

  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(arg world_name)"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>
</launch>
```

## Testing Physics Behavior with Different Parameters

### Experiment 1: Varying Gravity Values

Let's create a simple test to observe how different gravity values affect robot behavior:

```bash
# Create a simple test script
cat << 'EOF' > ~/humanoid_robot_ws/src/humanoid_robot_simulation/test/gravity_test.sh
#!/bin/bash

echo "Testing different gravity values..."

# Create a temporary world file with low gravity (moon-like)
cat << 'WORLD' > /tmp/moon_gravity.sdf
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="moon_world">
    <physics type="ode" enabled="true">
      <gravity>0 0 -1.62</gravity>  <!-- Moon gravity -->
      <ode>
        <solver><type>quick</type><iters>100</iters><sor>1.3</sor></solver>
        <constraints><cfm>0.0</cfm><erp>0.2</erp></constraints>
      </ode>
    </physics>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
  </world>
</sdf>
WORLD

# Create a temporary world file with high gravity
cat << 'WORLD' > /tmp/high_gravity.sdf
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="high_gravity_world">
    <physics type="ode" enabled="true">
      <gravity>0 0 -20.0</gravity>  <!-- Very high gravity -->
      <ode>
        <solver><type>quick</type><iters>100</iters><sor>1.3</sor></solver>
        <constraints><cfm>0.0</cfm><erp>0.2</erp></constraints>
      </ode>
    </physics>
    <include><uri>model://ground_plane</uri></include>
    <include><uri>model://sun</uri></include>
  </world>
</sdf>
WORLD

echo "Created temporary world files for testing:"
echo "  - Moon gravity: /tmp/moon_gravity.sdf"
echo "  - High gravity: /tmp/high_gravity.sdf"
EOF

chmod +x ~/humanoid_robot_ws/src/humanoid_robot_simulation/test/gravity_test.sh
```

### Experiment 2: Testing Friction Effects

To test friction effects, we'll create a simple slope model:

```xml
<!-- Create a friction_test.sdf file -->
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="slope_test">
    <pose>0 0 0 0 0 0</pose>

    <!-- Slope base -->
    <link name="slope_base">
      <pose>0 0 0 0 0 0</pose>
      <collision name="slope_collision">
        <geometry>
          <mesh><uri>model://slope.dae</uri></mesh>
        </geometry>
      </collision>
      <visual name="slope_visual">
        <geometry>
          <mesh><uri>model://slope.dae</uri></mesh>
        </geometry>
      </visual>
      <surface>
        <friction>
          <ode>
            <mu>0.1</mu>    <!-- Low friction -->
            <mu2>0.1</mu2>
          </ode>
        </friction>
      </surface>
    </link>

    <!-- Test object -->
    <link name="test_object">
      <pose>0 0 2 0 0 0</pose>
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.01</iyy><iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>
      <collision name="object_collision">
        <geometry>
          <sphere><radius>0.1</radius></sphere>
        </geometry>
      </collision>
      <visual name="object_visual">
        <geometry>
          <sphere><radius>0.1</radius></sphere>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
```

## Validating Physics Simulation Accuracy

### Method 1: Comparing with Real-World Physics

To validate your physics simulation, compare the behavior of objects in simulation with real-world expectations:

1. **Free Fall Test**: Drop an object and measure its acceleration
2. **Pendulum Test**: Simulate a pendulum and compare period with theoretical calculation
3. **Collision Test**: Test collisions between objects and verify conservation of momentum

### Method 2: Using Built-in Gazebo Tools

Gazebo provides several tools to help validate physics:

```bash
# Check physics statistics
gz stats

# Monitor physics topics
ros2 topic echo /gazebo/model_state

# Get detailed physics information
gz service -s /gazebo/worlds --req-type gz.msgs.StringMsg --rep-type gz.msgs.StringMsg --timeout 1000
```

### Method 3: Creating Validation Scripts

Create a Python script to validate physics behavior:

```python
#!/usr/bin/env python3
"""
Physics validation script for humanoid robot simulation
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelStates
import numpy as np
import math

class PhysicsValidator(Node):
    def __init__(self):
        super().__init__('physics_validator')

        # Subscribe to Gazebo model states
        self.model_sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_callback,
            10
        )

        # Timer for periodic validation
        self.timer = self.create_timer(0.1, self.validate_physics)

        self.initial_pose = None
        self.start_time = self.get_clock().now()
        self.gravity_measured = False

        self.get_logger().info('Physics Validator initialized')

    def model_callback(self, msg):
        """Callback to get model positions"""
        # Find your test object in the model states
        for i, name in enumerate(msg.name):
            if name == 'test_sphere':
                self.current_pose = msg.pose[i]
                break

    def validate_physics(self):
        """Validate physics simulation"""
        if self.current_pose is not None:
            # Calculate position and velocity
            pos_z = self.current_pose.position.z

            # Calculate expected position under gravity: z = z0 - 0.5*g*t^2
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time).nanoseconds / 1e9

            # Assuming initial drop from height z0 = 2.0m
            z0 = 2.0
            expected_pos = z0 - 0.5 * 9.8 * elapsed_time**2

            if expected_pos < 0:  # Hit the ground
                expected_pos = 0.0

            error = abs(pos_z - expected_pos)

            if error < 0.1:  # Within tolerance
                self.get_logger().info(f'Physics validation passed: error = {error:.3f}')
            else:
                self.get_logger().warn(f'Physics validation failed: error = {error:.3f}')

def main(args=None):
    rclpy.init(args=args)
    validator = PhysicsValidator()

    try:
        rclpy.spin(validator)
    except KeyboardInterrupt:
        pass
    finally:
        validator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Different Physics Engines Comparison

### ODE (Open Dynamics Engine)
- **Pros**: Stable, widely used, good for general-purpose simulation
- **Cons**: May struggle with complex contacts
- **Best for**: Standard humanoid robot simulation, general robotics research

### Bullet
- **Pros**: Advanced collision detection, good for complex geometries
- **Cons**: Can be less stable with certain configurations
- **Best for**: Complex contact scenarios, articulated robots with many joints

### DART
- **Pros**: Highly accurate, robust constraint solving
- **Cons**: Higher computational overhead
- **Best for**: Precise physics simulation, biomechanical modeling

## Practical Exercise: Configuring Physics for Your Humanoid Robot

Let's apply what we've learned by configuring physics for a simple humanoid robot model:

1. **Create a physics configuration file** for your robot:

```xml
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="humanoid_physics_world">
    <!-- Physics engine configuration -->
    <physics type="ode" enabled="true" default="true">
      <gravity>0 0 -9.8</gravity>

      <ode>
        <solver>
          <type>quick</type>
          <iters>200</iters>    <!-- Higher iterations for stability -->
          <sor>1.0</sor>
        </solver>

        <constraints>
          <cfm>1e-5</cfm>       <!-- Lower CFM for tighter constraints -->
          <erp>0.2</erp>
          <contact_max_correcting_vel>10.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- Include your humanoid robot -->
    <include>
      <uri>model://humanoid_robot</uri>
    </include>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
  </world>
</sdf>
```

2. **Configure individual robot links** with appropriate friction and contact properties:

```xml
<!-- Example for foot link with high friction -->
<link name="left_foot">
  <collision name="left_foot_collision">
    <geometry>
      <box>
        <size>0.15 0.08 0.02</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>0.8</mu>         <!-- High friction for stable stance -->
          <mu2>0.8</mu2>
        </ode>
      </friction>
      <contact>
        <ode>
          <kp>1e+6</kp>        <!-- Stiff contact for solid footing -->
          <kd>1e+3</kd>
          <max_vel>100.0</max_vel>
          <min_depth>0.001</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

## Summary

In this lesson, we explored the fundamentals of physics simulation for humanoid robotics in Gazebo. We covered:

- **Gravity configuration**: Setting appropriate gravitational forces for realistic movement
- **Friction parameters**: Configuring static and dynamic friction for natural surface interactions
- **Collision detection**: Fine-tuning collision parameters for accurate physical interactions
- **Material properties**: Defining surface characteristics that affect robot-object interactions
- **Physics engines**: Understanding the differences between ODE, Bullet, and DART
- **Validation techniques**: Methods to verify physics simulation accuracy

The physics configuration we've established forms the foundation for realistic robot behavior in simulation. Properly configured physics parameters are essential for training humanoid robots that will eventually operate in the real world.

## Next Steps

With physics fundamentals established, we're ready to move on to Lesson 2.2, where we'll explore LiDAR sensor simulation. The physics parameters we've configured here will provide the realistic environment needed for accurate sensor simulation.

Before proceeding to the next lesson, ensure your physics configuration is working correctly by:
1. Testing different parameter values and observing their effects
2. Validating simulation behavior against real-world expectations
3. Verifying that your humanoid robot model interacts appropriately with the environment