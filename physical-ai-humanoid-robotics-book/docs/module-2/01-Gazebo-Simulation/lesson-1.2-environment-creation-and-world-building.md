# Lesson 1.2 – Environment Creation and World Building

## Learning Objectives

By the end of this lesson, you will be able to:
- Create custom environments for humanoid robot simulation with proper lighting and terrain
- Build both static and dynamic environments with appropriate environmental parameters
- Configure environment parameters for realistic robot testing
- Create and structure environment files for robot testing scenarios
- Apply best practices for environment design in Gazebo

## Introduction to Environment Creation

Creating realistic and appropriate environments is crucial for effective humanoid robot simulation. The environment determines how your robot interacts with the world, tests its capabilities, and validates its behaviors. In this lesson, we'll explore how to build custom environments that accurately represent real-world scenarios for humanoid robotics.

### Importance of Realistic Environments

For humanoid robots, the environment is not just a backdrop—it's an integral part of the system. A well-designed environment allows you to:
- Test locomotion and navigation capabilities
- Validate balance and stability systems
- Assess interaction with objects and obstacles
- Evaluate sensor performance under various conditions
- Prepare for real-world deployment scenarios

## Understanding SDF (Simulation Description Format)

Before creating environments, it's essential to understand SDF, Gazebo's native environment description format. While URDF is used for robot models, SDF is used for world descriptions.

### SDF Structure

An SDF world file typically includes:
- **World Definition**: Overall world parameters and physics properties
- **Models**: Static and dynamic objects in the environment
- **Lights**: Lighting configuration for visual rendering
- **Physics**: Physics engine configuration and parameters
- **GUI**: Visualization and interface settings

### Basic SDF World Structure
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- World parameters go here -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Models go here -->
    <model name="ground_plane">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <!-- Link definition -->
      </link>
    </model>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

## Creating Static Environments

Static environments form the foundation of most robot testing scenarios. These include ground planes, walls, furniture, and other non-moving objects.

### Basic Static Environment: Indoor Room

Let's create a simple indoor environment suitable for humanoid robot testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_room">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>10 10</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.01 0.01 0.01 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="wall_front">
      <pose>0 -5 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_back">
      <pose>0 5 1 0 0 3.14159</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_left">
      <pose>-5 0 1 0 0 1.5708</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_right">
      <pose>5 0 1 0 0 -1.5708</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.1 2</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture: Table -->
    <model name="table">
      <pose>2 0 0.4 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.5 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.4 0.2 1</ambient>
            <diffuse>0.6 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

### Saving and Using the World File

1. **Save the above XML content to a file** named `indoor_room.world`
2. **Place it in your Gazebo worlds directory** (typically `~/.gazebo/models/` or your custom directory)
3. **Launch the world with the command:**
   ```bash
   gazebo ~/.gazebo/models/indoor_room.world
   ```

## Creating Dynamic Environments

Dynamic environments include moving objects, changing conditions, and interactive elements that provide more challenging testing scenarios for humanoid robots.

### Basic Dynamic Environment: Moving Obstacle

Let's create an environment with a moving obstacle that tests the robot's navigation capabilities:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="dynamic_obstacle_world">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
            </plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>20 20</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Moving obstacle -->
    <model name="moving_obstacle">
      <pose>0 0 0.5 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>0.5</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <!-- Model plugin to make the obstacle move -->
        <plugin name="model_push" filename="libgazebo_ros_pubslish_wheel_odometry.so">
          <alwaysOn>true</alwaysOn>
          <updateRate>30</updateRate>
          <bodyName>link</bodyName>
          <topicName>cmd_vel</topicName>
          <timeout>30</timeout>
        </plugin>
      </link>
    </model>

    <!-- Static obstacles -->
    <model name="static_obstacle_1">
      <pose>-3 2 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="static_obstacle_2">
      <pose>3 -2 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1</length>
            </cylinder>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <cylinder>
              <radius>0.5</radius>
              <length>1</length>
            </cylinder>
          </geometry>
          <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

## Lighting and Terrain Configuration

Proper lighting and terrain configuration are essential for creating realistic environments that accurately test robot sensors and perception systems.

### Lighting Types and Configuration

Gazebo supports several lighting types:
- **Directional**: Simulates sunlight with parallel rays
- **Point**: Omnidirectional light source from a point
- **Spot**: Conical light beam with adjustable properties

### Example: Complex Lighting Setup

```xml
<!-- Ambient light -->
<light name="ambient_light" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.3 0.3 0.3 1</diffuse>
  <specular>0.1 0.1 0.1 1</specular>
  <direction>-0.3 0.1 -0.9</direction>
</light>

<!-- Primary directional light (sun) -->
<light name="sun" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.8 0.8 0.7 1</diffuse>
  <specular>0.2 0.2 0.2 1</specular>
  <direction>-0.5 0.1 -0.9</direction>
</light>

<!-- Fill light -->
<light name="fill_light" type="directional">
  <pose>0 0 10 0 0 0</pose>
  <diffuse>0.2 0.2 0.3 1</diffuse>
  <specular>0.1 0.1 0.1 1</specular>
  <direction>0.5 -0.2 -0.8</direction>
</light>
```

### Terrain Configuration

For outdoor environments, terrain configuration is crucial:

```xml
<model name="uneven_terrain">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://heightmaps/uneven_terrain.png</uri>
          <size>20 20 2</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://heightmaps/uneven_terrain.png</uri>
          <size>20 20 2</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

## Environment Parameters for Realistic Testing

Configuring environment parameters correctly ensures that your simulations accurately reflect real-world conditions.

### Physics Parameters

- **Gravity**: Typically set to 9.8 m/s² for Earth-like conditions
- **Time Step**: Smaller steps (e.g., 0.001s) provide more accurate simulation but require more computation
- **Real-time Factor**: Ratio of simulation time to real time (1.0 = real-time)
- **Friction Coefficients**: Mu1 and Mu2 parameters for contact friction

### Example Physics Configuration

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Best Practices for Environment Design

### 1. Scale Appropriately
- Ensure environment objects are scaled correctly relative to your humanoid robot
- Use real-world dimensions when possible for accurate physics simulation

### 2. Optimize for Performance
- Balance visual fidelity with simulation performance
- Use simpler geometries for collision detection than for visual rendering
- Limit the number of dynamic objects to maintain real-time performance

### 3. Consider Robot Capabilities
- Design environments that test the specific capabilities of your humanoid robot
- Include obstacles and challenges appropriate to the robot's intended use case
- Consider the robot's sensor range and capabilities when placing objects

### 4. Plan for Testing Scenarios
- Create multiple environment variants for different test scenarios
- Include safety margins and recovery areas in your environments
- Document environment parameters for reproducible testing

## Tools Required for This Lesson

- **Gazebo**: Physics simulation environment
- **Text Editor**: For creating and editing world files
- **ROS 2 (Humble Hawksbill)**: For advanced environment control (if needed)
- **Model Creation Tools**: For complex environment elements (optional)

## Creating Environment Files for Robot Testing

### Organizing Your Environments

Create a structured directory for your environment files:

```
~/gazebo_worlds/
├── indoor/
│   ├── simple_room.world
│   ├── office.world
│   └── corridor.world
├── outdoor/
│   ├── flat_terrain.world
│   ├── uneven_terrain.world
│   └── obstacle_course.world
└── specialized/
    ├── navigation_test.world
    └── manipulation_test.world
```

### Testing Your Environments

1. **Load the environment in Gazebo** to verify it renders correctly
2. **Check physics properties** by testing object interactions
3. **Validate lighting** to ensure sensors will function properly
4. **Test with a simple robot model** to verify the environment is suitable for robot testing

## Summary

In this lesson, you've learned to create both static and dynamic environments for humanoid robot simulation. You've explored the SDF format for world description, configured lighting and terrain properties, and learned best practices for environment design. You've also created example world files that can be used for testing humanoid robots in various scenarios.

The environments you create will serve as the testing grounds for your robots, allowing you to validate their capabilities in controlled, repeatable conditions before moving to physical hardware. Proper environment design is crucial for accurate simulation results and effective robot development.

## Next Steps

In the next lesson, you'll build upon these environment creation skills by integrating your humanoid robots into the Gazebo simulation environment. You'll learn to import URDF models, convert them to SDF format, and configure joint constraints and collision properties for realistic physics simulation.