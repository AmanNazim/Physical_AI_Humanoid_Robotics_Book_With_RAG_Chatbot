# Lesson 1.1 – Introduction to Gazebo and Physics Simulation

## Learning Objectives

By the end of this lesson, you will be able to:
- Install and configure Gazebo simulation environment on Ubuntu 22.04 LTS
- Understand Gazebo's integration with ROS 2 and its role in robotics development
- Navigate the Gazebo interface and understand basic simulation concepts
- Identify different physics engines and their application to humanoid robotics
- Launch and verify basic Gazebo simulations to confirm installation functionality

## Introduction to Gazebo

Gazebo is an open-source, physics-based simulation environment that has become the de facto standard for robotics simulation. It provides accurate and efficient simulation of robots in complex indoor and outdoor environments, making it an essential tool for humanoid robotics development.

### Why Gazebo for Humanoid Robotics?

Humanoid robots present unique challenges in simulation due to their complex multi-joint systems, balance requirements, and interaction with diverse environments. Gazebo excels at handling these challenges through:

- **Accurate Physics Modeling**: Realistic simulation of gravity, friction, and collisions
- **Complex Multi-body Systems**: Handling robots with many degrees of freedom
- **Sensor Simulation**: Integration of various sensors like LiDAR, cameras, and IMUs
- **Environment Flexibility**: Creation of diverse testing environments
- **ROS Integration**: Seamless integration with ROS 2 for control and communication

## Installing Gazebo

Gazebo comes in different versions, and for this course, we'll be using Gazebo Garden, which is the latest stable version compatible with ROS 2 Humble Hawksbill. The installation process involves setting up the appropriate package repositories and installing the necessary components.

### Prerequisites

Before installing Gazebo, ensure you have:
- Ubuntu 22.04 LTS installed
- ROS 2 Humble Hawksbill properly installed and sourced
- Internet connection for package downloads

### Installation Steps

1. **Update your system packages:**
   ```bash
   sudo apt update
   sudo apt upgrade
   ```

2. **Install the Gazebo package:**
   ```bash
   sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-plugins ros-humble-gazebo-dev
   ```

3. **Install the standalone Gazebo simulator:**
   ```bash
   sudo apt install gazebo
   ```

4. **Verify the installation by launching Gazebo:**
   ```bash
   gazebo
   ```

If the Gazebo GUI launches successfully, your installation is complete. You should see the Gazebo interface with a default empty world.

### Alternative Installation Method

If the above method doesn't work, you can install Gazebo Garden separately:

1. **Add the Gazebo package repository:**
   ```bash
   sudo apt install software-properties-common
   sudo add-apt-repository ppa:openrobotics/gazebo
   sudo apt update
   ```

2. **Install Gazebo Garden:**
   ```bash
   sudo apt install gazebo
   ```

## Understanding Gazebo Interface

The Gazebo interface is designed to provide an intuitive environment for creating and interacting with robot simulations. Let's explore its main components:

### Main Interface Components

1. **Menu Bar**: Contains file operations, simulation controls, and preferences
2. **Toolbar**: Quick access to common simulation functions
3. **Scene Graph**: Displays all objects in the current simulation
4. **3D Viewport**: The main visualization area where the simulation runs
5. **Layers Panel**: Controls visibility of different simulation elements
6. **Time Panel**: Shows simulation time and controls playback speed
7. **Status Bar**: Provides information about the current simulation state

### Navigation Controls

- **Orbit**: Right-click and drag to rotate the camera around the scene
- **Pan**: Middle-click and drag to move the camera horizontally and vertically
- **Zoom**: Scroll wheel to zoom in and out
- **Focus**: Double-click on any object to center the view on it

## Physics Engines in Gazebo

Gazebo supports multiple physics engines, each with different characteristics suitable for various types of simulations:

### ODE (Open Dynamics Engine)
- **Best for**: General-purpose simulation, good balance of speed and accuracy
- **Characteristics**: Fast, stable, good for most humanoid robotics applications
- **Use case**: Default choice for most humanoid robot simulations

### Bullet Physics
- **Best for**: More complex collision detection and dynamics
- **Characteristics**: More accurate collision detection, slightly slower than ODE
- **Use case**: Applications requiring precise collision handling

### DART (Dynamic Animation and Robotics Toolkit)
- **Best for**: Complex multi-body systems with articulated joints
- **Characteristics**: Advanced constraint solving, good for complex humanoid structures
- **Use case**: High-fidelity humanoid robot simulations

### Setting Physics Engine

The physics engine can be configured in the world file or through launch parameters. For humanoid robotics, ODE is typically the best choice due to its balance of performance and accuracy.

## ROS 2 Integration

Gazebo integrates seamlessly with ROS 2 through the `gazebo_ros` packages, which provide bridges between Gazebo's native interface and ROS 2 topics and services.

### Key ROS 2 Integration Components

1. **gazebo_ros_pkgs**: Core packages that enable ROS 2 integration
2. **gazebo_ros_spawn_model**: Service to spawn models in simulation
3. **gazebo_ros_node**: ROS 2 node that manages communication with Gazebo
4. **Model plugins**: Custom plugins that provide ROS 2 interfaces for robot models

### Communication Channels

- **Topics**: Sensor data publishing, joint state updates, command interfaces
- **Services**: Model spawning, parameter configuration, simulation control
- **Actions**: Complex behaviors that may take time to complete

## Basic Gazebo Simulation Verification

Let's verify your Gazebo installation by launching a basic simulation:

### Launching a Basic World

1. **Open terminal and source ROS 2:**
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Launch Gazebo with a basic world:**
   ```bash
   gazebo --verbose worlds/empty.world
   ```

3. **If you don't have the world file, launch with default:**
   ```bash
   gazebo
   ```

### Adding Objects to the Simulation

1. **Click on the "Insert" tab in the Gazebo interface**
2. **Select "Models" from the dropdown**
3. **Choose a model (e.g., "ground plane", "box", "sphere")**
4. **Click in the 3D viewport to place the model**

### Controlling the Simulation

- **Play/Pause**: Use the play button in the toolbar to start/stop simulation
- **Reset**: Reset simulation time to zero
- **Step**: Advance simulation by one time step
- **Speed Control**: Adjust real-time factor to control simulation speed

### Testing Physics

1. **Insert a sphere and box into the simulation**
2. **Start the simulation**
3. **Observe how the objects respond to gravity and collisions**
4. **This demonstrates that physics simulation is working correctly**

## Troubleshooting Common Installation Issues

### Issue: Gazebo doesn't launch
- **Solution**: Check if your graphics drivers are properly installed
- **Solution**: Try running `nvidia-smi` to verify GPU access

### Issue: ROS 2 packages not found
- **Solution**: Ensure you've sourced your ROS 2 installation: `source /opt/ros/humble/setup.bash`

### Issue: Segmentation fault on startup
- **Solution**: Try running with software rendering: `gazebo --verbose --render-engine=ogre`

## Tools Required for This Lesson

- **Gazebo**: Physics simulation environment
- **ROS 2 (Humble Hawksbill)**: Robot operating system for communication
- **Ubuntu 22.04 LTS**: Operating system environment
- **Terminal**: Command line interface for installation and launching

## Summary

In this lesson, you've successfully installed Gazebo and learned about its interface and physics engines. You've verified your installation by launching basic simulations and observed how physics work in the environment. This foundation is essential for the next lessons where you'll create custom environments and integrate your humanoid robots.

The integration between Gazebo and ROS 2 that you've established here forms the backbone of your simulation environment. Understanding the physics engines and their characteristics will help you make informed decisions about simulation accuracy and performance as you progress through the module.

## Next Steps

In the next lesson, you'll build upon this foundation by creating custom environments for humanoid robot simulation with proper lighting and terrain. You'll learn to build both static and dynamic environments that will serve as the testing grounds for your robots.