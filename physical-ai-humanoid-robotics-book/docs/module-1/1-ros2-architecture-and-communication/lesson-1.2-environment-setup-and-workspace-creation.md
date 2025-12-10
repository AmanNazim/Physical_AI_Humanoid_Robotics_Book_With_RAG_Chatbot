---
title: Lesson 1.2 - Environment Setup and Workspace Creation
---

# Lesson 1.2 – Environment Setup and Workspace Creation

## Learning Objectives and Scope

**Learning Objective**: Install ROS2 Humble Hawksbill on Ubuntu 22.04 environment, create and configure a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, verify ROS2 installation with basic commands

**Lesson Scope**: This hands-on lesson guides you through setting up your ROS2 development environment. You'll install ROS2 Humble Hawksbill, create your first workspace, and configure the build system. The lesson emphasizes best practices for workspace organization and includes troubleshooting tips for common installation issues. You'll create your first package.xml and setup.py files, establishing a proper development workflow.

**Key Outcomes**: By the end of this lesson, you will have successfully installed ROS2 Humble Hawksbill on Ubuntu 22.04 environment, created and configured a ROS2 workspace with proper directory structure, set up the development environment with colcon build system, and verified ROS2 installation with basic commands.

**Tools**: ROS2 Humble Hawksbill, colcon build system, Ubuntu 22.04

**Key Activities**:
- Installing ROS2 on Ubuntu 22.04
- Creating workspace directory structure in ~/ros2_ws/src
- Setting up colcon build system
- Creating basic package configuration files

## Prerequisites and System Requirements

Before beginning this lesson, ensure your system meets the following requirements:

### Operating System
- **Ubuntu 22.04 LTS** (Jammy Jellyfish) - This is the recommended and most tested platform for ROS2 Humble Hawksbill
- Alternative: Ubuntu 20.04 LTS (Focal Fossa) with appropriate compatibility packages

### Hardware Requirements
- **Processor**: Multi-core processor (Intel i5 or equivalent recommended)
- **Memory**: 8GB RAM minimum (16GB recommended for simulation work)
- **Storage**: 20GB free disk space for ROS2 installation and workspace
- **Network**: Internet connection for package installation and updates

### Software Dependencies
- Python 3.8 or higher
- Git version control system
- Basic development libraries (build-essential, cmake, etc.)

## Installing ROS2 Humble Hawksbill

ROS2 Humble Hawksbill is the recommended distribution for this course, providing long-term support and compatibility with the latest features required for Physical AI applications.

### Step 1: Setting up Your Sources

First, add the ROS2 package repository to your system:

```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### Step 2: Adding the Repository

Add the ROS2 repository to your system's software sources:

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Step 3: Installing ROS2 Packages

Update your package list and install the ROS2 desktop package:

```bash
sudo apt update
sudo apt install ros-humble-desktop
```

This will install the full ROS2 desktop environment, including all necessary packages for development.

### Step 4: Installing Additional Dependencies

Install the Python package manager and other essential tools:

```bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-vcstool
sudo rosdep init
rosdep update
```

## Setting Up Your ROS2 Environment

### Step 5: Sourcing the Setup Script

To use ROS2, you need to source the setup script. You can do this temporarily for the current session:

```bash
source /opt/ros/humble/setup.bash
```

Or add it to your shell profile to make it permanent:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Verifying Installation

Verify that ROS2 is properly installed:

```bash
ros2 --version
```

You should see output similar to: `ros2 humble`

## Creating Your ROS2 Workspace

### Step 7: Workspace Directory Structure

A ROS2 workspace is a directory where you'll develop and build your ROS2 packages. The standard structure includes:

- `src/` - Source code for your packages
- `build/` - Build artifacts (created during compilation)
- `install/` - Installation directory (created after building)
- `log/` - Build logs and other diagnostic information

Create your workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### Step 8: Building Your Workspace

Even though your `src` directory is empty, you can build your workspace to ensure everything is set up correctly:

```bash
cd ~/ros2_ws
colcon build
```

After building, source your workspace:

```bash
source ~/ros2_ws/install/setup.bash
```

### Step 9: Adding Workspace to Your Profile

To automatically source your workspace when opening a new terminal:

```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Understanding the Colcon Build System

The `colcon` build system is the recommended build tool for ROS2. It's designed to build multiple packages in parallel efficiently.

### Key Colcon Commands:
- `colcon build` - Builds all packages in the workspace
- `colcon build --packages-select <package_name>` - Builds only a specific package
- `colcon build --symlink-install` - Creates symlinks instead of copying files (useful during development)
- `colcon test` - Runs tests for all packages
- `colcon test-result` - Shows test results

## Creating Basic Package Configuration Files

### Step 10: Creating a Basic Package

Let's create a simple package to verify everything works. First, navigate to your source directory:

```bash
cd ~/ros2_ws/src
```

Create a basic package using the ROS2 package creation tool:

```bash
ros2 pkg create --build-type ament_python my_first_package
```

This creates a basic Python-based ROS2 package with the necessary configuration files.

### Step 11: Package Structure and Configuration

The package structure includes:

- `package.xml` - Package manifest containing metadata and dependencies
- `setup.py` - Python package configuration
- `setup.cfg` - Installation configuration
- `my_first_package/` - Python package directory
- `my_first_package/__init__.py` - Python package initialization
- `my_first_package/__main__.py` - Entry point for the package

The `package.xml` file is crucial as it defines your package's dependencies, maintainers, license, and other metadata required by the ROS2 build system.

## Verification and Testing

### Step 12: Building Your First Package

Navigate to your workspace and build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_first_package
```

Source the workspace again to update the environment:

```bash
source install/setup.bash
```

### Step 13: Verifying Package Installation

Check if your package is recognized by ROS2:

```bash
ros2 pkg list | grep my_first_package
```

This should return `my_first_package` if everything is set up correctly.

### Step 14: Basic ROS2 Commands Verification

Test some basic ROS2 commands to ensure everything is working:

```bash
# List available ROS2 commands
ros2 --help

# List available topics (should be empty initially)
ros2 topic list

# List available nodes (should be empty initially)
ros2 node list
```

## Troubleshooting Common Installation Issues

### Issue 1: Permission Denied Errors
If you encounter permission errors, ensure you're not trying to install ROS2 in a restricted directory and that your user has appropriate permissions.

### Issue 2: Missing Dependencies
If packages fail to install due to missing dependencies, run:
```bash
sudo apt update && sudo apt upgrade
```

### Issue 3: Environment Not Sourcing Properly
If ROS2 commands are not recognized, ensure the setup script is properly sourced in your `.bashrc` file:
```bash
cat ~/.bashrc | grep ros
```


## Lesson Summary

In this lesson, you have learned:

- **ROS2 Installation**: How to properly install ROS2 Humble Hawksbill on Ubuntu 22.04
- **Environment Setup**: How to configure your shell environment to work with ROS2
- **Workspace Creation**: The standard directory structure for ROS2 development
- **Colcon Build System**: How to use the recommended build tool for ROS2 packages
- **Package Configuration**: The basic files needed to create and manage ROS2 packages
- **Verification**: How to test that your installation is working correctly

You now have a properly configured development environment ready for creating ROS2 nodes and implementing communication patterns in subsequent lessons.

## Tools / References
- ROS2 Humble Hawksbill
- colcon build system
- Ubuntu 22.04