---
title: Lesson 3.3 - AI Decision Making and Action Planning
sidebar_position: 4
---

# Lesson 3.3: AI Decision Making and Action Planning

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement AI decision-making systems for robot behavior using the NVIDIA Isaac ecosystem
- Connect AI reasoning with action planning frameworks for coherent robot actions
- Create adaptive systems that respond to environmental conditions in real-time
- Design and implement decision trees and state machines for humanoid robot autonomy
- Integrate perception data with decision-making algorithms to enable intelligent responses
- Validate AI decision-making performance with environmental response scenarios

## Overview

In this lesson, we'll explore the critical component of cognitive architectures: AI decision-making and action planning. Building upon the perception processing pipelines established in Lesson 3.2, we'll implement sophisticated AI decision-making systems that allow humanoid robots to process sensory information and generate appropriate behavioral responses. We'll focus on connecting AI reasoning with action planning frameworks to create adaptive systems that respond intelligently to environmental conditions.

The AI decision-making system serves as the brain of the cognitive architecture, processing inputs from perception systems and generating appropriate action sequences. This system must handle uncertainty, adapt to changing environments, and make real-time decisions that ensure the robot behaves safely and effectively.

## Understanding AI Decision-Making Systems for Robotics

AI decision-making systems in robotics are responsible for selecting appropriate actions based on current perceptions, goals, and environmental conditions. Unlike traditional programming approaches where actions are predetermined, AI decision-making systems must handle uncertainty, learn from experience, and adapt to novel situations.

### Key Components of AI Decision-Making Systems

1. **State Representation**: Maintaining a representation of the current world state based on sensor inputs
2. **Goal Representation**: Defining and maintaining goals that guide decision-making
3. **Action Selection**: Choosing appropriate actions based on current state and goals
4. **Execution Monitoring**: Tracking action execution and adapting plans as needed
5. **Learning Mechanisms**: Improving decision-making over time based on experience

### Types of Decision-Making Approaches

There are several approaches to AI decision-making in robotics:

- **Rule-Based Systems**: Use predefined rules to determine actions based on conditions
- **State Machines**: Represent decision logic as transitions between discrete states
- **Decision Trees**: Use hierarchical decision structures to select actions
- **Planning Algorithms**: Generate action sequences to achieve specific goals
- **Reinforcement Learning**: Learn optimal decision policies through trial and error

The NVIDIA Isaac ecosystem provides specialized cognitive architecture tools that facilitate the implementation of these approaches. Isaac's behavior tree framework, planning algorithms, and AI reasoning components are specifically designed for robotic applications and provide optimized performance on NVIDIA hardware.

## Setting Up the AI Decision-Making Environment

Before implementing our decision-making system, we need to establish the environment and dependencies required for AI processing on NVIDIA platforms.

### Isaac Cognitive Architecture Tools

The NVIDIA Isaac ecosystem provides specialized cognitive architecture tools that are essential for implementing AI decision-making systems:

1. **Isaac Behavior Tree Framework**: A powerful system for creating complex robot behaviors using hierarchical tree structures
2. **Isaac Planning Components**: Optimized path planning and motion planning algorithms that run efficiently on NVIDIA hardware
3. **Isaac AI Reasoning Engine**: Provides advanced reasoning capabilities for autonomous decision-making
4. **Isaac Perception Integration**: Tools that connect perception data with decision-making algorithms
5. **Isaac Performance Monitoring**: Built-in tools for validating and monitoring cognitive system performance

These tools are specifically designed to work together seamlessly and provide optimized performance on NVIDIA GPUs.

```bash
# Install required packages for AI decision-making
sudo apt-get update
sudo apt-get install -y ros-humble-behavior-tree-cpp-v3
sudo apt-get install -y ros-humble-nav2-behaviors
sudo apt-get install -y ros-humble-dwb-core
sudo apt-get install -y ros-humble-isaac-ros-behavior-tree
```

### Configuring NVIDIA GPU for AI Processing

```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Set up environment variables for GPU acceleration
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=all
```

## Implementing Behavior Trees for Action Planning

Behavior trees are a popular choice for action planning in robotics because they provide a flexible, modular way to represent complex behaviors. They're particularly well-suited for humanoid robots that need to handle multiple concurrent tasks and respond to environmental changes.

### Basic Behavior Tree Structure

A behavior tree consists of nodes that can be either control nodes or leaf nodes:

- **Control Nodes**: Manage the flow of execution (sequences, selectors, parallels)
- **Leaf Nodes**: Execute specific actions or conditions (tasks, conditions)

```cpp
// behavior_tree_node.cpp - AI Decision Making Node
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/loggers/basic_logger.h>
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Bool.h>

using namespace BT;

class ApproachTarget : public SyncActionNode
{
public:
    ApproachTarget(const std::string& name, const NodeConfiguration& config)
        : SyncActionNode(name, config)
    {
        cmd_pub = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    }

    NodeStatus tick() override
    {
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.3;  // Move forward at 0.3 m/s
        cmd.angular.z = 0.0; // No rotation
        cmd_pub.publish(cmd);

        // Simulate approach completion after some condition
        // In practice, this would check distance to target
        ros::Duration(1.0).sleep();

        return NodeStatus::SUCCESS;
    }

    static PortsList providedPorts() { return {}; }

private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub;
};

class RotateToTarget : public SyncActionNode
{
public:
    RotateToTarget(const std::string& name, const NodeConfiguration& config)
        : SyncActionNode(name, config)
    {
        cmd_pub = nh_.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    }

    NodeStatus tick() override
    {
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.0;
        cmd.angular.z = 0.5;  // Rotate at 0.5 rad/s
        cmd_pub.publish(cmd);

        // Simulate rotation completion
        ros::Duration(1.0).sleep();

        return NodeStatus::SUCCESS;
    }

    static PortsList providedPorts() { return {}; }

private:
    ros::NodeHandle nh_;
    ros::Publisher cmd_pub;
};

class CheckObstacle : public ConditionNode
{
public:
    CheckObstacle(const std::string& name, const NodeConfiguration& config)
        : ConditionNode(name, config)
    {
        scan_sub = nh_.subscribe("/scan", 1, &CheckObstacle::laserCallback, this);
    }

    NodeStatus tick() override
    {
        if (obstacle_detected_)
            return NodeStatus::SUCCESS;
        else
            return NodeStatus::FAILURE;
    }

    static PortsList providedPorts() { return {}; }

private:
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        obstacle_detected_ = false;
        for (float range : msg->ranges)
        {
            if (range > msg->range_min && range < msg->range_max && range < 1.0)
            {
                obstacle_detected_ = true;
                break;
            }
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber scan_sub;
    bool obstacle_detected_ = false;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ai_decision_making_node");
    ros::NodeHandle nh;

    // Register custom nodes
    BehaviorTreeFactory factory;
    factory.registerNodeType<ApproachTarget>("ApproachTarget");
    factory.registerNodeType<RotateToTarget>("RotateToTarget");
    factory.registerNodeType<CheckObstacle>("CheckObstacle");

    // Define the behavior tree
    auto tree = factory.createTreeFromText(R"(
        <root BTCPP_format="4">
            <BehaviorTree>
                <Sequence>
                    <CheckObstacle/>
                    <Fallback>
                        <Sequence>
                            <RotateToTarget/>
                            <ApproachTarget/>
                        </Sequence>
                        <ApproachTarget/>
                    </Fallback>
                </Sequence>
            </BehaviorTree>
        </root>
    )");

    // Create a logger
    StdCoutLogger logger_cout(tree);

    // Run the tree continuously
    while(ros::ok())
    {
        tree.tickRoot();
        ros::Duration(0.1).sleep();
        ros::spinOnce();
    }

    return 0;
}
```

### CMakeLists.txt for Behavior Tree Node

```cmake
cmake_minimum_required(VERSION 3.8)
project(ai_decision_making)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(behaviortree_cpp_v3 REQUIRED)

# Add executable
add_executable(ai_decision_making_node src/behavior_tree_node.cpp)

# Link libraries
target_link_libraries(ai_decision_making_node
  ${BT_LIBRARIES}
)

# Link against dependencies
ament_target_dependencies(ai_decision_making_node
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  behaviortree_cpp_v3
)

install(TARGETS ai_decision_making_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
```

## Integrating Perception Data with Decision-Making

The effectiveness of AI decision-making systems depends heavily on the quality and interpretation of perception data. In this section, we'll explore how to integrate perception data from various sensors with our decision-making algorithms.

### Perception Integration Node

```cpp
// perception_integration_node.cpp
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int8.h>
#include <cmath>

class PerceptionIntegration
{
public:
    PerceptionIntegration()
    {
        ros::NodeHandle nh;

        // Publishers
        obstacle_distance_pub = nh.advertise<std_msgs::Float32>("/obstacle_distance", 1);
        danger_level_pub = nh.advertise<std_msgs::Int8>("/danger_level", 1);
        target_direction_pub = nh.advertise<geometry_msgs::PoseStamped>("/target_direction", 1);

        // Subscribers
        laser_sub = nh.subscribe("/scan", 1, &PerceptionIntegration::laserCallback, this);
        map_sub = nh.subscribe("/map", 1, &PerceptionIntegration::mapCallback, this);
        target_sub = nh.subscribe("/target_pose", 1, &PerceptionIntegration::targetCallback, this);

        tf_buffer = std::make_shared<tf2_ros::Buffer>();
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        rate = ros::Rate(10); // 10 Hz
    }

    void run()
    {
        while(ros::ok())
        {
            processPerceptionData();
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        // Process laser scan data to detect obstacles
        float min_range = std::numeric_limits<float>::max();

        for(size_t i = 0; i < msg->ranges.size(); ++i)
        {
            if(msg->ranges[i] > msg->range_min &&
               msg->ranges[i] < msg->range_max &&
               msg->ranges[i] < min_range)
            {
                min_range = msg->ranges[i];
            }
        }

        // Publish minimum obstacle distance
        std_msgs::Float32 obstacle_dist_msg;
        obstacle_dist_msg.data = min_range;
        obstacle_distance_pub.publish(obstacle_dist_msg);

        // Calculate danger level based on proximity
        int8_t danger_level = 0;
        if(min_range < 0.5) danger_level = 3; // High danger
        else if(min_range < 1.0) danger_level = 2; // Medium danger
        else if(min_range < 2.0) danger_level = 1; // Low danger

        std_msgs::Int8 danger_msg;
        danger_msg.data = danger_level;
        danger_level_pub.publish(danger_msg);
    }

    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
    {
        // Process occupancy grid data for path planning
        // This could include identifying free space, obstacles, and potential paths
        // For now, we'll just store the map data
        current_map = *msg;
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        // Store target pose for direction calculation
        target_pose = *msg;
    }

    void processPerceptionData()
    {
        // Calculate direction to target
        try
        {
            geometry_msgs::TransformStamped transform = tf_buffer->lookupTransform(
                "base_link", target_pose.header.frame_id,
                ros::Time(0), ros::Duration(1.0));

            // Transform target pose to robot frame
            geometry_msgs::PoseStamped transformed_target;
            tf2::doTransform(target_pose, transformed_target, transform);

            // Calculate direction vector
            double dx = transformed_target.pose.position.x;
            double dy = transformed_target.pose.position.y;
            double distance_to_target = sqrt(dx*dx + dy*dy);

            // Publish target direction
            geometry_msgs::PoseStamped direction_msg;
            direction_msg.header.stamp = ros::Time::now();
            direction_msg.header.frame_id = "base_link";
            direction_msg.pose.position.x = dx;
            direction_msg.pose.position.y = dy;
            direction_msg.pose.orientation.w = 1.0;
            target_direction_pub.publish(direction_msg);
        }
        catch (tf2::TransformException &ex)
        {
            ROS_WARN("Could not transform target pose: %s", ex.what());
        }
    }

    ros::Publisher obstacle_distance_pub;
    ros::Publisher danger_level_pub;
    ros::Publisher target_direction_pub;

    ros::Subscriber laser_sub;
    ros::Subscriber map_sub;
    ros::Subscriber target_sub;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;

    nav_msgs::OccupancyGrid current_map;
    geometry_msgs::PoseStamped target_pose;

    ros::Rate rate;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "perception_integration_node");
    PerceptionIntegration perception_integration;
    perception_integration.run();
    return 0;
}
```

## Creating Adaptive Systems for Environmental Response

Adaptive systems are crucial for humanoid robots operating in dynamic environments. These systems must continuously monitor environmental conditions and adjust behavior accordingly. In this section, we'll implement adaptive mechanisms that allow the robot to respond to changing conditions.

### Adaptive Behavior Controller

```cpp
// adaptive_behavior_controller.cpp
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <dynamic_reconfigure/server.h>
#include <std_srvs/SetBool.h>
#include <cmath>

class AdaptiveBehaviorController
{
public:
    AdaptiveBehaviorController()
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        // Publishers
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

        // Subscribers
        obstacle_distance_sub = nh.subscribe("/obstacle_distance", 1,
                                           &AdaptiveBehaviorController::obstacleDistanceCallback, this);
        danger_level_sub = nh.subscribe("/danger_level", 1,
                                      &AdaptiveBehaviorController::dangerLevelCallback, this);
        odometry_sub = nh.subscribe("/odom", 1,
                                  &AdaptiveBehaviorController::odometryCallback, this);

        // Parameters
        private_nh.param<double>("linear_velocity", linear_velocity_, 0.5);
        private_nh.param<double>("angular_velocity", angular_velocity_, 0.3);
        private_nh.param<double>("safe_distance", safe_distance_, 1.0);
        private_nh.param<double>("emergency_distance", emergency_distance_, 0.5);
        private_nh.param<double>("max_linear_velocity", max_linear_velocity_, 1.0);
        private_nh.param<double>("min_linear_velocity", min_linear_velocity_, 0.1);

        // Initialize variables
        current_danger_level_ = 0;
        obstacle_distance_ = std::numeric_limits<float>::max();
        current_linear_vel_ = linear_velocity_;
        current_angular_vel_ = angular_velocity_;

        rate = ros::Rate(20); // 20 Hz
    }

    void run()
    {
        while(ros::ok())
        {
            updateBehavior();
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    void obstacleDistanceCallback(const std_msgs::Float32::ConstPtr& msg)
    {
        obstacle_distance_ = msg->data;
    }

    void dangerLevelCallback(const std_msgs::Int8::ConstPtr& msg)
    {
        current_danger_level_ = msg->data;
    }

    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        // Store current position and velocity for adaptive control
        current_odom_ = *msg;
    }

    void updateBehavior()
    {
        geometry_msgs::Twist cmd;

        // Adjust velocities based on danger level and obstacle distance
        switch(current_danger_level_)
        {
            case 0: // Safe
                cmd.linear.x = current_linear_vel_;
                cmd.angular.z = current_angular_vel_;
                break;

            case 1: // Low danger
                cmd.linear.x = current_linear_vel_ * 0.7; // Reduce speed
                cmd.angular.z = current_angular_vel_ * 1.2; // Increase turning ability
                break;

            case 2: // Medium danger
                cmd.linear.x = current_linear_vel_ * 0.4; // Further reduce speed
                cmd.angular.z = current_angular_vel_ * 1.5; // Higher turning priority
                break;

            case 3: // High danger
                cmd.linear.x = 0.0; // Stop linear motion
                cmd.angular.z = current_angular_vel_ * 2.0; // Maximum turning
                break;

            default:
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.0;
                break;
        }

        // Additional safety check based on obstacle distance
        if(obstacle_distance_ < emergency_distance_)
        {
            cmd.linear.x = 0.0; // Emergency stop
            cmd.angular.z = current_angular_vel_; // Allow turning to escape
        }
        else if(obstacle_distance_ < safe_distance_)
        {
            // Gradually reduce speed as obstacle gets closer
            double reduction_factor = obstacle_distance_ / safe_distance_;
            cmd.linear.x *= reduction_factor;
        }

        // Apply velocity limits
        cmd.linear.x = std::max(min_linear_velocity_,
                               std::min(max_linear_velocity_, cmd.linear.x));
        cmd.angular.z = std::min(angular_velocity_ * 2.0,
                                std::max(-angular_velocity_ * 2.0, cmd.angular.z));

        // Publish command
        cmd_vel_pub.publish(cmd);
    }

    ros::Publisher cmd_vel_pub;
    ros::Subscriber obstacle_distance_sub;
    ros::Subscriber danger_level_sub;
    ros::Subscriber odometry_sub;

    double linear_velocity_;
    double angular_velocity_;
    double safe_distance_;
    double emergency_distance_;
    double max_linear_velocity_;
    double min_linear_velocity_;

    float current_danger_level_;
    float obstacle_distance_;
    double current_linear_vel_;
    double current_angular_vel_;

    nav_msgs::Odometry current_odom_;

    ros::Rate rate;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "adaptive_behavior_controller");
    AdaptiveBehaviorController controller;
    controller.run();
    return 0;
}
```

## Implementing Finite State Machines for Complex Behaviors

Finite State Machines (FSMs) are excellent for modeling complex robot behaviors with distinct operational modes. They provide a clear structure for managing different robot states and transitions between them based on environmental conditions.

### State Machine Implementation

```cpp
// state_machine_behavior.cpp
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <std_srvs/SetBool.h>
#include <vector>
#include <string>
#include <cmath>

enum RobotState {
    IDLE,
    EXPLORING,
    NAVIGATING_TO_TARGET,
    AVOIDING_OBSTACLE,
    REACHED_TARGET,
    EMERGENCY_STOP
};

class StateMachineBehavior
{
public:
    StateMachineBehavior()
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        // Publishers
        cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
        current_state_pub = nh.advertise<std_msgs::String>("/robot_state", 1);

        // Subscribers
        laser_sub = nh.subscribe("/scan", 1, &StateMachineBehavior::laserCallback, this);
        target_pose_sub = nh.subscribe("/target_pose", 1, &StateMachineBehavior::targetCallback, this);
        obstacle_distance_sub = nh.subscribe("/obstacle_distance", 1, &StateMachineBehavior::obstacleDistanceCallback, this);

        // Services
        start_service = nh.advertiseService("/start_robot_behavior", &StateMachineBehavior::startRobotBehavior, this);
        stop_service = nh.advertiseService("/stop_robot_behavior", &StateMachineBehavior::stopRobotBehavior, this);

        // Parameters
        private_nh.param<double>("linear_velocity", linear_velocity_, 0.5);
        private_nh.param<double>("angular_velocity", angular_velocity_, 0.3);
        private_nh.param<double>("safe_distance", safe_distance_, 1.0);
        private_nh.param<double>("target_tolerance", target_tolerance_, 0.5);

        // Initialize state
        current_state_ = IDLE;
        robot_active_ = false;
        obstacle_distance_ = std::numeric_limits<float>::max();

        rate = ros::Rate(20); // 20 Hz

        ROS_INFO("State Machine Behavior Node Initialized");
    }

    void run()
    {
        while(ros::ok())
        {
            if(robot_active_)
            {
                updateStateMachine();
            }
            else
            {
                // If robot is not active, send zero velocity
                geometry_msgs::Twist stop_cmd;
                cmd_vel_pub.publish(stop_cmd);
            }

            publishCurrentState();
            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        // Find minimum range in laser scan
        float min_range = std::numeric_limits<float>::max();
        for(float range : msg->ranges)
        {
            if(range > msg->range_min && range < msg->range_max && range < min_range)
            {
                min_range = range;
            }
        }
        obstacle_distance_ = min_range;
    }

    void targetCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        target_pose_ = *msg;
    }

    void obstacleDistanceCallback(const std_msgs::Float32::ConstPtr& msg)
    {
        obstacle_distance_ = msg->data;
    }

    bool startRobotBehavior(std_srvs::SetBool::Request &req,
                           std_srvs::SetBool::Response &res)
    {
        if(req.data)
        {
            robot_active_ = true;
            current_state_ = EXPLORING; // Start with exploration
            res.success = true;
            res.message = "Robot behavior started";
            ROS_INFO("Robot behavior started");
        }
        else
        {
            robot_active_ = false;
            current_state_ = IDLE;
            res.success = true;
            res.message = "Robot behavior stopped";
            ROS_INFO("Robot behavior stopped");
        }
        return true;
    }

    bool stopRobotBehavior(std_srvs::SetBool::Request &req,
                          std_srvs::SetBool::Response &res)
    {
        robot_active_ = false;
        current_state_ = IDLE;
        // Send stop command immediately
        geometry_msgs::Twist stop_cmd;
        cmd_vel_pub.publish(stop_cmd);
        res.success = true;
        res.message = "Robot stopped immediately";
        ROS_INFO("Robot stopped immediately");
        return true;
    }

    void updateStateMachine()
    {
        RobotState new_state = current_state_;

        // State transition logic
        switch(current_state_)
        {
            case IDLE:
                if(robot_active_)
                {
                    new_state = EXPLORING;
                }
                break;

            case EXPLORING:
                if(hasTarget())
                {
                    new_state = NAVIGATING_TO_TARGET;
                }
                else if(obstacle_distance_ < safe_distance_ * 0.5)
                {
                    new_state = AVOIDING_OBSTACLE;
                }
                break;

            case NAVIGATING_TO_TARGET:
                if(isNearTarget())
                {
                    new_state = REACHED_TARGET;
                }
                else if(obstacle_distance_ < safe_distance_ * 0.7)
                {
                    new_state = AVOIDING_OBSTACLE;
                }
                else if(!hasTarget()) // Target lost
                {
                    new_state = EXPLORING;
                }
                break;

            case AVOIDING_OBSTACLE:
                if(obstacle_distance_ > safe_distance_)
                {
                    if(hasTarget())
                    {
                        new_state = NAVIGATING_TO_TARGET;
                    }
                    else
                    {
                        new_state = EXPLORING;
                    }
                }
                break;

            case REACHED_TARGET:
                // Stay in this state until new target is set
                if(!isNearTarget())
                {
                    new_state = NAVIGATING_TO_TARGET;
                }
                break;

            case EMERGENCY_STOP:
                // Emergency stop state - only exit via service call
                if(obstacle_distance_ > safe_distance_ * 2.0)
                {
                    new_state = EXPLORING;
                }
                break;
        }

        // Execute behavior for current state
        executeCurrentState();

        // Update state if changed
        if(new_state != current_state_)
        {
            ROS_INFO("State transition: %s -> %s",
                     getStateName(current_state_).c_str(),
                     getStateName(new_state).c_str());
            current_state_ = new_state;
        }
    }

    void executeCurrentState()
    {
        geometry_msgs::Twist cmd;

        switch(current_state_)
        {
            case IDLE:
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.0;
                break;

            case EXPLORING:
                // Random exploration behavior
                cmd.linear.x = linear_velocity_ * 0.8; // Slightly slower exploration
                cmd.angular.z = 0.0; // Move forward initially
                break;

            case NAVIGATING_TO_TARGET:
                // Navigate towards target
                cmd.linear.x = linear_velocity_;
                cmd.angular.z = calculateAngularVelocityToTarget();
                break;

            case AVOIDING_OBSTACLE:
                // Simple obstacle avoidance - turn away from obstacle
                cmd.linear.x = 0.0;
                cmd.angular.z = angular_velocity_ * 1.5; // Turn to avoid
                break;

            case REACHED_TARGET:
                // Stop when target reached
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.0;
                break;

            case EMERGENCY_STOP:
                // Full stop in emergency
                cmd.linear.x = 0.0;
                cmd.angular.z = 0.0;
                break;
        }

        cmd_vel_pub.publish(cmd);
    }

    bool hasTarget()
    {
        // Check if a valid target is available
        return target_pose_.header.stamp.toSec() > 0;
    }

    bool isNearTarget()
    {
        // Simplified distance check - in practice, this would require TF transforms
        // to compare robot position with target position
        return obstacle_distance_ < target_tolerance_;
    }

    double calculateAngularVelocityToTarget()
    {
        // Simplified calculation - in practice, this would use TF to determine
        // the angle between robot orientation and target direction
        return angular_velocity_ * 0.5; // Reduced angular velocity for smooth navigation
    }

    std::string getStateName(RobotState state)
    {
        switch(state)
        {
            case IDLE: return "IDLE";
            case EXPLORING: return "EXPLORING";
            case NAVIGATING_TO_TARGET: return "NAVIGATING_TO_TARGET";
            case AVOIDING_OBSTACLE: return "AVOIDING_OBSTACLE";
            case REACHED_TARGET: return "REACHED_TARGET";
            case EMERGENCY_STOP: return "EMERGENCY_STOP";
            default: return "UNKNOWN";
        }
    }

    void publishCurrentState()
    {
        std_msgs::String state_msg;
        state_msg.data = getStateName(current_state_);
        current_state_pub.publish(state_msg);
    }

    ros::Publisher cmd_vel_pub;
    ros::Publisher current_state_pub;
    ros::Subscriber laser_sub;
    ros::Subscriber target_pose_sub;
    ros::Subscriber obstacle_distance_sub;
    ros::ServiceServer start_service;
    ros::ServiceServer stop_service;

    double linear_velocity_;
    double angular_velocity_;
    double safe_distance_;
    double target_tolerance_;

    RobotState current_state_;
    bool robot_active_;
    float obstacle_distance_;
    geometry_msgs::PoseStamped target_pose_;

    ros::Rate rate;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "state_machine_behavior");
    StateMachineBehavior sm;
    sm.run();
    return 0;
}
```

## Validating AI Decision-Making Performance

Performance validation is crucial for AI decision-making systems. We need to ensure that our systems respond appropriately to various environmental conditions and make decisions that lead to successful robot behavior.

### Performance Monitoring Node

```cpp
// performance_monitor_node.cpp
#include <ros/ros.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int8.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <fstream>
#include <iomanip>

class PerformanceMonitor
{
public:
    PerformanceMonitor()
    {
        ros::NodeHandle nh;
        ros::NodeHandle private_nh("~");

        // Publishers
        performance_metrics_pub = nh.advertise<std_msgs::Float32MultiArray>("/performance_metrics", 1);
        visualization_marker_pub = nh.advertise<visualization_msgs::Marker>("/performance_visualization", 1);

        // Subscribers
        state_sub = nh.subscribe("/robot_state", 1, &PerformanceMonitor::stateCallback, this);
        cmd_vel_sub = nh.subscribe("/cmd_vel", 1, &PerformanceMonitor::cmdVelCallback, this);
        laser_sub = nh.subscribe("/scan", 1, &PerformanceMonitor::laserCallback, this);
        odom_sub = nh.subscribe("/odom", 1, &PerformanceMonitor::odometryCallback, this);

        // Parameters
        private_nh.param<std::string>("log_file", log_file_, "/tmp/robot_performance_log.csv");
        private_nh.param<double>("update_rate", update_rate_, 1.0);
        private_nh.param<double>("safe_distance", safe_distance_, 1.0);

        // Initialize variables
        current_state_ = "UNKNOWN";
        last_update_time_ = ros::Time::now();
        total_distance_traveled_ = 0.0;
        total_time_ = 0.0;
        obstacle_encounters_ = 0;
        emergency_stops_ = 0;
        last_position_ = {0, 0};

        // Open log file
        log_file_stream_.open(log_file_, std::ios::out | std::ios::app);
        if(log_file_stream_.is_open())
        {
            log_file_stream_ << "timestamp,state,distance_traveled,avg_velocity,obstacle_encounters,emergency_stops\n";
        }

        rate = ros::Rate(update_rate_);
    }

    ~PerformanceMonitor()
    {
        if(log_file_stream_.is_open())
        {
            log_file_stream_.close();
        }
    }

    void run()
    {
        while(ros::ok())
        {
            if((ros::Time::now() - last_update_time_).toSec() >= (1.0 / update_rate_))
            {
                publishPerformanceMetrics();
                visualizePerformance();
                last_update_time_ = ros::Time::now();
            }

            ros::spinOnce();
            rate.sleep();
        }
    }

private:
    void stateCallback(const std_msgs::String::ConstPtr& msg)
    {
        current_state_ = msg->data;
        if(current_state_ == "EMERGENCY_STOP")
        {
            emergency_stops_++;
        }
    }

    void cmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg)
    {
        current_linear_velocity_ = msg->linear.x;
        current_angular_velocity_ = msg->angular.z;
    }

    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
    {
        // Count obstacle encounters (any range within safe distance)
        bool near_obstacle = false;
        for(float range : msg->ranges)
        {
            if(range > msg->range_min && range < msg->range_max && range < safe_distance_)
            {
                near_obstacle = true;
                break;
            }
        }

        if(near_obstacle && !was_near_obstacle_)
        {
            obstacle_encounters_++;
        }
        was_near_obstacle_ = near_obstacle;
    }

    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
    {
        // Calculate distance traveled
        double current_x = msg->pose.pose.position.x;
        double current_y = msg->pose.pose.position.y;

        if(first_odom_received_)
        {
            double dx = current_x - last_position_.first;
            double dy = current_y - last_position_.second;
            double distance_increment = sqrt(dx*dx + dy*dy);
            total_distance_traveled_ += distance_increment;
        }
        else
        {
            first_odom_received_ = true;
        }

        last_position_ = {current_x, current_y};
        total_time_ += 0.05; // Assuming 20Hz odometry update
    }

    void publishPerformanceMetrics()
    {
        std_msgs::Float32MultiArray metrics;

        // Performance metrics: [distance_traveled, avg_velocity, obstacle_encounters, emergency_stops, time_running]
        metrics.data.push_back(total_distance_traveled_);
        double avg_velocity = total_time_ > 0 ? total_distance_traveled_ / total_time_ : 0.0;
        metrics.data.push_back(avg_velocity);
        metrics.data.push_back(obstacle_encounters_);
        metrics.data.push_back(emergency_stops_);
        metrics.data.push_back(total_time_);

        performance_metrics_pub.publish(metrics);

        // Log to file
        if(log_file_stream_.is_open())
        {
            ros::Time current_time = ros::Time::now();
            log_file_stream_ << std::fixed << std::setprecision(3)
                            << current_time.toSec() << ","
                            << current_state_ << ","
                            << total_distance_traveled_ << ","
                            << avg_velocity << ","
                            << obstacle_encounters_ << ","
                            << emergency_stops_ << "\n";
            log_file_stream_.flush();
        }
    }

    void visualizePerformance()
    {
        visualization_msgs::Marker marker;

        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.ns = "performance_monitor";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        marker.action = visualization_msgs::Marker::ADD;

        // Position the text marker above the robot
        marker.pose.position.x = last_position_.first;
        marker.pose.position.y = last_position_.second;
        marker.pose.position.z = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.z = 0.2; // Text height
        marker.color.a = 1.0; // Alpha
        marker.color.r = 1.0; // Red
        marker.color.g = 1.0; // Green
        marker.color.b = 1.0; // Blue

        std::ostringstream oss;
        oss << "State: " << current_state_
            << "\nDist: " << std::fixed << std::setprecision(2) << total_distance_traveled_ << "m"
            << "\nAvg Vel: " << std::fixed << std::setprecision(2) << (total_time_ > 0 ? total_distance_traveled_/total_time_ : 0.0) << "m/s"
            << "\nObstacles: " << obstacle_encounters_
            << "\nEmergencies: " << emergency_stops_;

        marker.text = oss.str();

        visualization_marker_pub.publish(marker);
    }

    ros::Publisher performance_metrics_pub;
    ros::Publisher visualization_marker_pub;
    ros::Subscriber state_sub;
    ros::Subscriber cmd_vel_sub;
    ros::Subscriber laser_sub;
    ros::Subscriber odom_sub;

    std::string current_state_;
    double current_linear_velocity_ = 0.0;
    double current_angular_velocity_ = 0.0;
    bool was_near_obstacle_ = false;

    double total_distance_traveled_;
    double total_time_;
    int obstacle_encounters_;
    int emergency_stops_;
    std::pair<double, double> last_position_;
    bool first_odom_received_ = false;

    std::string log_file_;
    std::ofstream log_file_stream_;
    double update_rate_;
    double safe_distance_;

    ros::Time last_update_time_;
    ros::Rate rate;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "performance_monitor");
    PerformanceMonitor pm;
    pm.run();
    return 0;
}
```

## Testing and Validation

Now that we've implemented our AI decision-making and action planning systems, let's create a launch file to bring everything together and test the integrated system.

### Launch File

```xml
<!-- ai_decision_making.launch -->
<launch>
  <!-- Launch the perception integration node -->
  <node name="perception_integration_node" pkg="ai_decision_making" type="perception_integration_node" output="screen">
    <param name="linear_velocity" value="0.5"/>
    <param name="angular_velocity" value="0.3"/>
    <param name="safe_distance" value="1.0"/>
  </node>

  <!-- Launch the adaptive behavior controller -->
  <node name="adaptive_behavior_controller" pkg="ai_decision_making" type="adaptive_behavior_controller" output="screen">
    <param name="linear_velocity" value="0.5"/>
    <param name="angular_velocity" value="0.3"/>
    <param name="safe_distance" value="1.0"/>
    <param name="emergency_distance" value="0.5"/>
    <param name="max_linear_velocity" value="1.0"/>
    <param name="min_linear_velocity" value="0.1"/>
  </node>

  <!-- Launch the state machine behavior -->
  <node name="state_machine_behavior" pkg="ai_decision_making" type="state_machine_behavior" output="screen">
    <param name="linear_velocity" value="0.5"/>
    <param name="angular_velocity" value="0.3"/>
    <param name="safe_distance" value="1.0"/>
    <param name="target_tolerance" value="0.5"/>
  </node>

  <!-- Launch the performance monitor -->
  <node name="performance_monitor" pkg="ai_decision_making" type="performance_monitor" output="screen">
    <param name="log_file" value="/tmp/robot_performance_log.csv"/>
    <param name="update_rate" value="1.0"/>
    <param name="safe_distance" value="1.0"/>
  </node>

  <!-- Launch RViz for visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find ai_decision_making)/config/performance_monitor.rviz" />

</launch>
```

## Practical Example: Autonomous Navigation with Adaptive Decision-Making

Let's put everything together with a practical example that demonstrates how the AI decision-making system responds to various environmental conditions.

### Example Scenario: Warehouse Navigation

Consider a humanoid robot tasked with navigating through a warehouse environment to reach a target location. The environment contains static obstacles (shelves, walls) and dynamic obstacles (moving forklifts, people). Our AI decision-making system must:

1. Plan a path to the target while avoiding static obstacles
2. Detect and respond to dynamic obstacles in real-time
3. Adapt its behavior based on the current situation
4. Switch between different operational modes as needed

```yaml
# warehouse_scenario.yaml - Configuration for warehouse navigation scenario
ai_decision_making:
  linear_velocity: 0.6
  angular_velocity: 0.4
  safe_distance: 1.2
  emergency_distance: 0.6
  target_tolerance: 0.4
  max_linear_velocity: 1.0
  min_linear_velocity: 0.1

  # Behavior tree configuration
  behavior_tree:
    root_sequence:
      - check_obstacle_proximity
      - select_navigation_strategy:
          - sequence_approach_target
          - fallback_avoidance

  # State machine configuration
  state_machine:
    idle_timeout: 30.0
    exploration_duration: 60.0
    target_search_timeout: 120.0

  # Performance monitoring
  performance_thresholds:
    min_avg_velocity: 0.2
    max_emergency_stops_per_minute: 2
    min_success_rate: 0.8
```

## Integration with Isaac Cognitive Architecture Tools

The NVIDIA Isaac ecosystem provides powerful tools for cognitive architecture development. Let's see how we can integrate our decision-making system with Isaac's cognitive components:

### Isaac Cognitive Architecture Integration

```python
#!/usr/bin/env python3
# isaac_cognitive_integration.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Float32, Int8
from builtin_interfaces.msg import Duration
import numpy as np
import math

class IsaacCognitiveIntegrator(Node):
    def __init__(self):
        super().__init__('isaac_cognitive_integrator')

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_state_pub = self.create_publisher(String, '/robot_state', 10)
        self.danger_level_pub = self.create_publisher(Int8, '/danger_level', 10)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.target_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_callback,
            10
        )

        # Parameters
        self.linear_velocity = self.declare_parameter('linear_velocity', 0.5).value
        self.angular_velocity = self.declare_parameter('angular_velocity', 0.3).value
        self.safe_distance = self.declare_parameter('safe_distance', 1.0).value

        # State variables
        self.current_state = 'IDLE'
        self.obstacle_distances = []
        self.target_pose = None

        # Timer for main loop
        self.timer = self.create_timer(0.05, self.main_loop)  # 20 Hz

        self.get_logger().info('Isaac Cognitive Integrator initialized')

    def laser_callback(self, msg):
        """Process laser scan data"""
        ranges = [r for r in msg.ranges if not math.isnan(r)]
        if ranges:
            self.obstacle_distances = ranges

    def target_callback(self, msg):
        """Receive target pose"""
        self.target_pose = msg

    def calculate_danger_level(self):
        """Calculate danger level based on obstacle distances"""
        if not self.obstacle_distances:
            return 0

        min_distance = min(self.obstacle_distances)

        if min_distance < 0.5:
            return 3  # High danger
        elif min_distance < 1.0:
            return 2  # Medium danger
        elif min_distance < 2.0:
            return 1  # Low danger
        else:
            return 0  # Safe

    def make_decision(self):
        """Core AI decision-making algorithm"""
        danger_level = self.calculate_danger_level()

        # Publish danger level
        danger_msg = Int8()
        danger_msg.data = danger_level
        self.danger_level_pub.publish(danger_msg)

        # State transition logic
        if danger_level >= 3:
            new_state = 'EMERGENCY_STOP'
        elif danger_level >= 2:
            new_state = 'AVOIDING_OBSTACLE'
        elif self.target_pose is not None:
            new_state = 'NAVIGATING_TO_TARGET'
        else:
            new_state = 'EXPLORING'

        if new_state != self.current_state:
            self.get_logger().info(f'State transition: {self.current_state} -> {new_state}')
            self.current_state = new_state

        # Publish current state
        state_msg = String()
        state_msg.data = self.current_state
        self.robot_state_pub.publish(state_msg)

        return new_state

    def execute_action(self, state):
        """Execute action based on current state"""
        cmd = Twist()

        if state == 'IDLE':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        elif state == 'EXPLORING':
            cmd.linear.x = self.linear_velocity * 0.7
            cmd.angular.z = 0.0
        elif state == 'NAVIGATING_TO_TARGET':
            cmd.linear.x = self.linear_velocity
            cmd.angular.z = self.calculate_angular_to_target()
        elif state == 'AVOIDING_OBSTACLE':
            cmd.linear.x = 0.0
            cmd.angular.z = self.angular_velocity * 1.5
        elif state == 'EMERGENCY_STOP':
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        # Adjust for safety
        if self.obstacle_distances:
            min_dist = min(self.obstacle_distances)
            if min_dist < self.safe_distance:
                cmd.linear.x *= (min_dist / self.safe_distance)

        self.cmd_vel_pub.publish(cmd)

    def calculate_angular_to_target(self):
        """Calculate angular velocity to face target"""
        if self.target_pose is None:
            return 0.0

        # Simplified calculation - in practice, this would use TF transforms
        return self.angular_velocity * 0.5

    def main_loop(self):
        """Main decision-making loop"""
        state = self.make_decision()
        self.execute_action(state)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacCognitiveIntegrator()

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

## Summary

In this lesson, we've implemented comprehensive AI decision-making and action planning systems for humanoid robots using the NVIDIA Isaac ecosystem. We covered:

1. **Behavior Trees**: Created flexible, modular action planning systems that can handle complex robotic behaviors
2. **Perception Integration**: Developed systems that integrate sensor data with decision-making algorithms
3. **Adaptive Systems**: Implemented mechanisms that respond to changing environmental conditions
4. **Finite State Machines**: Designed state-based behavior controllers for distinct operational modes
5. **Performance Monitoring**: Created validation tools to ensure system reliability and effectiveness
6. **Isaac Integration**: Connected our decision-making systems with NVIDIA's cognitive architecture tools

These systems form the cognitive core of our humanoid robot, enabling it to perceive its environment, make intelligent decisions, and execute appropriate actions. The combination of behavior trees, state machines, and adaptive systems creates a robust framework that can handle the complexity and uncertainty inherent in real-world robotic applications.

The AI decision-making system we've developed connects the perception processing pipelines from Lesson 3.2 with the cognitive architecture framework from Lesson 3.1, creating a complete cognitive architecture that enables intelligent robot behavior. This system will serve as the foundation for the AI system integration in Chapter 4, where we'll connect these decision-making capabilities with higher-level vision-language-action systems.

## Next Steps

In the next chapter, we'll integrate all the cognitive architecture components we've developed with advanced AI systems, including vision-language-action models that will enable our humanoid robots to engage in complex human-robot interactions and perform sophisticated tasks requiring multimodal perception and reasoning.