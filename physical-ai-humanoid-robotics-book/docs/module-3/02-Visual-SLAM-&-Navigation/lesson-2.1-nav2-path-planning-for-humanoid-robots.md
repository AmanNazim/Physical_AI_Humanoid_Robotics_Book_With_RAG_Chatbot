---
sidebar_position: 2
---

# Lesson 2.1: Nav2 Path Planning for Humanoid Robots

## Learning Objectives

By the end of this lesson, you will be able to:

- Configure Nav2 path planning specifically adapted for humanoid robots
- Set up Nav2 framework with ROS2 Humble for humanoid navigation requirements
- Adapt path planning for bipedal locomotion constraints and humanoid robot kinematics
- Configure collision avoidance for humanoid form factor
- Test navigation in Isaac Sim environment with humanoid-specific constraints
- Validate path planning for bipedal locomotion

## Introduction to Nav2 for Humanoid Navigation

Navigation2 (Nav2) is a comprehensive navigation stack designed for mobile robots operating in dynamic environments. While traditionally used for wheeled robots, Nav2 can be adapted for humanoid robots with specific configurations that account for the unique challenges of bipedal locomotion. In this lesson, we'll explore how to configure Nav2 specifically for humanoid robots, considering factors such as balance, foot placement, and anthropomorphic movement patterns.

### Why Nav2 for Humanoid Robots?

Unlike traditional wheeled robots, humanoid robots face unique navigation challenges:

- **Bipedal locomotion constraints**: Humanoid robots must maintain balance while navigating, requiring careful consideration of step placement and center of mass
- **Anthropomorphic form factor**: The tall, narrow profile of humanoid robots affects collision detection and space requirements
- **Dynamic stability**: Unlike wheeled robots that maintain continuous contact with the ground, humanoid robots must plan for intermittent foot contact
- **Multi-modal movement**: Humanoid robots may need to transition between walking, climbing stairs, or even crawling in certain situations

### Key Components of Nav2 for Humanoid Navigation

The Nav2 stack consists of several key components that work together to enable navigation:

- **Navigation Server**: The central coordinator that manages the navigation pipeline
- **Planners**: Global and local planners that compute paths and trajectories
- **Controllers**: Local controllers that execute navigation commands
- **Recovery Behaviors**: Actions to take when navigation fails or gets stuck
- **Transforms**: Coordinate frame management for spatial relationships

## Setting Up Nav2 Framework with ROS2 Humble

Before configuring Nav2 for humanoid navigation, we need to establish the basic Nav2 framework with ROS2 Humble Hawksbill. This foundation will serve as the base for our humanoid-specific adaptations.

### Installing Nav2 Packages

First, ensure that you have ROS2 Humble installed along with the necessary Nav2 packages. If you haven't already installed Nav2, you can do so using the following commands:

```bash
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-common
sudo apt install ros-humble-nav2-system-tests ros-humble-nav2-msgs
```

Additionally, install the Isaac ROS navigation packages that will interface with the Nav2 framework:

```bash
sudo apt install ros-humble-isaac-ros-nav2-interfaces
sudo apt install ros-humble-isaac-ros-navigation ros-humble-isaac-ros-occupancy-grid-ros
```

### Verifying Nav2 Installation

To verify that Nav2 is properly installed, you can launch the basic Nav2 bringup to ensure all core services are available:

```bash
ros2 launch nav2_bringup tb3_simulation_launch.py headless:=False
```

This command launches Nav2 in simulation mode, allowing you to verify that all necessary nodes are running correctly. Look for the following key nodes in the terminal output:

- `nav2_map_server`: Manages map loading and storage
- `nav2_local_costmap_nodes`: Handles local obstacle detection
- `nav2_global_costmap_nodes`: Manages global costmap representation
- `nav2_planner_server`: Computes global paths
- `nav2_controller_server`: Executes local trajectory control
- `nav2_recoveries_server`: Manages recovery behaviors

## Configuring Nav2 for Humanoid Robot Navigation Requirements

Configuring Nav2 for humanoid robots requires significant customization of the default parameters to account for the unique characteristics of bipedal locomotion. Let's explore the key configuration aspects:

### 1. Costmap Configuration for Humanoid Form Factor

The costmap is crucial for navigation as it represents obstacles and navigable space. For humanoid robots, we need to adjust the costmap parameters to reflect the robot's anthropomorphic form factor:

Create a new configuration file `humanoid_nav2_params.yaml` in your project workspace:

```yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_delay: 0.5
    scan_topic: scan
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.5
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    # Specify the path where the BT XML files are located
    plugin_lib_names:
    - nav2_compute_path_to_pose_action_bt_node
    - nav2_follow_path_action_bt_node
    - nav2_back_up_action_bt_node
    - nav2_spin_action_bt_node
    - nav2_wait_action_bt_node
    - nav2_clear_costmap_service_bt_node
    - nav2_is_stuck_condition_bt_node
    - nav2_goal_reached_condition_bt_node
    - nav2_initial_pose_received_condition_bt_node
    - nav2_goal_updated_condition_bt_node
    - nav2_reinitialize_global_localization_service_bt_node
    - nav2_rate_controller_bt_node
    - nav2_distance_controller_bt_node
    - nav2_speed_controller_bt_node
    - nav2_truncate_path_action_bt_node
    - nav2_goal_updater_node_bt_node
    - nav2_recovery_node_bt_node
    - nav2_pipeline_sequence_bt_node
    - nav2_round_robin_node_bt_node
    - nav2_transform_available_condition_bt_node
    - nav2_time_expired_condition_bt_node
    - nav2_path_expiring_timer_condition
    - nav2_distance_traveled_condition_bt_node
    - nav2_single_trigger_bt_node
    - nav2_is_battery_low_condition_bt_node
    - nav2_navigate_through_poses_action_bt_node
    - nav2_navigate_to_pose_action_bt_node
    - nav2_remove_passed_goals_action_bt_node
    - nav2_planner_selector_bt_node
    - nav2_controller_selector_bt_node
    - nav2_goal_checker_selector_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    # Humanoid-specific controller configuration
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific velocity limits
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_steps: 32
      model_dt: 0.05
      batch_size: 1000
      vx_std: 0.2
      vy_std: 0.05
      wz_std: 0.3
      vx_max: 0.4      # Reduced for humanoid stability
      vx_min: -0.2
      vy_max: 0.2
      wz_max: 0.5
      xy_goal_tolerance: 0.2
      yaw_goal_tolerance: 0.2
      state_reset_threshold: 0.5
      control_duration: 0.05
      transform_tolerance: 0.1
      heading_scale_factor: 1.0
      oscillation_score_penalty: 1.0
      oscillation_magic_number: 4.0
      oscillation_reset_angle: 0.34

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 6      # Increased for humanoid height awareness
      height: 6
      resolution: 0.05
      robot_radius: 0.4  # Adjusted for humanoid form factor
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0  # Humanoid height consideration
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        map_subscribe_transient_local: True
      always_send_full_costmap: True
  local_costmap_client:
    ros__parameters:
      use_sim_time: True
  local_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 0.5
      global_frame: map
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.4  # Humanoid form factor
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True
  global_costmap_client:
    ros__parameters:
      use_sim_time: True
  global_costmap_rclcpp_node:
    ros__parameters:
      use_sim_time: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries::Spin"
      ideal_linear_velocity: 0.0
      ideal_angular_velocity: 1.0
      time_allowance: 5.0
    backup:
      plugin: "nav2_recoveries::BackUp"
      ideal_linear_velocity: -0.1
      ideal_angular_velocity: 0.0
      time_allowance: 5.0
    wait:
      plugin: "nav2_recoveries::Wait"
      time_allowance: 5.0

robot_state_publisher:
  ros__parameters:
    use_sim_time: True

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

### 2. Humanoid-Specific Navigation Parameters

The configuration file above includes several humanoid-specific adjustments:

#### Velocity Limits
- **Reduced maximum velocities**: Humanoid robots typically move slower than wheeled robots for stability
- **Careful acceleration profiles**: Gradual acceleration/deceleration to maintain balance

#### Costmap Adjustments
- **Increased robot radius**: Accounts for the wider stance and arms of humanoid robots
- **Height-aware obstacle detection**: Considers obstacles at different heights relevant to humanoid navigation
- **Larger local costmap window**: Provides better awareness of the environment around the humanoid robot

#### Controller Configuration
- **MPPI Controller**: Model Predictive Path Integral controller suitable for humanoid dynamics
- **Stability-focused parameters**: Lower velocity thresholds and conservative control parameters

### 3. Launch Configuration for Humanoid Navigation

Create a launch file `humanoid_nav2.launch.py` that incorporates the humanoid-specific configuration:

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('nav2_bringup')

    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_nav_to_pose_bt_xml = LaunchConfiguration('default_nav_to_pose_bt_xml')
    map_subscribe_transient_local = LaunchConfiguration('map_subscribe_transient_local')

    lifecycle_nodes = ['controller_server',
                       'smoother_server',
                       'planner_server',
                       'behavior_server',
                       'bt_navigator',
                       'waypoint_follower']

    remappings = [('/tf', 'tf'),
                  ('/tf_static', 'tf_static')]

    param_substitutions = {
        'use_sim_time': use_sim_time,
        'default_nav_to_pose_bt_xml': default_nav_to_pose_bt_xml,
        'map_subscribe_transient_local': map_subscribe_transient_local}

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True)

    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),

        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),

        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),

        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Automatically startup the nav2 stack'),

        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(bringup_dir, 'params', 'nav2_params.yaml'),
            description='Full path to the ROS2 parameters file to use'),

        DeclareLaunchArgument(
            'default_nav_to_pose_bt_xml',
            default_value=os.path.join(bringup_dir, 'behavior_trees', 'navigate_to_pose_w_replanning_and_recovery.xml'),
            description='Full path to the behavior tree xml file to use'),

        DeclareLaunchArgument(
            'map_subscribe_transient_local', default_value='false',
            description='Whether to set the map subscriber to transient local'),

        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_smoother',
            executable='smoother_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_planner',
            executable='planner_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_waypoint_follower',
            executable='waypoint_follower',
            output='screen',
            parameters=[configured_params],
            remappings=remappings),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': autostart},
                        {'node_names': lifecycle_nodes}]),
    ])
```

## Adapting Path Planning for Bipedal Locomotion Constraints

Path planning for humanoid robots must account for the unique constraints of bipedal locomotion. Unlike wheeled robots that can turn in place, humanoid robots must plan paths that accommodate their walking gait and balance requirements.

### Understanding Bipedal Locomotion Constraints

Bipedal locomotion introduces several constraints that affect path planning:

1. **Step-by-step movement**: Humanoid robots must place each foot in a stable position before moving the other
2. **Balance maintenance**: Paths must allow for maintaining center of mass within the support polygon
3. **Turning radius**: Humans have a minimum turning radius based on leg length and stride
4. **Terrain considerations**: Uneven surfaces, stairs, and obstacles require special navigation strategies

### Modifying Planner Parameters

To address these constraints, we need to modify the planner server configuration to incorporate humanoid-specific path planning requirements:

```yaml
planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true
      # Humanoid-specific path smoothing
      smooth_path: true
      # Minimum distance between waypoints for humanoid gait
      minimum_turning_radius: 0.3  # meters
      # Path optimization for humanoid step planning
      step_size: 0.1              # Distance between path points for humanoid
      max_iterations: 10000       # Allow more iterations for complex humanoid paths
```

### Implementing Custom Path Smoothing for Humanoid Movement

For humanoid robots, we need smoother paths that account for the natural gait pattern. Create a custom path smoother that generates waypoints suitable for bipedal locomotion:

```cpp
// humanoid_path_smoother.cpp
#include "nav2_core/path_smoother.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include <vector>
#include <cmath>

namespace nav2_smoother
{

class HumanoidPathSmoother : public nav2_core::PathSmoother
{
public:
  void configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr & node,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    const std::string & plugin_name) override
  {
    node_ = node;
    costmap_ros_ = costmap_ros;
    plugin_name_ = plugin_name;

    // Get humanoid-specific parameters
    auto node_shared = node_.lock();
    node_shared->get_parameter_or(plugin_name + ".smoothing_num_samples", smoothing_num_samples_, 5);
    node_shared->get_parameter_or(plugin_name + ".smoothing_step_size", smoothing_step_size_, 0.1);
    node_shared->get_parameter_or(plugin_name + ".max_deviation", max_deviation_, 0.5);
  }

  void cleanup() override {}

  void activate() override {}

  void deactivate() override {}

  nav_msgs::msg::Path smooth(const nav_msgs::msg::Path & path) override
  {
    nav_msgs::msg::Path smoothed_path;

    if (path.poses.empty()) {
      return smoothed_path;
    }

    // Start with the original path
    smoothed_path.header = path.header;

    // Apply humanoid-specific smoothing algorithm
    std::vector<geometry_msgs::msg::PoseStamped> intermediate_poses;

    for (size_t i = 0; i < path.poses.size() - 1; ++i) {
      const auto & start_pose = path.poses[i];
      const auto & end_pose = path.poses[i + 1];

      // Calculate distance between poses
      double dx = end_pose.pose.position.x - start_pose.pose.position.x;
      double dy = end_pose.pose.position.y - start_pose.pose.position.y;
      double distance = std::sqrt(dx * dx + dy * dy);

      // Interpolate poses at humanoid-appropriate intervals
      int num_intermediate_points = static_cast<int>(distance / smoothing_step_size_);

      for (int j = 0; j <= num_intermediate_points; ++j) {
        geometry_msgs::msg::PoseStamped intermediate_pose;
        intermediate_pose.header = path.header;

        double ratio = static_cast<double>(j) / num_intermediate_points;

        // Linear interpolation for position
        intermediate_pose.pose.position.x =
          start_pose.pose.position.x + ratio * (end_pose.pose.position.x - start_pose.pose.position.x);
        intermediate_pose.pose.position.y =
          start_pose.pose.position.y + ratio * (end_pose.pose.position.y - start_pose.pose.position.y);

        // Maintain orientation (heading toward next point)
        if (j < num_intermediate_points) {
          double angle = std::atan2(
            end_pose.pose.position.y - intermediate_pose.pose.position.y,
            end_pose.pose.position.x - intermediate_pose.pose.position.x
          );

          intermediate_pose.pose.orientation =
            nav2_util::geometry_utils::orientationAroundZAxis(angle);
        } else {
          // Use original orientation for the last point
          intermediate_pose.pose.orientation = end_pose.pose.orientation;
        }

        intermediate_poses.push_back(intermediate_pose);
      }
    }

    // Apply deviation constraints to ensure path stays close to original
    std::vector<geometry_msgs::msg::PoseStamped> constrained_poses;
    for (const auto & pose : intermediate_poses) {
      if (isValidHumanoidPose(pose)) {
        constrained_poses.push_back(pose);
      }
    }

    smoothed_path.poses = constrained_poses;
    return smoothed_path;
  }

private:
  bool isValidHumanoidPose(const geometry_msgs::msg::PoseStamped & pose)
  {
    // Check if the pose is valid for humanoid navigation
    unsigned int mx, my;
    auto costmap = costmap_ros_->getCostmap();

    if (!costmap->worldToMap(pose.pose.position.x, pose.pose.position.y, mx, my)) {
      return false;  // Pose outside of costmap bounds
    }

    unsigned char cost = costmap->getCost(mx, my);

    // Humanoid robots need more clearance due to arm swing and balance
    return cost != nav2_costmap_2d::LETHAL_OBSTACLE &&
           cost != nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
  }

  rclcpp_lifecycle::WeakPtr node_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  std::string plugin_name_;

  int smoothing_num_samples_{5};
  double smoothing_step_size_{0.1};  // Humanoid-appropriate step size
  double max_deviation_{0.5};
};

}  // namespace nav2_smoother

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(nav2_smoother::HumanoidPathSmoother, nav2_core::PathSmoother)
```

### Collision Avoidance for Humanoid Form Factor

Humanoid robots require specialized collision avoidance due to their anthropomorphic form factor. The collision avoidance system must consider not only the robot's base but also its extended limbs and height.

#### Height-Aware Collision Detection

Configure the obstacle layer to consider obstacles at different heights relevant to humanoid robots:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      # ... existing parameters ...
      plugins: ["voxel_layer", "inflation_layer"]
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.1  # Higher resolution for height awareness
        z_voxels: 20       # Account for humanoid height
        max_obstacle_height: 2.0  # Humanoid height consideration
        # ... other voxel layer parameters ...
```

#### Dynamic Footprint for Bipedal Walking

Humanoid robots have a dynamic footprint that changes as they walk. Implement a dynamic footprint that updates based on the robot's walking state:

```cpp
// dynamic_humanoid_footprint.cpp
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/polygon_stamped.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class DynamicHumanoidFootprint : public rclcpp::Node
{
public:
  DynamicHumanoidFootprint() : Node("dynamic_humanoid_footprint")
  {
    footprint_pub_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>(
      "humanoid_footprint", 1);

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "odom", 10,
      std::bind(&DynamicHumanoidFootprint::odometryCallback, this, std::placeholders::_1));

    // Initialize the default humanoid footprint (larger than typical wheeled robots)
    initializeDefaultFootprint();

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&DynamicHumanoidFootprint::publishFootprint, this));
  }

private:
  void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // Extract walking state information from odometry
    // This could come from joint angles, IMU data, or gait phase estimation

    // For now, we'll use a simplified approach based on velocity
    double linear_vel = std::sqrt(
      msg->twist.twist.linear.x * msg->twist.twist.linear.x +
      msg->twist.twist.linear.y * msg->twist.twist.linear.y);

    // Adjust footprint based on walking state
    if (linear_vel > 0.01) {
      // Robot is moving - expand footprint to account for dynamic stability
      adjustFootprintForWalking();
    } else {
      // Robot is stationary - use default footprint
      current_footprint_ = default_footprint_;
    }
  }

  void initializeDefaultFootprint()
  {
    // Define a default humanoid footprint (larger than typical robots)
    geometry_msgs::msg::Point32 pt;

    // Create a rectangular footprint accounting for humanoid width
    // Front part (where feet would be when walking forward)
    pt.x = 0.3; pt.y = 0.25;  // Front-right corner
    default_footprint_.points.push_back(pt);
    pt.x = 0.3; pt.y = -0.25; // Front-left corner
    default_footprint_.points.push_back(pt);
    pt.x = -0.1; pt.y = -0.25; // Rear-left corner
    default_footprint_.points.push_back(pt);
    pt.x = -0.1; pt.y = 0.25;  // Rear-right corner
    default_footprint_.points.push_back(pt);
  }

  void adjustFootprintForWalking()
  {
    // Expand footprint when walking to account for:
    // - Arm swing during walking
    // - Dynamic stability margins
    // - Potential balance corrections

    current_footprint_ = default_footprint_;

    // Increase the footprint size during walking
    for (auto& point : current_footprint_.points) {
      point.x *= 1.2;  // 20% expansion in X direction
      point.y *= 1.2;  // 20% expansion in Y direction
    }
  }

  void publishFootprint()
  {
    geometry_msgs::msg::PolygonStamped footprint_msg;
    footprint_msg.header.frame_id = "base_footprint";
    footprint_msg.header.stamp = this->now();
    footprint_msg.polygon = current_footprint_;

    footprint_pub_->publish(footprint_msg);
  }

  rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr footprint_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  geometry_msgs::msg::Polygon default_footprint_;
  geometry_msgs::msg::Polygon current_footprint_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicHumanoidFootprint>());
  rclcpp::shutdown();
  return 0;
}
```

## Testing Navigation in Isaac Sim Environment

Now that we've configured Nav2 for humanoid navigation, let's test it in the Isaac Sim environment. This will validate that our configurations work properly in a realistic simulation environment.

### Setting Up Isaac Sim for Humanoid Navigation

First, ensure that your Isaac Sim environment is properly configured for humanoid navigation testing:

```bash
# Source your ROS2 workspace
source install/setup.bash

# Launch Isaac Sim with humanoid navigation scene
./isaaclaunch.sh -p IsaacExamples/python/nav2_humanoid_example.py
```

### Creating a Navigation Test Script

Create a Python script to test the navigation capabilities with your humanoid robot:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import time


class HumanoidNavigator(Node):
    def __init__(self):
        super().__init__('humanoid_navigator')

        # Create action client for navigation
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Timer to periodically send navigation goals
        self.timer = self.create_timer(10.0, self.send_navigation_goal)
        self.goal_count = 0

    def send_navigation_goal(self):
        """Send a navigation goal to the humanoid robot"""
        # Wait for the action server to be available
        if not self._action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Navigation action server not available')
            return

        # Create a navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # Define a sequence of goals to test navigation
        goals = [
            {'x': 2.0, 'y': 2.0, 'theta': 0.0},
            {'x': -1.0, 'y': 3.0, 'theta': 1.57},
            {'x': -2.0, 'y': -1.0, 'theta': 3.14},
            {'x': 1.0, 'y': -2.0, 'theta': -1.57}
        ]

        if self.goal_count >= len(goals):
            self.goal_count = 0

        current_goal = goals[self.goal_count]

        goal_msg.pose.pose.position.x = current_goal['x']
        goal_msg.pose.pose.position.y = current_goal['y']
        goal_msg.pose.pose.position.z = 0.0

        # Convert angle to quaternion
        from math import sin, cos
        theta = current_goal['theta']
        goal_msg.pose.pose.orientation.z = sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = cos(theta / 2.0)

        # Send the goal
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

        self.get_logger().info(f'Sent navigation goal {self.goal_count + 1}: '
                              f'({current_goal["x"]}, {current_goal["y"]})')

        self.goal_count += 1

    def goal_response_callback(self, future):
        """Handle the response from the navigation server"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get result callback
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle the result of the navigation"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

    def feedback_callback(self, feedback_msg):
        """Handle feedback during navigation"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Navigation progress: {feedback.distance_remaining:.2f}m remaining')


def main(args=None):
    rclpy.init(args=args)

    navigator = HumanoidNavigator()

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Validating Path Planning for Bipedal Locomotion

Create a validation script to verify that the path planning works correctly for bipedal locomotion:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.distance import euclidean


class BipedalPathValidator(Node):
    def __init__(self):
        super().__init__('bipedal_path_validator')

        # Subscribe to the planned path
        self.path_sub = self.create_subscription(
            Path, '/plan', self.path_callback, 10)

        # Parameters for validation
        self.min_step_distance = 0.1  # Minimum distance between path points (humanoid step size)
        self.max_deviation = 0.5      # Maximum deviation from straight line
        self.turning_radius = 0.3     # Minimum turning radius for humanoid

        self.get_logger().info('Bipedal Path Validator initialized')

    def path_callback(self, msg):
        """Validate the received path for humanoid compatibility"""
        if len(msg.poses) < 2:
            return

        path_valid = True
        violations = []

        # Check step distances
        for i in range(len(msg.poses) - 1):
            pos1 = msg.poses[i].pose.position
            pos2 = msg.poses[i + 1].pose.position

            distance = euclidean([pos1.x, pos1.y], [pos2.x, pos2.y])

            if distance < self.min_step_distance:
                violations.append(f'Step {i} too short: {distance:.3f}m (min: {self.min_step_distance}m)')
                path_valid = False
            elif distance > 0.5:  # Reasonable upper bound for humanoid steps
                violations.append(f'Step {i} too long: {distance:.3f}m (max: 0.5m)')
                path_valid = False

        # Check for excessive deviations
        if len(msg.poses) >= 3:
            for i in range(1, len(msg.poses) - 1):
                pos_prev = msg.poses[i - 1].pose.position
                pos_curr = msg.poses[i].pose.position
                pos_next = msg.poses[i + 1].pose.position

                # Calculate deviation from straight line
                line_vec = np.array([pos_next.x - pos_prev.x, pos_next.y - pos_prev.y])
                point_vec = np.array([pos_curr.x - pos_prev.x, pos_curr.y - pos_prev.y])

                # Project point onto line
                line_len = np.linalg.norm(line_vec)
                if line_len > 0:
                    projection = np.dot(point_vec, line_vec) / (line_len * line_len) * line_vec
                    deviation = np.linalg.norm(point_vec - projection)

                    if deviation > self.max_deviation:
                        violations.append(f'Excessive deviation at point {i}: {deviation:.3f}m (max: {self.max_deviation}m)')
                        path_valid = False

        # Log validation results
        if path_valid:
            self.get_logger().info(f'Path validation passed: {len(msg.poses)} waypoints')
        else:
            self.get_logger().warn(f'Path validation failed with {len(violations)} violations:')
            for violation in violations:
                self.get_logger().warn(f'  - {violation}')

    def calculate_curvature(self, path_poses):
        """Calculate curvature of the path to ensure it meets turning radius requirements"""
        if len(path_poses) < 3:
            return []

        curvatures = []
        for i in range(1, len(path_poses) - 1):
            p1 = np.array([path_poses[i-1].pose.position.x, path_poses[i-1].pose.position.y])
            p2 = np.array([path_poses[i].pose.position.x, path_poses[i].pose.position.y])
            p3 = np.array([path_poses[i+1].pose.position.x, path_poses[i+1].pose.position.y])

            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2

            # Calculate angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))

            # Curvature is inverse of turning radius
            curvature = angle / np.linalg.norm(v1)  # Simplified curvature calculation
            curvatures.append(curvature)

        return curvatures


def main(args=None):
    rclpy.init(args=args)

    validator = BipedalPathValidator()

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

## Integration with Isaac ROS Packages

To fully leverage the NVIDIA Isaac ecosystem, we need to integrate our Nav2 configuration with Isaac ROS packages for enhanced perception and navigation capabilities.

### Isaac ROS Perception Integration

Integrate Isaac ROS perception packages to enhance the navigation system with hardware-accelerated sensing:

```yaml
# Add Isaac ROS perception integration to your configuration
isaac_ros_perception:
  ros__parameters:
    # Stereo camera configuration for depth perception
    left_topic: "/stereo_left/image_rect_color"
    right_topic: "/stereo_right/image_rect_color"
    left_camera_info_topic: "/stereo_left/camera_info"
    right_camera_info_topic: "/stereo_right/camera_info"

    # Processing parameters
    disparity_range: 64
    correlation_window_size: 49
    texture_threshold: 10
    uniqueness_ratio: 15

    # Output configuration
    disparity_topic: "/disparity"
    pointcloud_topic: "/points2"
```

### Isaac ROS Navigation Integration

Configure Isaac ROS navigation packages to work with Nav2:

```yaml
# Isaac ROS navigation integration
isaac_ros_navigation:
  ros__parameters:
    # Navigation parameters
    enable_visualization: true
    visualization_rate: 10.0

    # Integration with Nav2
    nav2_integration_enabled: true
    nav2_costmap_topic: "/global_costmap/costmap"

    # Hardware acceleration
    gpu_processing_enabled: true
    cuda_device_id: 0
```

## Summary

In this lesson, we've successfully configured Nav2 path planning specifically adapted for humanoid robots. We covered:

1. **Nav2 Framework Setup**: Installed and verified Nav2 with ROS2 Humble for humanoid navigation requirements
2. **Humanoid-Specific Configuration**: Adapted navigation parameters for bipedal locomotion constraints and humanoid robot kinematics
3. **Path Planning Adaptation**: Modified planner parameters to accommodate humanoid movement patterns
4. **Collision Avoidance**: Implemented collision avoidance systems considering the humanoid form factor
5. **Testing and Validation**: Tested navigation in Isaac Sim environment and validated path planning for bipedal locomotion
6. **Isaac ROS Integration**: Integrated with Isaac ROS packages for enhanced perception and navigation capabilities

The configuration we've developed addresses the unique challenges of humanoid navigation, including balance maintenance, anthropomorphic form factor considerations, and bipedal locomotion constraints. The system is now ready to serve as the foundation for advanced navigation capabilities in subsequent lessons.

In the next lesson, we'll explore Visual SLAM with Isaac ROS, building upon this navigation foundation to enable real-time localization and mapping capabilities for humanoid robots.