# Lesson 2.2 – Action Grounding and Motion Planning

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement action grounding systems that connect AI decisions to physical movements
- Configure motion planning algorithms for humanoid robots
- Translate high-level goals into specific motor commands
- Understand how to use motion planning libraries, trajectory generation tools, and ROS 2 interfaces

## Introduction to Action Grounding

Action grounding is the critical process that connects abstract AI decisions to concrete physical movements in humanoid robots. While the previous lesson focused on how AI systems make decisions based on multimodal inputs, this lesson addresses how those decisions are translated into specific motor commands that result in purposeful robot behavior.

Action grounding involves several key challenges:

1. **Semantic Gap Bridging**: Converting high-level goals ("pick up the red cup") into low-level motor commands
2. **Physical Constraints**: Ensuring actions are feasible given robot kinematics and environmental constraints
3. **Temporal Coordination**: Sequencing actions in time to achieve complex behaviors
4. **Safety Integration**: Maintaining safety throughout the grounding process

## Understanding Action Grounding Systems

### The Action Grounding Pipeline

Action grounding systems follow a structured pipeline that transforms abstract decisions into executable actions:

**High-Level Goal Specification**:
- Natural language instructions or task descriptions
- Environmental context and constraints
- Success criteria and safety requirements

**Task Decomposition**:
- Breaking complex goals into simpler subtasks
- Identifying required skills and capabilities
- Sequencing subtasks in appropriate order

**Motion Planning**:
- Generating specific movement trajectories
- Accounting for robot kinematics and dynamics
- Planning collision-free paths

**Motor Command Generation**:
- Converting trajectories to joint commands
- Controlling actuators and effectors
- Executing coordinated movements

### Symbol Grounding in Action Systems

Symbol grounding extends beyond connecting language to visual objects to connecting language to physical actions. This creates a mapping between abstract concepts and concrete behaviors:

```python
class ActionSymbolGrounding:
    def __init__(self):
        self.action_mappings = {
            'grasp': ['close_gripper', 'approach_object', 'lift'],
            'transport': ['navigate', 'carry_object', 'position'],
            'place': ['align_object', 'release', 'withdraw']
        }
        self.object_action_mappings = {
            'cup': ['grasp', 'lift', 'carry'],
            'box': ['push', 'slide', 'reposition'],
            'tool': ['grasp', 'manipulate', 'use']
        }

    def ground_action_to_motor_commands(self, action_verb, target_object):
        """Map abstract action to specific motor commands"""
        # Get base action sequence
        if action_verb in self.action_mappings:
            base_actions = self.action_mappings[action_verb]
        else:
            return self.get_default_action_sequence(action_verb)

        # Adapt to specific object properties
        if target_object and target_object['type'] in self.object_action_mappings:
            object_specific_actions = self.object_action_mappings[target_object['type']]
            return self.adapt_actions_to_object(base_actions, object_specific_actions, target_object)

        return base_actions

    def adapt_actions_to_object(self, base_actions, object_actions, target_object):
        """Adapt general actions to specific object properties"""
        adapted_actions = []

        for action in base_actions:
            if action in object_actions:
                # Adjust parameters based on object properties
                adjusted_action = self.adjust_action_for_object(action, target_object)
                adapted_actions.append(adjusted_action)
            else:
                adapted_actions.append(action)

        return adapted_actions
```

### Action Representation and Planning

Effective action grounding requires proper representation of both the action space and the environmental constraints:

**Action Space Representation**:
- Joint space: Direct control of robot joint angles
- Cartesian space: Control of end-effector position and orientation
- Task space: High-level task specifications
- Skill space: Pre-learned movement patterns

**Constraint Integration**:
- Kinematic constraints (joint limits, reachability)
- Dynamic constraints (velocity, acceleration limits)
- Environmental constraints (obstacles, workspace boundaries)
- Safety constraints (collision avoidance, force limits)

## Motion Planning for Humanoid Robots

### Humanoid-Specific Motion Planning Challenges

Humanoid robots present unique challenges for motion planning due to their complex kinematic structure and the need for stable, human-like movement:

**Balance and Stability**:
- Maintaining center of mass within support polygon
- Coordinating upper and lower body movements
- Managing zero-moment point (ZMP) for stable locomotion

**Multi-Limb Coordination**:
- Synchronizing arm and leg movements
- Avoiding self-collisions between limbs
- Managing redundant degrees of freedom

**Human-like Movement Patterns**:
- Generating natural-looking motion trajectories
- Maintaining anthropomorphic movement characteristics
- Adapting to different walking gaits and postures

### Motion Planning Algorithms

Several motion planning algorithms are particularly suited for humanoid robots:

**RRT (Rapidly-exploring Random Trees)**:
- Effective for high-dimensional configuration spaces
- Good for finding collision-free paths in complex environments
- Can handle kinematic constraints and joint limits

**Trajectory Optimization**:
- Generates smooth, time-parameterized trajectories
- Can optimize for multiple objectives (speed, energy, safety)
- Incorporates dynamic constraints and stability requirements

**Sampling-Based Methods**:
- Efficient for complex humanoid kinematics
- Can handle non-holonomic constraints
- Suitable for real-time applications with proper optimization

### Implementation Example: Humanoid Motion Planner

```python
import numpy as np
from scipy.interpolate import interp1d
from typing import List, Dict, Tuple, Optional

class HumanoidMotionPlanner:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.kinematics_solver = self._initialize_kinematics_solver()
        self.collision_checker = self._initialize_collision_checker()
        self.stability_checker = self._initialize_stability_checker()

    def _initialize_kinematics_solver(self):
        """Initialize inverse kinematics solver for humanoid robot"""
        return {
            'arm_ik_solver': self._setup_arm_ik_solver(),
            'leg_ik_solver': self._setup_leg_ik_solver(),
            'whole_body_solver': self._setup_whole_body_solver()
        }

    def _initialize_collision_checker(self):
        """Initialize collision detection system"""
        return {
            'self_collision_threshold': 0.05,  # meters
            'environment_collision_threshold': 0.1  # meters
        }

    def _initialize_stability_checker(self):
        """Initialize stability verification system"""
        return {
            'zmp_limits': {'x': (-0.1, 0.1), 'y': (-0.05, 0.05)},  # meters
            'com_support_threshold': 0.05  # meters from support polygon edge
        }

    def plan_arm_motion(self, start_pose, goal_pose, obstacles=None):
        """
        Plan motion for humanoid arm to reach goal pose
        """
        # Check if goal is reachable
        if not self._is_reachable(goal_pose, 'arm'):
            raise ValueError("Goal pose is not reachable")

        # Generate initial trajectory using inverse kinematics
        initial_trajectory = self._generate_ik_trajectory(start_pose, goal_pose)

        # Optimize trajectory to avoid obstacles
        if obstacles:
            optimized_trajectory = self._optimize_trajectory_for_obstacles(
                initial_trajectory, obstacles
            )
        else:
            optimized_trajectory = initial_trajectory

        # Verify joint limits and smooth trajectory
        final_trajectory = self._verify_and_smooth_trajectory(optimized_trajectory)

        return final_trajectory

    def plan_walking_trajectory(self, start_pose, goal_position, step_height=0.05):
        """
        Plan walking trajectory for humanoid to reach goal position
        """
        # Calculate number of steps needed
        distance = np.linalg.norm(np.array(goal_position[:2]) - np.array(start_pose[:2]))
        step_length = 0.3  # meters
        num_steps = int(np.ceil(distance / step_length))

        # Generate step locations
        step_positions = self._generate_step_positions(
            start_pose, goal_position, num_steps
        )

        # Generate complete walking trajectory
        walking_trajectory = self._generate_walking_trajectory(
            step_positions, step_height
        )

        # Verify stability throughout trajectory
        if not self._verify_stability(walking_trajectory):
            raise ValueError("Walking trajectory is not stable")

        return walking_trajectory

    def _is_reachable(self, pose, body_part):
        """Check if pose is reachable by specified body part"""
        # Calculate workspace bounds for body part
        workspace_bounds = self.robot_model.get_workspace_bounds(body_part)

        # Check if pose is within workspace
        position = np.array(pose[:3])
        is_in_workspace = (
            workspace_bounds['min'][0] <= position[0] <= workspace_bounds['max'][0] and
            workspace_bounds['min'][1] <= position[1] <= workspace_bounds['max'][1] and
            workspace_bounds['min'][2] <= position[2] <= workspace_bounds['max'][2]
        )

        return is_in_workspace

    def _generate_ik_trajectory(self, start_pose, goal_pose):
        """Generate trajectory using inverse kinematics"""
        # Interpolate between start and goal poses
        num_waypoints = 50
        t_values = np.linspace(0, 1, num_waypoints)

        trajectory = []
        for t in t_values:
            # Linear interpolation in Cartesian space
            current_pose = self._interpolate_poses(start_pose, goal_pose, t)

            # Solve inverse kinematics
            joint_angles = self._solve_inverse_kinematics(current_pose)

            trajectory.append({
                'time': t * 2.0,  # Assume 2 seconds for the motion
                'joint_angles': joint_angles,
                'cartesian_pose': current_pose
            })

        return trajectory

    def _interpolate_poses(self, start_pose, goal_pose, t):
        """Interpolate between two poses"""
        start_pos = np.array(start_pose[:3])
        goal_pos = np.array(goal_pose[:3])

        # Linear interpolation for position
        current_pos = start_pos + t * (goal_pos - start_pos)

        # Slerp for orientation (simplified as linear interpolation)
        start_quat = np.array(start_pose[3:])
        goal_quat = np.array(goal_pose[3:])
        current_quat = start_quat + t * (goal_quat - start_quat)
        current_quat = current_quat / np.linalg.norm(current_quat)  # Normalize

        return list(current_pos) + list(current_quat)

    def _solve_inverse_kinematics(self, pose):
        """Solve inverse kinematics for given pose"""
        # In practice, this would use a proper IK solver
        # For this example, we'll simulate the process
        target_position = np.array(pose[:3])

        # Calculate joint angles (simplified calculation)
        joint_angles = self._calculate_joint_angles_from_position(target_position)

        return joint_angles

    def _calculate_joint_angles_from_position(self, position):
        """Calculate joint angles to reach given position (simplified)"""
        # This is a simplified calculation - real IK would be more complex
        # For a 6-DOF arm, we might calculate angles based on position
        x, y, z = position

        # Simplified inverse kinematics (for demonstration)
        # In practice, use proper IK libraries like KDL, MoveIt, etc.
        joint_angles = [
            np.arctan2(y, x),  # Shoulder pan
            np.arctan2(z, np.sqrt(x**2 + y**2)),  # Shoulder lift
            0.0,  # Elbow
            0.0,  # Wrist 1
            0.0,  # Wrist 2
            0.0   # Wrist 3
        ]

        return joint_angles

    def _optimize_trajectory_for_obstacles(self, trajectory, obstacles):
        """Optimize trajectory to avoid obstacles"""
        # Use trajectory optimization to avoid obstacles
        optimized_trajectory = []

        for waypoint in trajectory:
            # Check for collisions
            collision_free = True
            for obstacle in obstacles:
                if self._check_collision(waypoint, obstacle):
                    collision_free = False
                    break

            if collision_free:
                optimized_trajectory.append(waypoint)
            else:
                # Find alternative path around obstacle
                alternative_waypoint = self._find_alternative_waypoint(waypoint, obstacles)
                optimized_trajectory.append(alternative_waypoint)

        return optimized_trajectory

    def _check_collision(self, waypoint, obstacle):
        """Check if waypoint causes collision with obstacle"""
        # Calculate robot position at waypoint
        robot_pos = self._calculate_robot_position(waypoint['joint_angles'])

        # Check distance to obstacle
        obstacle_pos = obstacle['position']
        distance = np.linalg.norm(np.array(robot_pos) - np.array(obstacle_pos))

        return distance < self.collision_checker['environment_collision_threshold']

    def _find_alternative_waypoint(self, waypoint, obstacles):
        """Find alternative waypoint to avoid obstacles"""
        # This would implement a path planning algorithm
        # For simplicity, we'll just offset the waypoint
        original_pos = np.array(waypoint['cartesian_pose'][:3])

        # Try different offsets to find collision-free position
        for offset in [0.1, -0.1, 0.2, -0.2]:
            for axis in [0, 1, 2]:  # x, y, z
                offset_pos = original_pos.copy()
                offset_pos[axis] += offset

                test_pose = list(offset_pos) + waypoint['cartesian_pose'][3:]

                # Check if this offset creates a collision-free path
                collision_free = True
                for obstacle in obstacles:
                    if self._check_collision({'cartesian_pose': test_pose}, obstacle):
                        collision_free = False
                        break

                if collision_free:
                    # Update joint angles for new pose
                    new_joint_angles = self._solve_inverse_kinematics(test_pose)
                    return {
                        'time': waypoint['time'],
                        'joint_angles': new_joint_angles,
                        'cartesian_pose': test_pose
                    }

        # If no collision-free alternative found, return original (will cause error later)
        return waypoint

    def _verify_and_smooth_trajectory(self, trajectory):
        """Verify joint limits and smooth trajectory"""
        verified_trajectory = []

        for i, waypoint in enumerate(trajectory):
            # Check joint limits
            joint_angles = np.array(waypoint['joint_angles'])

            # Apply joint limits (example values)
            joint_limits_min = np.array([-2.0, -1.5, -2.0, -2.0, -2.0, -2.0])
            joint_limits_max = np.array([2.0, 1.5, 2.0, 2.0, 2.0, 2.0])

            # Clamp to joint limits
            clamped_angles = np.clip(joint_angles, joint_limits_min, joint_limits_max)

            # Smooth with previous waypoint to avoid large jumps
            if i > 0:
                prev_angles = np.array(verified_trajectory[-1]['joint_angles'])
                max_change = 0.1  # radians per step

                # Limit joint angle changes
                angle_diff = clamped_angles - prev_angles
                limited_diff = np.clip(angle_diff, -max_change, max_change)
                clamped_angles = prev_angles + limited_diff

            waypoint['joint_angles'] = clamped_angles.tolist()
            verified_trajectory.append(waypoint)

        return verified_trajectory

    def _generate_step_positions(self, start_pose, goal_position, num_steps):
        """Generate positions for each walking step"""
        # Calculate step positions along straight line
        start_pos = np.array(start_pose[:2])
        goal_pos = np.array(goal_position[:2])

        step_positions = []
        for i in range(num_steps + 1):
            t = i / num_steps
            step_pos = start_pos + t * (goal_pos - start_pos)
            step_positions.append(step_pos.tolist())

        return step_positions

    def _generate_walking_trajectory(self, step_positions, step_height):
        """Generate complete walking trajectory"""
        walking_trajectory = []

        for i, pos in enumerate(step_positions):
            # Add step with appropriate height
            z_offset = step_height if i % 2 == 0 else 0.0  # Alternate step height

            # Create trajectory point
            trajectory_point = {
                'time': i * 0.8,  # 0.8 seconds per step
                'position': [pos[0], pos[1], z_offset],
                'support_leg': 'left' if i % 2 == 0 else 'right'
            }

            walking_trajectory.append(trajectory_point)

        return walking_trajectory

    def _verify_stability(self, trajectory):
        """Verify that walking trajectory is stable"""
        for point in trajectory:
            # Check ZMP (Zero Moment Point) constraints
            zmp_x, zmp_y = self._calculate_zmp(point)

            x_min, x_max = self.stability_checker['zmp_limits']['x']
            y_min, y_max = self.stability_checker['zmp_limits']['y']

            if not (x_min <= zmp_x <= x_max and y_min <= zmp_y <= y_max):
                return False

        return True

    def _calculate_zmp(self, trajectory_point):
        """Calculate Zero Moment Point for trajectory point"""
        # Simplified ZMP calculation
        # In practice, this would use full dynamics model
        pos = trajectory_point['position']

        # Approximate ZMP as center of support polygon
        # For single support, this is roughly the foot position
        zmp_x = pos[0]
        zmp_y = pos[1]

        return zmp_x, zmp_y

# Example usage
def main():
    # Mock robot model
    class MockRobotModel:
        def get_workspace_bounds(self, body_part):
            if body_part == 'arm':
                return {
                    'min': [-0.5, -0.5, 0.0],
                    'max': [0.5, 0.5, 1.0]
                }
            return {'min': [0, 0, 0], 'max': [1, 1, 1]}

    # Initialize motion planner
    robot_model = MockRobotModel()
    planner = HumanoidMotionPlanner(robot_model)

    # Example: Plan arm motion
    start_pose = [0.3, 0.0, 0.8, 0, 0, 0, 1]  # [x, y, z, qx, qy, qz, qw]
    goal_pose = [0.4, 0.2, 0.6, 0, 0, 0, 1]

    obstacles = [
        {'position': [0.35, 0.1, 0.7], 'size': [0.1, 0.1, 0.2]}
    ]

    try:
        trajectory = planner.plan_arm_motion(start_pose, goal_pose, obstacles)
        print(f"Generated trajectory with {len(trajectory)} waypoints")
        print("First waypoint:", trajectory[0])
        print("Last waypoint:", trajectory[-1])
    except ValueError as e:
        print(f"Motion planning failed: {e}")

if __name__ == "__main__":
    main()
```

## Action Grounding Integration with AI Decision-Making

### Connecting Decisions to Actions

Action grounding systems must seamlessly integrate with the AI decision-making frameworks from Lesson 2.1:

```python
class IntegratedDecisionToAction:
    def __init__(self):
        self.motion_planner = HumanoidMotionPlanner(robot_model=None)
        self.action_grounding = ActionSymbolGrounding()
        self.decision_framework = SimpleVLAReasoningEngine()  # From Lesson 2.1

    def process_decision_and_execute(self, visual_data, language_input):
        """Process decision and convert to executable actions"""
        # Step 1: Make decision using AI framework
        decision = self.decision_framework.process_input(visual_data, language_input)

        if decision['decision_type'] == 'safety_error':
            return self._handle_safety_error(decision)

        # Step 2: Ground decision to physical actions
        grounded_actions = self._ground_decision_to_actions(
            decision['action_plan'],
            decision
        )

        # Step 3: Plan detailed motions for each action
        executable_trajectory = self._plan_detailed_trajectories(grounded_actions)

        # Step 4: Validate and execute
        if self._validate_trajectory(executable_trajectory):
            return self._execute_trajectory(executable_trajectory)
        else:
            return self._handle_validation_failure(executable_trajectory)

    def _ground_decision_to_actions(self, action_plan, decision_context):
        """Ground high-level action plan to detailed physical actions"""
        grounded_actions = []

        for action in action_plan:
            if action['action'] == 'move_to_object':
                # Get target object position from context
                target_obj = decision_context.get('matched_object')
                if target_obj:
                    detailed_action = {
                        'type': 'navigation',
                        'target_position': target_obj['position'],
                        'approach_distance': 0.3,
                        'motion_parameters': self._get_navigation_parameters(target_obj)
                    }
                    grounded_actions.append(detailed_action)

            elif action['action'] == 'grasp_object':
                target_obj = decision_context.get('matched_object')
                if target_obj:
                    detailed_action = {
                        'type': 'manipulation',
                        'target_object': target_obj,
                        'grasp_type': 'top_grasp',
                        'motion_parameters': self._get_manipulation_parameters(target_obj)
                    }
                    grounded_actions.append(detailed_action)

        return grounded_actions

    def _plan_detailed_trajectories(self, grounded_actions):
        """Plan detailed motion trajectories for grounded actions"""
        trajectory_sequence = []

        for action in grounded_actions:
            if action['type'] == 'navigation':
                nav_trajectory = self.motion_planner.plan_walking_trajectory(
                    start_pose=[0, 0, 0, 0, 0, 0, 1],
                    goal_position=action['target_position'][:2] + [0]
                )
                trajectory_sequence.extend(nav_trajectory)

            elif action['type'] == 'manipulation':
                # Plan arm motion to grasp object
                obj_pos = action['target_object']['position']
                grasp_pose = self._calculate_grasp_pose(obj_pos, action['grasp_type'])

                arm_trajectory = self.motion_planner.plan_arm_motion(
                    start_pose=[0.3, 0.0, 0.8, 0, 0, 0, 1],
                    goal_pose=grasp_pose
                )
                trajectory_sequence.extend(arm_trajectory)

        return trajectory_sequence

    def _get_navigation_parameters(self, target_object):
        """Get navigation parameters for target object"""
        return {
            'step_height': 0.05,
            'step_length': 0.3,
            'orientation_tolerance': 0.1
        }

    def _get_manipulation_parameters(self, target_object):
        """Get manipulation parameters for target object"""
        return {
            'approach_distance': 0.1,
            'grasp_width': target_object.get('size', [0.1, 0.1, 0.1])[0] * 1.2,
            'grasp_force': 10.0  # Newtons
        }

    def _calculate_grasp_pose(self, object_position, grasp_type):
        """Calculate appropriate grasp pose for object"""
        if grasp_type == 'top_grasp':
            # Position above object, approach from top
            grasp_pos = [
                object_position[0],
                object_position[1],
                object_position[2] + 0.1  # 10cm above object
            ]
        else:
            # Side grasp
            grasp_pos = [
                object_position[0] + 0.1,  # 10cm in front
                object_position[1],
                object_position[2] + object_position[2] / 2  # Mid-height
            ]

        # Default orientation (looking down for top grasp)
        grasp_orientation = [0, 0, 0, 1]  # Identity quaternion

        return grasp_pos + grasp_orientation

    def _validate_trajectory(self, trajectory):
        """Validate trajectory for safety and feasibility"""
        # Check for collisions
        for point in trajectory:
            if 'collision_risk' in point and point['collision_risk'] > 0.1:
                return False

        # Check for joint limits
        for point in trajectory:
            if 'joint_angles' in point:
                for angle in point['joint_angles']:
                    if abs(angle) > 3.0:  # Beyond reasonable joint limits
                        return False

        return True

    def _execute_trajectory(self, trajectory):
        """Execute validated trajectory"""
        # In practice, this would send commands to robot controllers
        execution_result = {
            'status': 'success',
            'trajectory_executed': len(trajectory),
            'execution_time': len(trajectory) * 0.1  # 0.1s per waypoint
        }
        return execution_result

    def _handle_safety_error(self, decision):
        """Handle safety errors in decision making"""
        return {
            'status': 'safety_error',
            'message': 'Decision contains safety violations',
            'suggested_action': 'Request human intervention'
        }

    def _handle_validation_failure(self, trajectory):
        """Handle trajectory validation failure"""
        return {
            'status': 'validation_failed',
            'message': 'Trajectory validation failed',
            'suggested_action': 'Plan alternative trajectory'
        }
```

## Integration with ROS 2

### ROS 2 Motion Planning Interfaces

ROS 2 provides standardized interfaces for motion planning and action execution:

```python
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import time

class HumanoidMotionPlannerNode(Node):
    def __init__(self):
        super().__init__('humanoid_motion_planner_node')

        # Publishers and subscribers
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/target_pose',
            self.target_pose_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/motion_commands',
            self.command_callback,
            10
        )

        # Action clients
        self.trajectory_client = self.create_client(
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # Initialize motion planner
        self.motion_planner = HumanoidMotionPlanner(robot_model=None)
        self.current_joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint'
        ]

    def target_pose_callback(self, msg):
        """Handle target pose requests"""
        target_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]

        # Plan motion to target pose
        try:
            trajectory = self.motion_planner.plan_arm_motion(
                start_pose=self.get_current_pose(),
                goal_pose=target_pose
            )

            # Convert to ROS 2 trajectory message
            ros_trajectory = self._convert_to_ros_trajectory(trajectory)

            # Publish trajectory
            self.trajectory_pub.publish(ros_trajectory)

        except Exception as e:
            self.get_logger().error(f'Motion planning failed: {e}')

    def command_callback(self, msg):
        """Handle motion commands"""
        command = msg.data

        if command.startswith('move_to:'):
            # Parse target position
            try:
                parts = command.split(':')[1].split(',')
                target_pos = [float(x.strip()) for x in parts[:3]]

                # Plan walking trajectory
                walking_trajectory = self.motion_planner.plan_walking_trajectory(
                    start_pose=self.get_current_pose(),
                    goal_position=target_pos
                )

                # Execute walking trajectory
                self._execute_walking_trajectory(walking_trajectory)

            except ValueError:
                self.get_logger().error('Invalid move_to command format')

    def _convert_to_ros_trajectory(self, trajectory):
        """Convert internal trajectory to ROS 2 JointTrajectory message"""
        msg = JointTrajectory()
        msg.joint_names = self.current_joint_names

        for point in trajectory:
            ros_point = JointTrajectoryPoint()
            ros_point.positions = point['joint_angles']
            ros_point.time_from_start = Duration(sec=int(point['time']), nanosec=0)

            # Add velocity and acceleration if available
            if 'velocities' in point:
                ros_point.velocities = point['velocities']
            if 'accelerations' in point:
                ros_point.accelerations = point['accelerations']

            msg.points.append(ros_point)

        return msg

    def _execute_walking_trajectory(self, walking_trajectory):
        """Execute walking trajectory on humanoid robot"""
        for step in walking_trajectory:
            # Move to step position
            self._move_to_position(step['position'], step['time'])

            # Wait for step completion
            time.sleep(0.1)  # Simulated wait

    def get_current_pose(self):
        """Get current robot pose (placeholder implementation)"""
        # In practice, this would get current pose from robot state
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

# Example usage function
def main():
    rclpy.init()
    node = HumanoidMotionPlannerNode()

    # Spin the node
    rclpy.spin(node)

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
```

## Practical Considerations

### Performance Optimization

Motion planning for humanoid robots requires careful attention to computational efficiency:

**Real-Time Constraints**:
- Motion planning must complete within robot control loop timing
- Use sampling-based methods for complex scenarios
- Implement hierarchical planning for efficiency

**Memory Management**:
- Efficient data structures for trajectory storage
- Streaming of trajectory data to controllers
- Proper cleanup of intermediate planning data

**Hardware Acceleration**:
- Utilize GPU acceleration for complex kinematic calculations
- Implement parallel processing where possible
- Optimize for specific robot hardware capabilities

### Safety and Validation

Safety remains paramount in action grounding systems:

**Pre-Execution Validation**:
- Verify all trajectories for collision-free paths
- Check joint limits and velocity constraints
- Validate stability throughout motion sequences

**Runtime Monitoring**:
- Monitor execution for deviations from planned trajectory
- Implement emergency stops for unexpected conditions
- Continuously validate safety constraints during execution

**Fallback Mechanisms**:
- Maintain safe home positions for all limbs
- Implement graceful degradation for failed motions
- Provide manual override capabilities

## Summary

In this lesson, you've learned about action grounding and motion planning for humanoid robots, including:

- The critical role of action grounding in connecting AI decisions to physical movements
- The action grounding pipeline from high-level goals to motor commands
- Symbol grounding techniques for connecting language to actions
- Motion planning algorithms specifically designed for humanoid robots
- Implementation of humanoid-specific motion planning challenges
- Integration of action grounding with AI decision-making systems
- ROS 2 interfaces for motion planning and execution
- Safety considerations and validation requirements

Action grounding systems form the essential bridge between cognitive decision-making and physical robot behavior. In the next lesson, you'll learn how to implement comprehensive safety constraints and validation systems to ensure these AI-driven behaviors operate safely in human environments.