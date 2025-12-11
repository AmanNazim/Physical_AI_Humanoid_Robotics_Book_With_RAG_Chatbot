---
title: Lesson 2.3 – Depth Camera and IMU Simulation
sidebar_position: 5
---

# Lesson 2.3 – Depth Camera and IMU Simulation

## Learning Objectives

By the end of this lesson, you will be able to:

- Implement depth cameras in Gazebo simulation environment with realistic depth image generation
- Simulate IMU sensors for orientation sensing capabilities in humanoid robotics
- Integrate depth camera and IMU data for sensor fusion in humanoid robotics applications
- Process multiple sensor types using ROS 2 communication patterns
- Validate depth camera and IMU performance in simulation
- Create sensor fusion algorithms that combine depth and IMU data

## Introduction to Depth Camera and IMU Simulation

In this lesson, we'll complete the sensor suite for your humanoid robot by implementing depth cameras and IMU sensors. These sensors provide complementary information:

- **Depth Cameras**: Generate 3D spatial information from visual data, enabling object recognition, scene understanding, and 3D mapping
- **IMU (Inertial Measurement Unit)**: Track orientation, acceleration, and angular velocity, providing crucial motion and stability information

Together with the LiDAR sensors from the previous lesson, these sensors form a comprehensive perception system for your humanoid robot.

## Depth Camera Simulation in Gazebo

### Understanding Depth Cameras

Depth cameras provide both color (RGB) and depth information for each pixel, creating rich 3D representations of the environment. In simulation, depth cameras must accurately model:

- **Intrinsic parameters**: Focal length, principal point, distortion coefficients
- **Extrinsic parameters**: Position and orientation relative to the robot
- **Depth accuracy**: Noise characteristics and measurement precision
- **Field of view**: Horizontal and vertical viewing angles

### Configuring Depth Cameras in Gazebo

Let's add a depth camera to your humanoid robot model:

```xml
<!-- Depth camera link -->
<link name="depth_camera_link">
  <inertial>
    <mass value="0.1" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001" />
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.02 0.04 0.02" />
    </geometry>
    <material name="black">
      <color rgba="0.1 0.1 0.1 1" />
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.02 0.04 0.02" />
    </geometry>
  </collision>
</link>

<!-- Joint to connect depth camera to robot -->
<joint name="depth_camera_joint" type="fixed">
  <parent link="base_link" />
  <child link="depth_camera_link" />
  <origin xyz="0.05 0 1.0" rpy="0 0 0" />  <!-- Positioned on robot's head -->
</joint>

<!-- Gazebo plugin for depth camera -->
<gazebo reference="depth_camera_link">
  <sensor name="depth_camera" type="depth">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="head">
      <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees in radians -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>rgb/image_raw:=rgb/image_raw</remapping>
        <remapping>rgb/camera_info:=rgb/camera_info</remapping>
        <remapping>depth/image_raw:=depth/image_raw</remapping>
        <remapping>depth/camera_info:=depth/camera_info</remapping>
      </ros>
      <frame_name>depth_camera_link</frame_name>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

### Advanced Depth Camera Configuration

For more sophisticated applications, you might want to configure stereo cameras or higher-resolution depth sensors:

```xml
<!-- High-resolution depth camera -->
<gazebo reference="high_res_depth_camera_link">
  <sensor name="high_res_depth_camera" type="depth">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>15</update_rate>
    <camera name="high_res">
      <horizontal_fov>1.396</horizontal_fov>  <!-- 80 degrees -->
      <image>
        <width>1280</width>  <!-- Higher resolution -->
        <height>720</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.05</near>    <!-- Closer minimum range -->
        <far>15.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.005</stddev>  <!-- Lower noise for high-res -->
      </noise>
    </camera>
    <plugin name="high_res_depth_controller" filename="libgazebo_ros_openni_kinect.so">
      <ros>
        <namespace>/humanoid_robot/high_res</namespace>
        <remapping>rgb/image_raw:=rgb/image_raw</remapping>
        <remapping>rgb/camera_info:=rgb/camera_info</remapping>
        <remapping>depth/image_raw:=depth/image_raw</remapping>
        <remapping>depth/camera_info:=depth/camera_info</remapping>
      </ros>
      <frame_name>high_res_depth_camera_link</frame_name>
      <point_cloud_cutoff>0.1</point_cloud_cutoff>
      <point_cloud_cutoff_max>10.0</point_cloud_cutoff_max>
    </plugin>
  </sensor>
</gazebo>
```

## IMU Sensor Simulation

### Understanding IMU Sensors

An IMU typically combines multiple sensors:
- **3-axis Accelerometer**: Measures linear acceleration
- **3-axis Gyroscope**: Measures angular velocity
- **3-axis Magnetometer**: Measures magnetic field (for heading)

In humanoid robotics, IMUs are crucial for:
- Balance and stability control
- Motion tracking
- Orientation estimation
- Fall detection

### Configuring IMU Sensors in Gazebo

Let's add an IMU to your humanoid robot:

```xml
<!-- IMU link (typically placed at robot's center of mass) -->
<link name="imu_link">
  <inertial>
    <mass value="0.01" />
    <origin xyz="0 0 0" rpy="0 0 0" />
    <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6" />
  </inertial>

  <visual>
    <origin xyz="0 0 0" rpy="0 0 0" />
    <geometry>
      <box size="0.01 0.01 0.01" />
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1" />
    </material>
  </visual>
</link>

<!-- Joint to connect IMU to robot's body -->
<joint name="imu_joint" type="fixed">
  <parent link="base_link" />
  <child link="imu_link" />
  <origin xyz="0 0 0.5" rpy="0 0 0" />  <!-- At robot's center of mass -->
</joint>

<!-- Gazebo plugin for IMU -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <pose>0 0 0 0 0 0</pose>
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>  <!-- ~0.1 deg/s (gyro noise) -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.0017</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>   <!-- ~0.017 m/s² (accel noise) -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.017</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>base_link</body_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Advanced IMU Configuration

For more realistic IMU simulation, you can add bias and drift characteristics:

```xml
<!-- Advanced IMU with bias and drift -->
<gazebo reference="advanced_imu_link">
  <sensor name="advanced_imu_sensor" type="imu">
    <pose>0 0 0 0 0 0</pose>
    <always_on>true</always_on>
    <update_rate>200</update_rate>
    <visualize>false</visualize>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.0001</bias_mean>    <!-- Bias -->
            <bias_stddev>0.00001</bias_stddev>  <!-- Bias drift -->
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.0001</bias_mean>
            <bias_stddev>0.00001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.001</stddev>
            <bias_mean>0.0001</bias_mean>
            <bias_stddev>0.00001</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
            <bias_mean>0.001</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
            <bias_mean>0.001</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>0.01</stddev>
            <bias_mean>0.001</bias_mean>
            <bias_stddev>0.0001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="advanced_imu_controller" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid_robot</namespace>
        <remapping>~/out:=imu/data_raw</remapping>
      </ros>
      <frame_name>advanced_imu_link</frame_name>
      <body_name>base_link</body_name>
      <update_rate>200</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

## Processing Depth Camera and IMU Data

### Depth Camera Data Processing

Here's a Python node to process depth camera data:

```python
#!/usr/bin/env python3
"""
Depth camera processing node for humanoid robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthCameraProcessor(Node):
    def __init__(self):
        super().__init__('depth_camera_processor')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Subscribe to RGB and depth images
        self.rgb_sub = self.create_subscription(
            Image,
            '/humanoid_robot/rgb/image_raw',
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/humanoid_robot/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/humanoid_robot/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for processed depth data
        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/humanoid_robot/point_cloud',
            10
        )

        self.camera_info = None
        self.get_logger().info('Depth Camera Processor initialized')

    def rgb_callback(self, msg):
        """Process RGB image data"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process RGB image (example: edge detection)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # You can add more sophisticated image processing here
            self.get_logger().debug(f'Processed RGB image: {cv_image.shape}')

        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def depth_callback(self, msg):
        """Process depth image data"""
        try:
            # Convert depth image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Convert to float32 for processing
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32) / 1000.0  # Convert mm to meters

            # Process depth data
            valid_depths = depth_image[depth_image > 0]

            if len(valid_depths) > 0:
                avg_depth = np.mean(valid_depths)
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)

                self.get_logger().debug(f'Depth stats - Avg: {avg_depth:.2f}m, Min: {min_depth:.2f}m, Max: {max_depth:.2f}m')

            # Generate point cloud if camera info is available
            if self.camera_info is not None:
                point_cloud = self.generate_point_cloud(depth_image, self.camera_info)
                if point_cloud is not None:
                    self.point_cloud_pub.publish(point_cloud)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def camera_info_callback(self, msg):
        """Store camera calibration information"""
        self.camera_info = msg

    def generate_point_cloud(self, depth_image, camera_info):
        """Generate point cloud from depth image and camera info"""
        try:
            # Extract camera parameters
            fx = camera_info.k[0]  # Focal length x
            fy = camera_info.k[4]  # Focal length y
            cx = camera_info.k[2]  # Principal point x
            cy = camera_info.k[5]  # Principal point y

            height, width = depth_image.shape
            points = []

            # Generate 3D points from depth image
            for v in range(height):
                for u in range(width):
                    depth = depth_image[v, u]

                    # Skip invalid depth values
                    if depth <= 0 or depth > 10.0:  # Max range 10m
                        continue

                    # Convert pixel coordinates to 3D world coordinates
                    x = (u - cx) * depth / fx
                    y = (v - cy) * depth / fy
                    z = depth

                    points.append([x, y, z])

            if points:
                # Create PointCloud2 message
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = 'depth_camera_link'

                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
                ]

                point_cloud = point_cloud2.create_cloud(header, fields, points)
                return point_cloud

            return None

        except Exception as e:
            self.get_logger().error(f'Error generating point cloud: {e}')
            return None

def main(args=None):
    rclpy.init(args=args)
    processor = DepthCameraProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### IMU Data Processing

Here's a Python node to process IMU data:

```python
#!/usr/bin/env python3
"""
IMU processing node for humanoid robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial.transform import Rotation as R

class IMUProcessor(Node):
    def __init__(self):
        super().__init__('imu_processor')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid_robot/imu/data',
            self.imu_callback,
            10
        )

        # Publishers for processed data
        self.orientation_pub = self.create_publisher(
            Quaternion,
            '/humanoid_robot/imu/orientation',
            10
        )

        self.angular_velocity_pub = self.create_publisher(
            Vector3,
            '/humanoid_robot/imu/angular_velocity',
            10
        )

        self.linear_acceleration_pub = self.create_publisher(
            Vector3,
            '/humanoid_robot/imu/linear_acceleration',
            10
        )

        # Publishers for derived metrics
        self.roll_pitch_yaw_pub = self.create_publisher(
            Vector3,
            '/humanoid_robot/imu/rpy',
            10
        )

        self.balance_score_pub = self.create_publisher(
            Float64,
            '/humanoid_robot/imu/balance_score',
            10
        )

        # Store previous values for filtering
        self.prev_orientation = None
        self.prev_time = None

        self.get_logger().info('IMU Processor initialized')

    def imu_callback(self, msg):
        """Process incoming IMU data"""
        try:
            # Extract orientation (as quaternion)
            orientation = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ])

            # Extract angular velocity
            angular_velocity = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            # Extract linear acceleration
            linear_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Publish raw data
            self.orientation_pub.publish(msg.orientation)
            self.angular_velocity_pub.publish(msg.angular_velocity)
            self.linear_acceleration_pub.publish(msg.linear_acceleration)

            # Convert quaternion to roll-pitch-yaw
            rpy = self.quaternion_to_rpy(orientation)
            rpy_msg = Vector3()
            rpy_msg.x = rpy[0]  # Roll
            rpy_msg.y = rpy[1]  # Pitch
            rpy_msg.z = rpy[2]  # Yaw
            self.roll_pitch_yaw_pub.publish(rpy_msg)

            # Calculate balance score based on orientation
            balance_score = self.calculate_balance_score(rpy, linear_accel)
            balance_msg = Float64()
            balance_msg.data = balance_score
            self.balance_score_pub.publish(balance_msg)

            # Log important values
            self.get_logger().debug(f'Roll: {np.degrees(rpy[0]):.2f}°, Pitch: {np.degrees(rpy[1]):.2f}°, Yaw: {np.degrees(rpy[2]):.2f}°')
            self.get_logger().debug(f'Balance Score: {balance_score:.3f}')

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def quaternion_to_rpy(self, quat):
        """Convert quaternion to roll-pitch-yaw angles"""
        try:
            # Create rotation object from quaternion
            rotation = R.from_quat(quat)
            # Convert to Euler angles (roll, pitch, yaw)
            rpy = rotation.as_euler('xyz')
            return rpy
        except Exception:
            # Fallback calculation
            w, x, y, z = quat
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (w * y - z * x)
            pitch = np.arcsin(sinp)

            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)

            return np.array([roll, pitch, yaw])

    def calculate_balance_score(self, rpy, linear_accel):
        """Calculate a balance score based on orientation and acceleration"""
        # Simple balance score: closer to upright position = higher score
        # Roll and pitch should be close to 0 for good balance
        roll_magnitude = abs(rpy[0])
        pitch_magnitude = abs(rpy[1])

        # Penalize large angular deviations
        orientation_penalty = (roll_magnitude + pitch_magnitude) / 2.0

        # Consider linear acceleration for dynamic balance
        accel_magnitude = np.linalg.norm(linear_accel)

        # Normalize to 0-1 scale (0 = perfectly balanced, 1 = falling)
        max_angle = np.pi / 3  # 60 degrees before considered falling
        balance_score = min(1.0, orientation_penalty / max_angle)

        # Invert so higher score = better balance
        return 1.0 - balance_score

def main(args=None):
    rclpy.init(args=args)
    processor = IMUProcessor()

    try:
        rclpy.spin(processor)
    except KeyboardInterrupt:
        pass
    finally:
        processor.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion: Combining Depth Camera and IMU Data

### Understanding Sensor Fusion

Sensor fusion combines data from multiple sensors to create a more accurate and reliable representation of the environment. For humanoid robots, combining depth camera and IMU data provides:

- **Stabilized depth data**: IMU orientation helps correct depth measurements during robot movement
- **Enhanced pose estimation**: Visual features combined with IMU motion tracking
- **Improved navigation**: More robust obstacle detection and avoidance

### Implementing Basic Sensor Fusion

Here's a sensor fusion node that combines depth camera and IMU data:

```python
#!/usr/bin/env python3
"""
Sensor fusion node for combining depth camera and IMU data
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to IMU and depth camera data
        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid_robot/imu/data',
            self.imu_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/humanoid_robot/depth/image_raw',
            self.depth_callback,
            10
        )

        # Publishers for fused data
        self.fused_pose_pub = self.create_publisher(
            PoseStamped,
            '/humanoid_robot/fused_pose',
            10
        )

        self.stabilized_depth_pub = self.create_publisher(
            Image,
            '/humanoid_robot/stabilized_depth',
            10
        )

        self.motion_compensated_pub = self.create_publisher(
            Float64,
            '/humanoid_robot/motion_compensation_factor',
            10
        )

        # Store sensor data
        self.current_imu = None
        self.current_depth = None
        self.imu_timestamp = None
        self.depth_timestamp = None

        self.get_logger().info('Sensor Fusion node initialized')

    def imu_callback(self, msg):
        """Store IMU data"""
        self.current_imu = msg
        self.imu_timestamp = msg.header.stamp

    def depth_callback(self, msg):
        """Process depth image with IMU compensation"""
        try:
            self.current_depth = msg
            self.depth_timestamp = msg.header.stamp

            # If we have both IMU and depth data, perform fusion
            if self.current_imu is not None:
                self.perform_sensor_fusion()

        except Exception as e:
            self.get_logger().error(f'Error in depth callback: {e}')

    def perform_sensor_fusion(self):
        """Perform sensor fusion between IMU and depth camera"""
        try:
            # Extract IMU orientation
            imu_quat = np.array([
                self.current_imu.orientation.x,
                self.current_imu.orientation.y,
                self.current_imu.orientation.z,
                self.current_imu.orientation.w
            ])

            # Convert to rotation matrix
            rotation = R.from_quat(imu_quat)
            rotation_matrix = rotation.as_matrix()

            # Extract angular velocity for motion compensation
            angular_vel = np.array([
                self.current_imu.angular_velocity.x,
                self.current_imu.angular_velocity.y,
                self.current_imu.angular_velocity.z
            ])

            # Calculate motion compensation factor
            motion_compensation = np.linalg.norm(angular_vel)

            # Create fused pose message
            fused_pose = PoseStamped()
            fused_pose.header.stamp = self.get_clock().now().to_msg()
            fused_pose.header.frame_id = 'odom'

            # Use IMU orientation as primary orientation source
            fused_pose.pose.orientation = self.current_imu.orientation

            # For position, we might integrate from other sources or use zero
            # In a real implementation, you'd combine with other sensors like wheel encoders
            fused_pose.pose.position.x = 0.0
            fused_pose.pose.position.y = 0.0
            fused_pose.pose.position.z = 0.0

            # Publish fused pose
            self.fused_pose_pub.publish(fused_pose)

            # Publish motion compensation factor
            motion_msg = Float64()
            motion_msg.data = motion_compensation
            self.motion_compensated_pub.publish(motion_msg)

            # Convert depth image for potential stabilization
            try:
                depth_cv = self.bridge.imgmsg_to_cv2(self.current_depth, desired_encoding='passthrough')

                # Apply motion compensation if needed
                if motion_compensation > 0.1:  # Threshold for significant motion
                    # In a real implementation, you would use the IMU data
                    # to compensate for motion blur in the depth image
                    self.get_logger().debug(f'Applying motion compensation: {motion_compensation:.3f}')

                # Publish the (potentially) stabilized depth image
                # For now, just republish the original
                self.stabilized_depth_pub.publish(self.current_depth)

            except Exception as e:
                self.get_logger().warn(f'Error processing depth image: {e}')

        except Exception as e:
            self.get_logger().error(f'Error in sensor fusion: {e}')

    def calculate_pose_from_imu(self, imu_msg):
        """Calculate pose estimate from IMU data"""
        # This is a simplified approach
        # In practice, you'd use sensor fusion algorithms like Kalman filters
        pose = PoseStamped()
        pose.header = imu_msg.header
        pose.pose.orientation = imu_msg.orientation

        # Position would typically come from integration of acceleration
        # or from other sensors
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0

        return pose

def main(args=None):
    rclpy.init(args=args)
    fusion_node = SensorFusion()

    try:
        rclpy.spin(fusion_node)
    except KeyboardInterrupt:
        pass
    finally:
        fusion_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Sensor Fusion with Kalman Filtering

For more sophisticated sensor fusion, you might implement a Kalman filter:

```python
#!/usr/bin/env python3
"""
Kalman filter-based sensor fusion for humanoid robot
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import numpy as np

class KalmanFilterFusion(Node):
    def __init__(self):
        super().__init__('kalman_fusion')

        # Subscribe to IMU data
        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid_robot/imu/data',
            self.imu_callback,
            10
        )

        # Initialize Kalman filter parameters
        self.initialize_kalman_filter()

        self.get_logger().info('Kalman Filter Fusion initialized')

    def initialize_kalman_filter(self):
        """Initialize Kalman filter state and parameters"""
        # State vector: [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.state = np.zeros(6)

        # State covariance matrix
        self.P = np.eye(6) * 0.1

        # Process noise covariance
        self.Q = np.eye(6) * 0.01

        # Measurement noise covariance
        self.R = np.eye(3) * 0.1  # For orientation measurements

        # Control input matrix (not used in this example)
        self.B = np.zeros((6, 0))

        # Measurement matrix
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])  # Observe orientation only

    def imu_callback(self, msg):
        """Process IMU data through Kalman filter"""
        try:
            # Extract measurement (orientation from IMU)
            measurement = np.array([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            ])

            # Extract control input (angular velocities)
            control_input = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            # Prediction step
            self.predict(control_input)

            # Update step
            self.update(measurement)

            # Publish filtered state
            self.publish_filtered_state()

        except Exception as e:
            self.get_logger().error(f'Kalman filter error: {e}')

    def predict(self, control_input):
        """Prediction step of Kalman filter"""
        # State transition model (simplified)
        dt = 0.01  # Assuming 100Hz update rate

        # Update state based on angular velocities
        self.state[3:6] = control_input  # Angular velocities
        self.state[0:3] += self.state[3:6] * dt  # Integrate to get orientation

        # State transition matrix (simplified)
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update step of Kalman filter"""
        # Calculate innovation
        innovation = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance
        I = np.eye(len(self.state))
        self.P = (I - K @ self.H) @ self.P

    def publish_filtered_state(self):
        """Publish the filtered state"""
        # In a real implementation, you would publish the filtered orientation
        # and other state variables to appropriate topics
        roll, pitch, yaw = self.state[0:3]
        self.get_logger().debug(f'Filtered RPY: [{np.degrees(roll):.2f}, {np.degrees(pitch):.2f}, {np.degrees(yaw):.2f}]')

def main(args=None):
    rclpy.init(args=args)
    kalman_node = KalmanFilterFusion()

    try:
        rclpy.spin(kalman_node)
    except KeyboardInterrupt:
        pass
    finally:
        kalman_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Validation and Testing

### Testing Depth Camera Performance

Create a validation script for your depth camera:

```python
#!/usr/bin/env python3
"""
Depth camera validation script
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np

class DepthCameraValidator(Node):
    def __init__(self):
        super().__init__('depth_camera_validator')

        self.bridge = CvBridge()

        self.depth_sub = self.create_subscription(
            Image,
            '/humanoid_robot/depth/image_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/humanoid_robot/rgb/camera_info',
            self.camera_info_callback,
            10
        )

        self.camera_info = None
        self.frame_count = 0
        self.timer = self.create_timer(5.0, self.report_status)

        self.get_logger().info('Depth Camera Validator started')

    def depth_callback(self, msg):
        """Validate depth image data"""
        try:
            self.frame_count += 1

            # Convert to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Validate depth values
            valid_depths = depth_image[depth_image > 0]

            if len(valid_depths) == 0:
                self.get_logger().warn('No valid depth values in frame')
                return

            # Check depth range
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)

            if min_depth < 0.05:  # Too close
                self.get_logger().warn(f'Depth too close: {min_depth:.3f}m')

            if max_depth > 10.0:  # Beyond expected range
                self.get_logger().warn(f'Depth beyond expected range: {max_depth:.3f}m')

            # Check for NaN or Inf values
            if np.any(np.isnan(depth_image)) or np.any(np.isinf(depth_image)):
                self.get_logger().warn('Depth image contains NaN or Inf values')

            # Validate image dimensions
            expected_width = 640 if self.camera_info is None else self.camera_info.width
            expected_height = 480 if self.camera_info is None else self.camera_info.height

            if depth_image.shape[1] != expected_width or depth_image.shape[0] != expected_height:
                self.get_logger().warn(f'Unexpected image dimensions: {depth_image.shape}')

        except Exception as e:
            self.get_logger().error(f'Error validating depth image: {e}')

    def camera_info_callback(self, msg):
        """Store camera information"""
        self.camera_info = msg

    def report_status(self):
        """Report validation status"""
        self.get_logger().info(f'Depth Camera Validator: Processed {self.frame_count} frames')

def main(args=None):
    rclpy.init(args=args)
    validator = DepthCameraValidator()

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

### Testing IMU Performance

Create a validation script for your IMU:

```python
#!/usr/bin/env python3
"""
IMU validation script
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import numpy as np

class IMUValidator(Node):
    def __init__(self):
        super().__init__('imu_validator')

        self.imu_sub = self.create_subscription(
            Imu,
            '/humanoid_robot/imu/data',
            self.imu_callback,
            10
        )

        self.reading_count = 0
        self.angular_velocity_buffer = []
        self.linear_acceleration_buffer = []
        self.timer = self.create_timer(5.0, self.report_status)

        self.get_logger().info('IMU Validator started')

    def imu_callback(self, msg):
        """Validate IMU data"""
        try:
            self.reading_count += 1

            # Extract angular velocity
            ang_vel = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

            # Extract linear acceleration
            lin_acc = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

            # Validate angular velocity (should be reasonable)
            ang_vel_mag = np.linalg.norm(ang_vel)
            if ang_vel_mag > 10.0:  # 10 rad/s is quite fast
                self.get_logger().warn(f'High angular velocity: {ang_vel_mag:.3f} rad/s')

            # Validate linear acceleration (should be around 9.8 m/s² when stationary)
            lin_acc_mag = np.linalg.norm(lin_acc)
            if lin_acc_mag < 5.0 or lin_acc_mag > 15.0:
                self.get_logger().warn(f'Unexpected acceleration magnitude: {lin_acc_mag:.3f} m/s²')

            # Validate quaternion normalization
            quat_norm = np.sqrt(
                msg.orientation.x**2 +
                msg.orientation.y**2 +
                msg.orientation.z**2 +
                msg.orientation.w**2
            )

            if abs(quat_norm - 1.0) > 0.01:
                self.get_logger().warn(f'Quaternion not normalized: {quat_norm:.6f}')

            # Store values for statistics
            self.angular_velocity_buffer.append(ang_vel_mag)
            self.linear_acceleration_buffer.append(lin_acc_mag)

            # Keep buffers at reasonable size
            if len(self.angular_velocity_buffer) > 100:
                self.angular_velocity_buffer.pop(0)
            if len(self.linear_acceleration_buffer) > 100:
                self.linear_acceleration_buffer.pop(0)

        except Exception as e:
            self.get_logger().error(f'Error validating IMU data: {e}')

    def report_status(self):
        """Report validation statistics"""
        if self.angular_velocity_buffer:
            avg_ang_vel = np.mean(self.angular_velocity_buffer)
            avg_lin_acc = np.mean(self.linear_acceleration_buffer)

            self.get_logger().info(
                f'IMU Validator: {self.reading_count} readings | '
                f'Avg ang vel: {avg_ang_vel:.3f} rad/s | '
                f'Avg lin acc: {avg_lin_acc:.3f} m/s²'
            )

def main(args=None):
    rclpy.init(args=args)
    validator = IMUValidator()

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

## Practical Exercise: Complete Sensor Integration

### Step 1: Add All Sensors to Your Robot Model

Combine all the sensor configurations in your robot's URDF:

```xml
<!-- Complete sensor integration for humanoid robot -->
<?xml version="1.0"?>
<robot name="humanoid_robot_with_sensors">

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
  </link>

  <!-- IMU at center of mass -->
  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>
    </inertial>
    <visual>
      <geometry><box size="0.01 0.01 0.01"/></geometry>
      <material name="red"><color rgba="1 0 0 1"/></material>
    </visual>
  </link>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- Depth camera on head -->
  <link name="depth_camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry><box size="0.02 0.04 0.02"/></geometry>
      <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>
    </visual>
  </link>
  <joint name="depth_camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="depth_camera_link"/>
    <origin xyz="0.05 0 0.45" rpy="0 0 0"/>
  </joint>

  <!-- LiDAR on head -->
  <link name="lidar_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0002"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.025" length="0.05"/></geometry>
      <material name="gray"><color rgba="0.5 0.5 0.5 1"/></material>
    </visual>
  </link>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar_link"/>
    <origin xyz="0.0 0 0.475" rpy="0 0 0"/>
  </joint>

  <!-- Include Gazebo plugins -->
  <gazebo reference="imu_link">
    <sensor name="imu_sensor" type="imu">
      <pose>0 0 0 0 0 0</pose>
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity><x><noise type="gaussian"><mean>0.0</mean><stddev>0.0017</stddev></noise></x>
        <y><noise type="gaussian"><mean>0.0</mean><stddev>0.0017</stddev></noise></y>
        <z><noise type="gaussian"><mean>0.0</mean><stddev>0.0017</stddev></noise></z></angular_velocity>
        <linear_acceleration><x><noise type="gaussian"><mean>0.0</mean><stddev>0.017</stddev></noise></x>
        <y><noise type="gaussian"><mean>0.0</mean><stddev>0.017</stddev></noise></y>
        <z><noise type="gaussian"><mean>0.0</mean><stddev>0.017</stddev></noise></z></linear_acceleration>
      </imu>
      <plugin name="imu_controller" filename="libgazebo_ros_imu.so">
        <ros><namespace>/humanoid_robot</namespace><remapping>~/out:=imu/data</remapping></ros>
        <frame_name>imu_link</frame_name>
        <body_name>base_link</body_name>
        <update_rate>100</update_rate>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="depth_camera_link">
    <sensor name="depth_camera" type="depth">
      <pose>0 0 0 0 0 0</pose>
      <visualize>true</visualize>
      <update_rate>30</update_rate>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>
        <image><width>640</width><height>480</height><format>R8G8B8</format></image>
        <clip><near>0.1</near><far>10.0</far></clip>
        <noise><type>gaussian</type><mean>0.0</mean><stddev>0.007</stddev></noise>
      </camera>
      <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
        <ros>
          <namespace>/humanoid_robot</namespace>
          <remapping>rgb/image_raw:=rgb/image_raw</remapping>
          <remapping>rgb/camera_info:=rgb/camera_info</remapping>
          <remapping>depth/image_raw:=depth/image_raw</remapping>
          <remapping>depth/camera_info:=depth/camera_info</remapping>
        </ros>
        <frame_name>depth_camera_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <gazebo reference="lidar_link">
    <sensor name="lidar_sensor" type="ray">
      <pose>0 0 0 0 0 0</pose>
      <visualize>true</visualize>
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal><samples>720</samples><resolution>1</resolution>
          <min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
        </scan>
        <range><min>0.1</min><max>20.0</max><resolution>0.01</resolution></range>
        <noise type="gaussian"><mean>0.0</mean><stddev>0.01</stddev></noise>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
        <ros><namespace>/humanoid_robot</namespace><remapping>~/out:=scan</remapping></ros>
        <output_type>sensor_msgs/LaserScan</output_type>
        <frame_name>lidar_link</frame_name>
      </plugin>
    </sensor>
  </gazebo>

</robot>
```

## Summary

In this lesson, we completed the sensor suite for your humanoid robot by implementing:

- **Depth camera simulation**: Creating realistic depth images with appropriate noise modeling and processing techniques
- **IMU sensor simulation**: Configuring orientation, acceleration, and angular velocity sensing with realistic noise characteristics
- **Sensor fusion**: Combining depth camera and IMU data for enhanced perception capabilities
- **Processing techniques**: Implementing ROS 2 nodes to handle and process multiple sensor types
- **Validation methods**: Testing and validating sensor performance in simulation

The comprehensive sensor system we've created provides your humanoid robot with the ability to perceive its environment through multiple modalities: visual information from depth cameras, spatial awareness from LiDAR (from the previous lesson), and motion/orientation data from IMUs. This multi-sensor approach is essential for robust humanoid robotics applications.

## Next Steps

With the complete physics and sensor simulation system established, you're now ready to move to Module 2 Chapter 3, where we'll integrate Unity's visualization systems. The sensor fusion concepts learned here will be crucial when implementing visualization systems that must accurately represent the sensor data generated in Gazebo.

Before proceeding, ensure your sensor integration is:
1. All sensors properly integrated into your robot model
2. Publishing data on correct ROS 2 topics
3. Generating realistic sensor data with appropriate noise modeling
4. Performing as expected in various environmental conditions
5. Sensor fusion algorithms properly combining multiple sensor inputs