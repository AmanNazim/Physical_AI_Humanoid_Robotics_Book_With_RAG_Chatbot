---
title: Lesson 4.2 – Sensor Data Consistency Across Platforms
sidebar_position: 2
---

# Lesson 4.2 – Sensor Data Consistency Across Platforms

## Learning Objectives

By the end of this lesson, you will be able to:

- Ensure sensor data consistency when using multiple simulators across platforms
- Implement calibration procedures for cross-platform compatibility
- Standardize data formats across Gazebo and Unity platforms
- Validate sensor data consistency between platforms
- Develop techniques for maintaining data integrity across simulation environments
- Create calibration frameworks for multi-simulator environments

## Introduction

Sensor data consistency is a critical challenge in multi-simulator environments. When using both Gazebo and Unity for robotics simulation, the same physical phenomena may be represented differently by each platform's sensor models. This lesson focuses on ensuring that sensor data maintains consistency and accuracy across both simulation platforms, enabling reliable robot behavior validation and cross-platform testing.

Maintaining sensor data consistency is essential for several reasons:

- **Validation Accuracy**: Robots must behave consistently across platforms for reliable validation
- **Data Integrity**: Inconsistent sensor data can lead to incorrect decision-making
- **Cross-Platform Compatibility**: Ensures that algorithms work reliably regardless of the simulation platform
- **Debugging Confidence**: Consistent data enables accurate problem identification

## Understanding Sensor Differences Between Platforms

Before implementing consistency mechanisms, it's important to understand the fundamental differences between Gazebo and Unity sensor implementations:

### Gazebo Sensor Characteristics

Gazebo provides physics-accurate sensor simulation with:

- **Realistic Physics Modeling**: Ray tracing, collision detection, and physical interactions
- **High Fidelity**: Accurate representation of sensor noise, range limitations, and environmental factors
- **ROS Integration**: Native support for ROS sensor message types
- **Configurable Parameters**: Detailed control over sensor properties like noise models, resolution, and range

### Unity Sensor Characteristics

Unity offers visualization-focused sensor simulation with:

- **Visual Fidelity**: High-quality rendering of sensor data, particularly for cameras and LIDAR
- **Performance Optimization**: Efficient rendering pipelines for real-time visualization
- **User Interaction**: Intuitive tools for sensor configuration and visualization
- **Game Engine Features**: Advanced rendering techniques like post-processing effects

### Key Differences

The primary differences that affect sensor data consistency include:

1. **Noise Models**: Different approaches to simulating sensor noise
2. **Resolution**: Different sampling rates and spatial resolution
3. **Update Frequencies**: Different timing for sensor data updates
4. **Coordinate Systems**: Potential differences in reference frames
5. **Data Representation**: Different internal data structures and formats

## Standardization Framework

To ensure sensor data consistency, we need a comprehensive standardization framework:

### Data Format Standardization

```yaml
# sensor_standardization_config.yaml
sensor_standards:
  # LIDAR sensor standardization
  lidar:
    data_type: "sensor_msgs/LaserScan"
    frame_id: "laser_frame"
    angle_min: -3.14159  # -π radians
    angle_max: 3.14159   # π radians
    angle_increment: 0.0174533  # 1 degree in radians
    time_increment: 0.0  # Time between measurements
    scan_time: 0.1  # Time between scans
    range_min: 0.1  # Minimum range (meters)
    range_max: 30.0  # Maximum range (meters)
    update_rate: 10  # Hz

  # Camera sensor standardization
  camera:
    data_type: "sensor_msgs/Image"
    encoding: "rgb8"
    width: 640
    height: 480
    frame_id: "camera_frame"
    update_rate: 30  # Hz
    fov_horizontal: 60  # degrees
    distortion_model: "plumb_bob"
    distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]

  # IMU sensor standardization
  imu:
    data_type: "sensor_msgs/Imu"
    frame_id: "imu_frame"
    update_rate: 100  # Hz
    linear_acceleration_covariance: [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
    angular_velocity_covariance: [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]
    orientation_covariance: [0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01]

  # Joint state standardization
  joint_state:
    data_type: "sensor_msgs/JointState"
    frame_id: "base_link"
    update_rate: 50  # Hz
```

### Coordinate System Standardization

```python
# coordinate_system_standardizer.py
import numpy as np
from scipy.spatial.transform import Rotation as R

class CoordinateSystemStandardizer:
    def __init__(self):
        # Define standard coordinate system (ROS standard: X forward, Y left, Z up)
        self.ros_standard = {
            'forward': np.array([1, 0, 0]),
            'left': np.array([0, 1, 0]),
            'up': np.array([0, 0, 1])
        }

        # Gazebo typically uses the same coordinate system as ROS
        self.gazebo_to_ros = np.eye(4)  # Identity transformation

        # Unity uses different coordinate system (X right, Y up, Z forward)
        self.unity_to_ros = np.array([
            [0, 0, 1, 0],   # Unity Z becomes ROS X
            [-1, 0, 0, 0],  # Unity -X becomes ROS Y
            [0, -1, 0, 0],  # Unity -Y becomes ROS Z
            [0, 0, 0, 1]
        ])

        # Inverse transformations
        self.ros_to_unity = np.linalg.inv(self.unity_to_ros)

    def transform_position(self, position, from_system, to_system):
        """Transform position vector from one coordinate system to another"""
        if from_system == to_system:
            return position

        # Convert to homogeneous coordinates
        pos_homo = np.append(position, 1)

        if from_system == 'unity' and to_system == 'ros':
            transformed = self.unity_to_ros @ pos_homo
        elif from_system == 'ros' and to_system == 'unity':
            transformed = self.ros_to_unity @ pos_homo
        elif from_system == 'gazebo' and to_system == 'ros':
            transformed = self.gazebo_to_ros @ pos_homo
        else:
            raise ValueError(f"Unsupported transformation: {from_system} to {to_system}")

        return transformed[:3]  # Return 3D coordinates

    def transform_orientation(self, orientation, from_system, to_system):
        """Transform orientation quaternion from one coordinate system to another"""
        if from_system == to_system:
            return orientation

        # Convert quaternion to rotation matrix
        r = R.from_quat(orientation)

        if from_system == 'unity' and to_system == 'ros':
            # Apply coordinate transformation
            unity_rotation_matrix = r.as_matrix()
            ros_rotation_matrix = self.unity_to_ros[:3, :3] @ unity_rotation_matrix @ np.linalg.inv(self.unity_to_ros[:3, :3])
            transformed_r = R.from_matrix(ros_rotation_matrix)
        elif from_system == 'ros' and to_system == 'unity':
            ros_rotation_matrix = r.as_matrix()
            unity_rotation_matrix = self.ros_to_unity[:3, :3] @ ros_rotation_matrix @ np.linalg.inv(self.ros_to_unity[:3, :3])
            transformed_r = R.from_matrix(unity_rotation_matrix)
        elif from_system == 'gazebo' and to_system == 'ros':
            # Gazebo and ROS use the same coordinate system
            return orientation
        else:
            raise ValueError(f"Unsupported transformation: {from_system} to {to_system}")

        return transformed_r.as_quat()

    def standardize_lidar_data(self, raw_data, source_platform):
        """Standardize LIDAR data from different platforms"""
        standardized = {
            'ranges': [],
            'intensities': [],
            'angle_min': -np.pi,
            'angle_max': np.pi,
            'angle_increment': 0.0174533,  # 1 degree
            'time_increment': 0.0,
            'scan_time': 0.1,
            'range_min': 0.1,
            'range_max': 30.0,
            'header': {'frame_id': 'laser_frame', 'timestamp': 0}
        }

        # Normalize data based on source platform
        if source_platform == 'gazebo':
            # Gazebo LIDAR data typically comes in a specific format
            standardized['ranges'] = self.normalize_gazebo_lidar_ranges(raw_data)
        elif source_platform == 'unity':
            # Unity LIDAR data may need different processing
            standardized['ranges'] = self.normalize_unity_lidar_ranges(raw_data)

        return standardized

    def normalize_gazebo_lidar_ranges(self, raw_data):
        """Normalize Gazebo LIDAR ranges to standard format"""
        # Implementation depends on Gazebo LIDAR output format
        normalized = []
        for value in raw_data:
            if value < 0.1:  # Below minimum range
                normalized.append(0.0)  # Invalid range
            elif value > 30.0:  # Beyond maximum range
                normalized.append(30.1)  # Maximum range + 1
            else:
                normalized.append(value)
        return normalized

    def normalize_unity_lidar_ranges(self, raw_data):
        """Normalize Unity LIDAR ranges to standard format"""
        # Unity might use different units or format
        normalized = []
        for value in raw_data:
            # Convert Unity units to meters if needed
            meters_value = value * 1.0  # Unity units to meters conversion factor
            if meters_value < 0.1:  # Below minimum range
                normalized.append(0.0)  # Invalid range
            elif meters_value > 30.0:  # Beyond maximum range
                normalized.append(30.1)  # Maximum range + 1
            else:
                normalized.append(meters_value)
        return normalized
```

## Calibration Procedures

Calibration ensures that sensor data from different platforms can be meaningfully compared:

### Cross-Platform Calibration Framework

```python
# cross_platform_calibrator.py
import numpy as np
from scipy.optimize import minimize
import json

class CrossPlatformCalibrator:
    def __init__(self):
        self.calibration_data = {}
        self.calibration_parameters = {}
        self.calibration_complete = False

    def collect_calibration_data(self, gazebo_data, unity_data, reference_data=None):
        """Collect synchronized data from both platforms for calibration"""
        # Ensure data is synchronized in time
        if len(gazebo_data) != len(unity_data):
            raise ValueError("Gazebo and Unity data must have the same length for calibration")

        self.calibration_data = {
            'gazebo': gazebo_data,
            'unity': unity_data,
            'reference': reference_data  # Optional ground truth
        }

    def calibrate_lidar(self):
        """Calibrate LIDAR sensor data between platforms"""
        if 'gazebo' not in self.calibration_data or 'unity' not in self.calibration_data:
            raise ValueError("Calibration data not available")

        gazebo_lidar = self.calibration_data['gazebo']['lidar']
        unity_lidar = self.calibration_data['unity']['lidar']

        # Calculate scaling and offset factors
        # This is a simplified example - real calibration would be more complex
        differences = []
        for i in range(min(len(gazebo_lidar), len(unity_lidar))):
            for j in range(len(gazebo_lidar[i]['ranges'])):
                if gazebo_lidar[i]['ranges'][j] > 0 and unity_lidar[i]['ranges'][j] > 0:
                    differences.append(gazebo_lidar[i]['ranges'][j] - unity_lidar[i]['ranges'][j])

        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)

            self.calibration_parameters['lidar'] = {
                'offset': float(mean_diff),
                'scale_factor': 1.0,  # For now, assume no scaling needed
                'std_deviation': float(std_diff)
            }

        print(f"LIDAR calibration complete: offset={mean_diff:.4f}, std={std_diff:.4f}")

    def calibrate_camera(self):
        """Calibrate camera sensor data between platforms"""
        # Camera calibration would involve intrinsic and extrinsic parameters
        # This is a simplified example
        self.calibration_parameters['camera'] = {
            'focal_length_factor': 1.0,
            'distortion_params': [0.0, 0.0, 0.0, 0.0, 0.0],
            'pixel_mapping_error': 0.0
        }

    def calibrate_imu(self):
        """Calibrate IMU sensor data between platforms"""
        gazebo_imu = self.calibration_data['gazebo']['imu']
        unity_imu = self.calibration_data['unity']['imu']

        # Calculate bias and scale factors for IMU data
        linear_acc_bias = np.mean([
            np.array(g['linear_acceleration']) - np.array(u['linear_acceleration'])
            for g, u in zip(gazebo_imu, unity_imu)
        ], axis=0)

        angular_vel_bias = np.mean([
            np.array(g['angular_velocity']) - np.array(u['angular_velocity'])
            for g, u in zip(gazebo_imu, unity_imu)
        ], axis=0)

        self.calibration_parameters['imu'] = {
            'linear_acceleration_bias': linear_acc_bias.tolist(),
            'angular_velocity_bias': angular_vel_bias.tolist(),
            'scale_factors': [1.0, 1.0, 1.0]  # No scaling for now
        }

    def apply_calibration(self, raw_data, platform, sensor_type):
        """Apply calibration parameters to raw sensor data"""
        if not self.calibration_complete:
            print("Warning: Applying calibration before full calibration process")

        if sensor_type not in self.calibration_parameters:
            return raw_data  # Return raw data if no calibration available

        calibrated_data = raw_data.copy()
        cal_params = self.calibration_parameters[sensor_type]

        if sensor_type == 'lidar':
            # Apply offset correction
            offset = cal_params.get('offset', 0.0)
            calibrated_data['ranges'] = [r + offset if r > 0 else r for r in raw_data['ranges']]

        elif sensor_type == 'imu':
            # Apply bias correction
            if 'linear_acceleration' in raw_data:
                bias = np.array(cal_params.get('linear_acceleration_bias', [0, 0, 0]))
                raw_acc = np.array(raw_data['linear_acceleration'])
                calibrated_data['linear_acceleration'] = (raw_acc - bias).tolist()

            if 'angular_velocity' in raw_data:
                bias = np.array(cal_params.get('angular_velocity_bias', [0, 0, 0]))
                raw_vel = np.array(raw_data['angular_velocity'])
                calibrated_data['angular_velocity'] = (raw_vel - bias).tolist()

        return calibrated_data

    def save_calibration(self, filename):
        """Save calibration parameters to file"""
        with open(filename, 'w') as f:
            json.dump(self.calibration_parameters, f, indent=2)
        print(f"Calibration parameters saved to {filename}")

    def load_calibration(self, filename):
        """Load calibration parameters from file"""
        with open(filename, 'r') as f:
            self.calibration_parameters = json.load(f)
        self.calibration_complete = True
        print(f"Calibration parameters loaded from {filename}")
```

### Automated Calibration Script

```python
# automated_calibrator.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image, JointState
import numpy as np
import time

class AutomatedCalibrator(Node):
    def __init__(self):
        super().__init__('automated_calibrator')

        # Subscribers for sensor data from both platforms
        self.gazebo_lidar_sub = self.create_subscription(
            LaserScan, '/gazebo/laser_scan', self.gazebo_lidar_callback, 10)
        self.unity_lidar_sub = self.create_subscription(
            LaserScan, '/unity/laser_scan', self.unity_lidar_callback, 10)

        self.gazebo_imu_sub = self.create_subscription(
            Imu, '/gazebo/imu', self.gazebo_imu_callback, 10)
        self.unity_imu_sub = self.create_subscription(
            Imu, '/unity/imu', self.unity_imu_callback, 10)

        # Data collection buffers
        self.gazebo_data_buffer = {'lidar': [], 'imu': []}
        self.unity_data_buffer = {'lidar': [], 'imu': []}

        # Timer for calibration process
        self.calibration_timer = self.create_timer(10.0, self.run_calibration)
        self.data_collection_time = 0.0
        self.max_collection_time = 60.0  # Collect for 1 minute

        self.calibrator = CrossPlatformCalibrator()
        self.calibration_complete = False

    def gazebo_lidar_callback(self, msg):
        """Process Gazebo LIDAR data"""
        if self.data_collection_time < self.max_collection_time and not self.calibration_complete:
            lidar_data = {
                'ranges': list(msg.ranges),
                'intensities': list(msg.intensities),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.gazebo_data_buffer['lidar'].append(lidar_data)

    def unity_lidar_callback(self, msg):
        """Process Unity LIDAR data"""
        if self.data_collection_time < self.max_collection_time and not self.calibration_complete:
            lidar_data = {
                'ranges': list(msg.ranges),
                'intensities': list(msg.intensities),
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.unity_data_buffer['lidar'].append(lidar_data)

    def gazebo_imu_callback(self, msg):
        """Process Gazebo IMU data"""
        if self.data_collection_time < self.max_collection_time and not self.calibration_complete:
            imu_data = {
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.gazebo_data_buffer['imu'].append(imu_data)

    def unity_imu_callback(self, msg):
        """Process Unity IMU data"""
        if self.data_collection_time < self.max_collection_time and not self.calibration_complete:
            imu_data = {
                'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
                'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            }
            self.unity_data_buffer['imu'].append(imu_data)

    def run_calibration(self):
        """Run the calibration process"""
        self.data_collection_time += 10.0

        if self.data_collection_time >= self.max_collection_time and not self.calibration_complete:
            self.get_logger().info('Starting calibration process...')

            # Collect data and perform calibration
            self.calibrator.collect_calibration_data(
                self.gazebo_data_buffer,
                self.unity_data_buffer
            )

            # Perform sensor-specific calibrations
            try:
                self.calibrator.calibrate_lidar()
                self.calibrator.calibrate_imu()
                self.calibrator.calibrate_camera()  # Simplified
            except Exception as e:
                self.get_logger().error(f'Calibration error: {e}')
                return

            # Save calibration results
            self.calibrator.save_calibration('/tmp/multi_sim_calib.json')
            self.calibration_complete = True

            self.get_logger().info('Calibration process completed and saved!')

            # Stop the timer after calibration
            self.calibration_timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    calibrator = AutomatedCalibrator()

    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        pass
    finally:
        calibrator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Data Validation Techniques

Validating sensor data consistency is crucial for ensuring reliable multi-simulator operation:

### Consistency Validation Framework

```python
# data_validator.py
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class SensorDataValidator:
    def __init__(self, tolerance_threshold=0.05):
        self.tolerance = tolerance_threshold
        self.validation_results = {}
        self.comparison_data = {}

    def validate_lidar_consistency(self, gazebo_lidar, unity_lidar, timestamps=None):
        """Validate LIDAR data consistency between platforms"""
        if len(gazebo_lidar) != len(unity_lidar):
            raise ValueError("Gazebo and Unity LIDAR data must have the same length")

        differences = []
        valid_comparisons = 0

        for i in range(len(gazebo_lidar)):
            gazebo_ranges = gazebo_lidar[i]['ranges']
            unity_ranges = unity_lidar[i]['ranges']

            if len(gazebo_ranges) != len(unity_ranges):
                continue  # Skip if ranges have different lengths

            # Calculate differences for valid range values
            for j in range(len(gazebo_ranges)):
                gz_val = gazebo_ranges[j]
                un_val = unity_ranges[j]

                # Only compare if both values are valid (positive)
                if gz_val > 0 and un_val > 0:
                    diff = abs(gz_val - un_val)
                    differences.append(diff)
                    valid_comparisons += 1

        if differences:
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            max_diff = np.max(differences)

            # Calculate consistency percentage (within tolerance)
            within_tolerance = sum(1 for d in differences if d <= self.tolerance)
            consistency_percentage = (within_tolerance / len(differences)) * 100

            result = {
                'mean_difference': float(mean_diff),
                'std_difference': float(std_diff),
                'max_difference': float(max_diff),
                'consistency_percentage': float(consistency_percentage),
                'total_comparisons': len(differences),
                'within_tolerance_count': within_tolerance,
                'is_consistent': consistency_percentage >= 95.0  # 95% consistency threshold
            }

            self.validation_results['lidar'] = result
            return result
        else:
            return {
                'mean_difference': 0.0,
                'std_difference': 0.0,
                'max_difference': 0.0,
                'consistency_percentage': 0.0,
                'total_comparisons': 0,
                'within_tolerance_count': 0,
                'is_consistent': False
            }

    def validate_imu_consistency(self, gazebo_imu, unity_imu):
        """Validate IMU data consistency between platforms"""
        if len(gazebo_imu) != len(unity_imu):
            raise ValueError("Gazebo and Unity IMU data must have the same length")

        # Validate linear acceleration
        linear_acc_diffs = []
        for gz, un in zip(gazebo_imu, unity_imu):
            gz_acc = np.array(gz['linear_acceleration'])
            un_acc = np.array(un['linear_acceleration'])
            diff = np.linalg.norm(gz_acc - un_acc)
            linear_acc_diffs.append(diff)

        # Validate angular velocity
        angular_vel_diffs = []
        for gz, un in zip(gazebo_imu, unity_imu):
            gz_vel = np.array(gz['angular_velocity'])
            un_vel = np.array(un['angular_velocity'])
            diff = np.linalg.norm(gz_vel - un_vel)
            angular_vel_diffs.append(diff)

        # Validate orientation (using quaternion distance)
        orientation_diffs = []
        for gz, un in zip(gazebo_imu, unity_imu):
            gz_quat = np.array(gz['orientation'])
            un_quat = np.array(un['orientation'])
            # Calculate quaternion distance
            dot_product = abs(np.dot(gz_quat, un_quat))
            angle_diff = 2 * np.arccos(min(1.0, dot_product))  # Ensure within valid range
            orientation_diffs.append(angle_diff)

        result = {
            'linear_acceleration': {
                'mean_difference': float(np.mean(linear_acc_diffs)),
                'std_difference': float(np.std(linear_acc_diffs)),
                'max_difference': float(np.max(linear_acc_diffs)),
                'consistency_percentage': float((sum(1 for d in linear_acc_diffs if d <= self.tolerance) / len(linear_acc_diffs)) * 100)
            },
            'angular_velocity': {
                'mean_difference': float(np.mean(angular_vel_diffs)),
                'std_difference': float(np.std(angular_vel_diffs)),
                'max_difference': float(np.max(angular_vel_diffs)),
                'consistency_percentage': float((sum(1 for d in angular_vel_diffs if d <= self.tolerance) / len(angular_vel_diffs)) * 100)
            },
            'orientation': {
                'mean_difference': float(np.mean(orientation_diffs)),
                'std_difference': float(np.std(orientation_diffs)),
                'max_difference': float(np.max(orientation_diffs)),
                'consistency_percentage': float((sum(1 for d in orientation_diffs if d <= self.tolerance) / len(orientation_diffs)) * 100)
            },
            'overall_consistency': all(
                data['consistency_percentage'] >= 95.0
                for data in [result['linear_acceleration'], result['angular_velocity'], result['orientation']]
            )
        }

        self.validation_results['imu'] = result
        return result

    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        report = {
            'timestamp': time.time(),
            'tolerance_threshold': self.tolerance,
            'validation_results': self.validation_results,
            'summary': {
                'all_consistent': all(
                    result.get('is_consistent', False) if isinstance(result, dict) and 'is_consistent' in result
                    else result.get('overall_consistency', False) if isinstance(result, dict)
                    for result in self.validation_results.values()
                )
            }
        }

        return report

    def plot_validation_results(self):
        """Create plots for validation results"""
        if not self.validation_results:
            print("No validation results to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sensor Data Consistency Validation Results')

        # Plot LIDAR validation results
        if 'lidar' in self.validation_results:
            lidar_result = self.validation_results['lidar']
            axes[0, 0].bar(['Mean', 'Std', 'Max'],
                          [lidar_result['mean_difference'],
                           lidar_result['std_difference'],
                           lidar_result['max_difference']])
            axes[0, 0].set_title('LIDAR Differences')
            axes[0, 0].set_ylabel('Difference (m)')

        # Plot IMU validation results
        if 'imu' in self.validation_results:
            imu_result = self.validation_results['imu']
            imu_categories = ['Linear Acc', 'Angular Vel', 'Orientation']
            means = [imu_result[cat]['mean_difference'] for cat in imu_categories]
            stds = [imu_result[cat]['std_difference'] for cat in imu_categories]

            x_pos = np.arange(len(imu_categories))
            axes[0, 1].bar(x_pos, means, yerr=stds, capsize=5)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(imu_categories)
            axes[0, 1].set_title('IMU Differences')
            axes[0, 1].set_ylabel('Difference')

        # Plot consistency percentages
        if 'lidar' in self.validation_results:
            consistency_pct = self.validation_results['lidar']['consistency_percentage']
            axes[1, 0].bar(['LIDAR'], [consistency_pct])
            axes[1, 0].axhline(y=95, color='r', linestyle='--', label='95% Threshold')
            axes[1, 0].set_title('LIDAR Consistency Percentage')
            axes[1, 0].set_ylabel('Percentage (%)')
            axes[1, 0].set_ylim([0, 100])
            axes[1, 0].legend()

        if 'imu' in self.validation_results:
            imu_result = self.validation_results['imu']
            imu_categories = ['Linear Acc', 'Angular Vel', 'Orientation']
            consistency_values = [imu_result[cat]['consistency_percentage'] for cat in imu_categories]

            x_pos = np.arange(len(imu_categories))
            axes[1, 1].bar(x_pos, consistency_values)
            axes[1, 1].axhline(y=95, color='r', linestyle='--', label='95% Threshold')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(imu_categories)
            axes[1, 1].set_title('IMU Consistency Percentages')
            axes[1, 1].set_ylabel('Percentage (%)')
            axes[1, 1].set_ylim([0, 100])
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig('/tmp/sensor_validation_report.png')
        plt.show()
```

## Implementation Example: Sensor Data Pipeline

Here's a complete example of how to implement a sensor data consistency pipeline:

```python
# sensor_consistency_pipeline.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, JointState
from std_msgs.msg import String
import json
import time

class SensorConsistencyPipeline(Node):
    def __init__(self):
        super().__init__('sensor_consistency_pipeline')

        # Initialize components
        self.standardizer = CoordinateSystemStandardizer()
        self.calibrator = CrossPlatformCalibrator()
        self.validator = SensorDataValidator(tolerance_threshold=0.05)

        # Load existing calibration if available
        try:
            self.calibrator.load_calibration('/tmp/multi_sim_calib.json')
        except FileNotFoundError:
            self.get_logger().info('No existing calibration found, will use default parameters')

        # Publishers for standardized data
        self.std_lidar_pub = self.create_publisher(LaserScan, '/standardized/laser_scan', 10)
        self.std_imu_pub = self.create_publisher(Imu, '/standardized/imu', 10)

        # Subscribers for raw platform data
        self.gazebo_lidar_sub = self.create_subscription(
            LaserScan, '/gazebo/laser_scan', self.process_gazebo_lidar, 10)
        self.unity_lidar_sub = self.create_subscription(
            LaserScan, '/unity/laser_scan', self.process_unity_lidar, 10)

        self.gazebo_imu_sub = self.create_subscription(
            Imu, '/gazebo/imu', self.process_gazebo_imu, 10)
        self.unity_imu_sub = self.create_subscription(
            Imu, '/unity/imu', self.process_unity_imu, 10)

        # Timer for periodic validation
        self.validation_timer = self.create_timer(30.0, self.run_validation)

        self.get_logger().info('Sensor consistency pipeline initialized')

    def process_gazebo_lidar(self, msg):
        """Process Gazebo LIDAR data through consistency pipeline"""
        # Convert to internal format
        raw_data = {
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        }

        # Standardize the data
        standardized_data = self.standardizer.standardize_lidar_data(raw_data, 'gazebo')

        # Apply calibration if available
        if self.calibrator.calibration_complete:
            calibrated_data = self.calibrator.apply_calibration(standardized_data, 'gazebo', 'lidar')
        else:
            calibrated_data = standardized_data

        # Publish standardized data
        std_msg = LaserScan()
        std_msg.header = msg.header
        std_msg.header.frame_id = 'standardized_laser_frame'
        std_msg.angle_min = calibrated_data['angle_min']
        std_msg.angle_max = calibrated_data['angle_max']
        std_msg.angle_increment = calibrated_data['angle_increment']
        std_msg.time_increment = calibrated_data['time_increment']
        std_msg.scan_time = calibrated_data['scan_time']
        std_msg.range_min = calibrated_data['range_min']
        std_msg.range_max = calibrated_data['range_max']
        std_msg.ranges = calibrated_data['ranges']
        std_msg.intensities = calibrated_data.get('intensities', [])

        self.std_lidar_pub.publish(std_msg)

    def process_unity_lidar(self, msg):
        """Process Unity LIDAR data through consistency pipeline"""
        # Convert to internal format
        raw_data = {
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max
        }

        # Standardize the data
        standardized_data = self.standardizer.standardize_lidar_data(raw_data, 'unity')

        # Apply calibration if available
        if self.calibrator.calibration_complete:
            calibrated_data = self.calibrator.apply_calibration(standardized_data, 'unity', 'lidar')
        else:
            calibrated_data = standardized_data

        # Publish standardized data
        std_msg = LaserScan()
        std_msg.header = msg.header
        std_msg.header.frame_id = 'standardized_laser_frame'
        std_msg.angle_min = calibrated_data['angle_min']
        std_msg.angle_max = calibrated_data['angle_max']
        std_msg.angle_increment = calibrated_data['angle_increment']
        std_msg.time_increment = calibrated_data['time_increment']
        std_msg.scan_time = calibrated_data['scan_time']
        std_msg.range_min = calibrated_data['range_min']
        std_msg.range_max = calibrated_data['range_max']
        std_msg.ranges = calibrated_data['ranges']
        std_msg.intensities = calibrated_data.get('intensities', [])

        self.std_lidar_pub.publish(std_msg)

    def process_gazebo_imu(self, msg):
        """Process Gazebo IMU data through consistency pipeline"""
        # Apply coordinate system transformation
        orientation_ros = self.standardizer.transform_orientation(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'gazebo', 'ros'
        )

        # Apply calibration if available
        if self.calibrator.calibration_complete:
            # Apply bias correction
            cal_params = self.calibrator.calibration_parameters.get('imu', {})
            bias_acc = np.array(cal_params.get('linear_acceleration_bias', [0, 0, 0]))
            bias_vel = np.array(cal_params.get('angular_velocity_bias', [0, 0, 0]))

            linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]) - bias_acc
            angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]) - bias_vel
        else:
            linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # Publish standardized data
        std_msg = Imu()
        std_msg.header = msg.header
        std_msg.header.frame_id = 'standardized_imu_frame'
        std_msg.orientation.x = orientation_ros[0]
        std_msg.orientation.y = orientation_ros[1]
        std_msg.orientation.z = orientation_ros[2]
        std_msg.orientation.w = orientation_ros[3]
        std_msg.linear_acceleration.x = linear_acc[0]
        std_msg.linear_acceleration.y = linear_acc[1]
        std_msg.linear_acceleration.z = linear_acc[2]
        std_msg.angular_velocity.x = angular_vel[0]
        std_msg.angular_velocity.y = angular_vel[1]
        std_msg.angular_velocity.z = angular_vel[2]

        # Copy covariance matrices
        std_msg.linear_acceleration_covariance = msg.linear_acceleration_covariance
        std_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        std_msg.orientation_covariance = msg.orientation_covariance

        self.std_imu_pub.publish(std_msg)

    def process_unity_imu(self, msg):
        """Process Unity IMU data through consistency pipeline"""
        # Apply coordinate system transformation
        orientation_ros = self.standardizer.transform_orientation(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'unity', 'ros'
        )

        # Apply calibration if available
        if self.calibrator.calibration_complete:
            # Apply bias correction
            cal_params = self.calibrator.calibration_parameters.get('imu', {})
            bias_acc = np.array(cal_params.get('linear_acceleration_bias', [0, 0, 0]))
            bias_vel = np.array(cal_params.get('angular_velocity_bias', [0, 0, 0]))

            linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]) - bias_acc
            angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]) - bias_vel
        else:
            linear_acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
            angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])

        # Publish standardized data
        std_msg = Imu()
        std_msg.header = msg.header
        std_msg.header.frame_id = 'standardized_imu_frame'
        std_msg.orientation.x = orientation_ros[0]
        std_msg.orientation.y = orientation_ros[1]
        std_msg.orientation.z = orientation_ros[2]
        std_msg.orientation.w = orientation_ros[3]
        std_msg.linear_acceleration.x = linear_acc[0]
        std_msg.linear_acceleration.y = linear_acc[1]
        std_msg.linear_acceleration.z = linear_acc[2]
        std_msg.angular_velocity.x = angular_vel[0]
        std_msg.angular_velocity.y = angular_vel[1]
        std_msg.angular_velocity.z = angular_vel[2]

        # Copy covariance matrices
        std_msg.linear_acceleration_covariance = msg.linear_acceleration_covariance
        std_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        std_msg.orientation_covariance = msg.orientation_covariance

        self.std_imu_pub.publish(std_msg)

    def run_validation(self):
        """Run periodic validation of sensor data consistency"""
        self.get_logger().info('Running sensor data validation...')

        # In a real implementation, we would collect recent data and validate it
        # For this example, we'll just report that validation is running
        self.get_logger().info('Sensor data validation completed')

def main(args=None):
    rclpy.init(args=args)
    pipeline = SensorConsistencyPipeline()

    try:
        rclpy.spin(pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Sensor Data Consistency

When implementing sensor data consistency across platforms, consider these best practices:

### 1. Regular Calibration
- Perform calibration periodically to account for drift
- Use automated calibration routines when possible
- Store calibration parameters for reuse

### 2. Data Validation
- Implement continuous validation during operation
- Set appropriate tolerance thresholds
- Log inconsistencies for analysis

### 3. Error Handling
- Handle sensor failures gracefully
- Provide fallback mechanisms
- Implement data quality indicators

### 4. Performance Optimization
- Minimize computational overhead
- Use efficient data structures
- Consider real-time constraints

## Summary

In this lesson, we explored the critical aspects of ensuring sensor data consistency across Gazebo and Unity simulation platforms. We covered:

- Understanding fundamental differences between sensor implementations in each platform
- Creating standardization frameworks for data formats and coordinate systems
- Implementing comprehensive calibration procedures
- Developing validation techniques to verify consistency
- Building complete sensor data processing pipelines

These techniques ensure that sensor data maintains reliability and accuracy across different simulation environments, enabling consistent robot behavior validation and cross-platform testing. In the next lesson, we'll focus on validation and verification techniques for multi-simulator environments.