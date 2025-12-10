from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package share directory
    pkg_dir = get_package_share_directory('advanced_communication_tutorials')

    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'default_robot_config.yaml'),
        description='Path to the parameters file'
    )

    return LaunchDescription([
        params_file_arg,

        # Multi-communication node
        Node(
            package='advanced_communication_tutorials',
            executable='multi_communication_node',
            name='multi_comm_node',
            output='screen',
        ),

        # Parameterized robot node with config file
        Node(
            package='advanced_communication_tutorials',
            executable='parameterized_robot_node',
            name='param_robot_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen',
        ),

        # Robot state server
        Node(
            package='advanced_communication_tutorials',
            executable='robot_state_server',
            name='robot_state_server',
            output='screen',
        ),
    ])