from setuptools import setup
from glob import glob
import os

package_name = 'advanced_communication_tutorials'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@todo.todo',
    description='Advanced ROS2 Communication Tutorials',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multi_communication_node = advanced_communication_tutorials.multi_communication_node:main',
            'sensor_publisher = advanced_communication_tutorials.sensor_publisher:main',
            'robot_state_server = advanced_communication_tutorials.robot_state_server:main',
            'robot_command_client = advanced_communication_tutorials.robot_command_client:main',
            'parameterized_robot_node = advanced_communication_tutorials.parameterized_robot_node:main',
        ],
    },
)