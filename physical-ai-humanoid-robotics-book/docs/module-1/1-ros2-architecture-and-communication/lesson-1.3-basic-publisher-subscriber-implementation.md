---
title: Lesson 1.3 - Basic Publisher/Subscriber Implementation
---

# Lesson 1.3 – Basic Publisher/Subscriber Implementation

## Learning Objectives

By the end of this lesson, you will be able to:
- Write and execute a basic publisher node in Python
- Write and execute a basic subscriber node in Python
- Launch and test a ROS2 communication graph
- Understand the message flow between publisher and subscriber nodes
- Use ROS2 tools to verify communication between nodes

## Concept Overview and Scope

This practical lesson teaches you to implement the most fundamental ROS2 communication pattern: publisher-subscriber. You'll write your first ROS2 nodes in Python, creating a publisher that sends messages and a subscriber that receives them. The lesson emphasizes understanding message flow and the asynchronous nature of topic-based communication.

The publisher-subscriber pattern is fundamental to Physical AI systems, enabling distributed, modular robot architectures where sensor nodes, processing nodes, and control nodes can communicate seamlessly.

## Understanding Publisher-Subscriber Communication Pattern

The publisher-subscriber pattern is the backbone of ROS2 communication. This pattern enables asynchronous, decoupled communication between nodes:

- **Publisher**: A node that sends messages to a specific topic
- **Subscriber**: A node that receives messages from a specific topic
- **Topic**: A named channel over which messages are sent
- **Message**: The data structure that carries information between nodes

This creates a decoupled system where publishers don't need to know about subscribers and vice versa. This decoupling is what allows for flexible, modular robot architectures.

## Creating the Publisher Node

### Step 1: Setting Up the Package Structure

First, navigate to your workspace and create a new package for our publisher/subscriber example:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python publisher_subscriber_example
cd publisher_subscriber_example
```

### Step 2: Creating the Publisher Node

Create the publisher node file in the package directory:

```bash
mkdir -p publisher_subscriber_example
touch publisher_subscriber_example/publisher_node.py
```

Now edit the publisher node file with the following content:

```python
#!/usr/bin/env python3
# publisher_subscriber_example/publisher_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 3: Understanding the Publisher Code

Let's break down the key components of the publisher node:

1. **Node Initialization**: `super().__init__('minimal_publisher')` creates a ROS2 node named 'minimal_publisher'
2. **Publisher Creation**: `self.create_publisher(String, 'topic', 10)` creates a publisher that sends String messages on the 'topic' with a queue size of 10
3. **Timer Callback**: Creates a timer that calls the callback function every 0.5 seconds
4. **Message Creation**: Creates a String message with the current count
5. **Message Publishing**: Publishes the message to the topic
6. **Logging**: Logs the published message to the console

## Creating the Subscriber Node

### Step 4: Creating the Subscriber Node

Create the subscriber node file:

```bash
touch publisher_subscriber_example/subscriber_node.py
```

Edit the subscriber node file with the following content:

```python
#!/usr/bin/env python3
# publisher_subscriber_example/subscriber_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Step 5: Understanding the Subscriber Code

Key components of the subscriber node:

1. **Node Initialization**: Creates a ROS2 node named 'minimal_subscriber'
2. **Subscription Creation**: `self.create_subscription(String, 'topic', self.listener_callback, 10)` creates a subscription to the 'topic' that receives String messages
3. **Callback Function**: `listener_callback` is called whenever a message is received on the topic
4. **Message Processing**: The callback function logs the received message to the console

## Configuring the Package

### Step 6: Setting Up the Entry Points

Edit the `setup.py` file in the package root to include entry points for our nodes:

```python
from setuptools import find_packages, setup

package_name = 'publisher_subscriber_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Basic publisher subscriber example',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher_node = publisher_subscriber_example.publisher_node:main',
            'subscriber_node = publisher_subscriber_example.subscriber_node:main',
        ],
    },
)
```

## Building and Running the Nodes

### Step 7: Building the Package

Navigate to your workspace and build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select publisher_subscriber_example
```

### Step 8: Sourcing the Workspace

Source your workspace to make the new nodes available:

```bash
source install/setup.bash
```

### Step 9: Running the Publisher Node

Open a new terminal and run the publisher node:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_example publisher_node
```

You should see output like:
```
[INFO] [1620000000.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1620000000.623456789] [minimal_publisher]: Publishing: "Hello World: 1"
```

### Step 10: Running the Subscriber Node

Open another terminal and run the subscriber node:

```bash
source ~/ros2_ws/install/setup.bash
ros2 run publisher_subscriber_example subscriber_node
```

You should see output like:
```
[INFO] [1620000000.123456789] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1620000000.623456789] [minimal_subscriber]: I heard: "Hello World: 1"
```

## Understanding Message Flow and Asynchronous Communication

### The Asynchronous Nature of Topics

The publisher-subscriber pattern is asynchronous, meaning:

1. **Publishers don't wait**: The publisher continues executing regardless of whether subscribers exist or are ready to receive messages
2. **Subscribers receive messages**: Subscribers process messages as they arrive from publishers
3. **Decoupled timing**: Publishers and subscribers can run at different rates without interfering with each other

### Quality of Service (QoS) Considerations

The third parameter in both `create_publisher` and `create_subscription` (10 in our example) represents the queue size. This affects how many messages are buffered when there's a mismatch between publishing and subscribing rates.

## Advanced Publisher-Subscriber Concepts

### Multiple Subscribers

You can run multiple subscriber nodes simultaneously, and all will receive the same messages from the publisher:

```bash
# Terminal 1: Publisher
ros2 run publisher_subscriber_example publisher_node

# Terminal 2: Subscriber 1
ros2 run publisher_subscriber_example subscriber_node

# Terminal 3: Subscriber 2
ros2 run publisher_subscriber_example subscriber_node
```

All subscribers will receive the same messages from the publisher.

### Topic Discovery

Use ROS2 command-line tools to examine the communication:

```bash
# List active topics
ros2 topic list

# Get information about a specific topic
ros2 topic info /topic

# Echo messages from a topic (without a subscriber node)
ros2 topic echo /topic std_msgs/msg/String
```

## Physical AI Context and Application

In Physical AI systems, the publisher-subscriber pattern is fundamental to creating distributed, modular robot architectures. For example:

- **Sensor nodes** publish sensor data (camera images, IMU readings, joint states)
- **Processing nodes** subscribe to sensor data and perform computations
- **Control nodes** subscribe to processed information and publish motor commands
- **Monitoring nodes** subscribe to various topics for logging and visualization

This pattern supports the three-layer system:
- **Perception Layer**: Sensor nodes publish raw and processed data
- **Cognition Layer**: Processing nodes interpret sensor data and make decisions
- **Actuation Layer**: Control nodes execute motor commands

## Error Handling and Best Practices

### Node Lifecycle Management

Always properly manage the node lifecycle by destroying the node explicitly when done:

```python
try:
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.destroy_node()
    rclpy.shutdown()
```

### Message Validation

In production code, always validate messages before using them:

```python
def listener_callback(self, msg):
    if len(msg.data) > 0:  # Validate message content
        self.get_logger().info('I heard: "%s"' % msg.data)
    else:
        self.get_logger().warn('Received empty message')
```

## Verification and Testing

### Step 11: Testing Communication Reliability

To verify the communication is working properly:

1. Run both nodes simultaneously
2. Verify that messages published by the publisher are received by the subscriber
3. Check that the message sequence numbers increment correctly
4. Test that multiple subscribers can receive the same messages

### Step 12: Using ROS2 Tools for Verification

```bash
# Check active nodes
ros2 node list

# Check active topics
ros2 topic list

# Check topic information
ros2 topic info /topic

# Monitor message rate
ros2 topic hz /topic
```

## Lesson Summary

In this lesson, you have learned:

- **Publisher Implementation**: How to create a ROS2 node that publishes messages to a topic
- **Subscriber Implementation**: How to create a ROS2 node that subscribes to messages from a topic
- **Message Flow**: How messages travel from publisher to subscriber in an asynchronous manner
- **Node Structure**: The proper structure and lifecycle management for ROS2 nodes
- **Communication Graph**: How publishers and subscribers form a communication network
- **Verification**: How to test and verify communication between nodes

You now have practical experience with the fundamental communication pattern that underlies all ROS2-based robotic systems.

## Tools / References
- Python 3.8+
- rclpy
- ROS2