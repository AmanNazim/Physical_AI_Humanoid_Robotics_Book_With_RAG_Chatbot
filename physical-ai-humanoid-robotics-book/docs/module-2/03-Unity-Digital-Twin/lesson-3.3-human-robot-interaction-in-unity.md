---
title: Lesson 3.3 – Human-Robot Interaction in Unity
sidebar_position: 3
---

# Lesson 3.3 – Human-Robot Interaction in Unity

## Learning Objectives

By the end of this lesson, you will be able to:
- Implement human-robot interaction scenarios in Unity environment with intuitive user interfaces
- Create user interfaces for interaction mechanics that enable effective human-robot collaboration
- Develop collaborative task scenarios for human-robot interaction in Unity environments
- Test interaction mechanics with humanoid robot models to validate functionality
- Apply best practices for human-robot interaction design in Unity
- Demonstrate safety protocols through simulation scenarios

## Introduction

Human-robot interaction (HRI) represents a critical aspect of modern robotics, especially for humanoid robots designed to work alongside humans. This lesson focuses on implementing sophisticated interaction systems within Unity that enable meaningful collaboration between humans and robots. Through intuitive user interfaces and well-designed interaction mechanics, you'll learn to create scenarios where humans and robots can work together effectively in shared environments.

The Unity environment provides unique advantages for developing and testing HRI scenarios. Its real-time rendering capabilities allow for immediate visual feedback, while its flexible scripting system enables complex interaction behaviors. The visual nature of Unity also makes it ideal for prototyping and validating HRI concepts in simulation environments.

## Understanding Human-Robot Interaction in Unity

### Core Concepts of HRI

Human-robot interaction encompasses various forms of communication and collaboration between humans and robots:

- **Command and Control**: Direct instruction of robot behaviors
- **Collaborative Tasks**: Joint activities where humans and robots work together
- **Social Interaction**: Communication through gestures, expressions, and social cues
- **Safety Protocols**: Mechanisms to ensure safe interaction in shared spaces

### Unity's Role in HRI Development

Unity serves as an ideal platform for HRI development due to:

- **Visual Feedback**: Real-time visualization of robot states and intentions
- **Interactive Elements**: Built-in UI systems for human input
- **Physics Simulation**: Realistic interaction with objects and environment
- **Animation Systems**: Sophisticated robot movement and expression capabilities
- **Multi-Modal Input**: Support for various input methods (keyboard, mouse, VR controllers)

## Creating User Interfaces for Interaction Mechanics

### Unity UI System Overview

Unity's UI system provides comprehensive tools for creating interactive interfaces:

- **Canvas**: The root object for all UI elements
- **Event System**: Handles input and user interactions
- **UI Elements**: Buttons, sliders, text, images, and other interactive components
- **Layout Groups**: Automatic arrangement of UI elements

### Step 1: Setting Up the UI Canvas

1. Create a new UI Canvas (GameObject > UI > Canvas)
2. Configure Canvas settings:
   - Render Mode: Screen Space - Overlay (for basic UI)
   - Scale Mode: Scale With Screen Size
   - Reference Resolution: Set appropriate resolution (e.g., 1920x1080)

3. Add an EventSystem (GameObject > UI > Event System) if not automatically created
4. Create a UI Image as the main background panel
5. Configure the panel's properties:
   - Color: Semi-transparent dark background
   - Size: Appropriate for your interface needs
   - Anchors: Set to stretch across desired area

### Step 2: Creating Robot Control Interface

#### Basic Command Panel
1. Create a vertical layout group for command buttons
2. Add buttons for basic robot commands:
   - Move Forward
   - Turn Left/Right
   - Stop
   - Reset Position

3. Create a script to handle button commands:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotCommandPanel : MonoBehaviour
{
    public GameObject robot; // Assign the robot GameObject
    public float moveSpeed = 2.0f;
    public float turnSpeed = 90.0f;

    private void Start()
    {
        // Find robot if not assigned in inspector
        if (robot == null)
            robot = GameObject.FindGameObjectWithTag("Robot");
    }

    public void MoveForward()
    {
        if (robot != null)
            robot.transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);
    }

    public void TurnLeft()
    {
        if (robot != null)
            robot.transform.Rotate(Vector3.up, -turnSpeed * Time.deltaTime);
    }

    public void TurnRight()
    {
        if (robot != null)
            robot.transform.Rotate(Vector3.up, turnSpeed * Time.deltaTime);
    }

    public void StopRobot()
    {
        // Implementation depends on your robot's movement system
        Debug.Log("Robot stop command received");
    }

    public void ResetPosition()
    {
        if (robot != null)
        {
            robot.transform.position = new Vector3(0, 0, 0);
            robot.transform.rotation = Quaternion.identity;
        }
    }
}
```

4. Assign the script to a GameObject and connect buttons to the appropriate methods

#### Advanced Control Panel
1. Add sliders for fine-tuning robot parameters:
   - Speed control slider
   - Joint angle sliders
   - Sensitivity adjustments

2. Create toggle switches for robot modes:
   - Autonomous vs. Manual control
   - Different behavior modes
   - Safety system toggles

3. Add text displays for robot status:
   - Current position coordinates
   - Battery level
   - Task completion status
   - Sensor readings

### Step 3: Implementing Gesture Recognition Interface

#### Touch and Mouse Interaction
1. Create interactive zones for gesture recognition:
   - Drag areas for robot movement
   - Click zones for specific actions
   - Gesture recognition areas

2. Implement gesture recognition script:

```csharp
using UnityEngine;
using UnityEngine.EventSystems;

public class GestureRecognition : MonoBehaviour, IPointerDownHandler, IDragHandler, IPointerUpHandler
{
    public GameObject robot;
    public Camera mainCamera;

    private Vector2 dragStartPosition;
    private bool isDragging = false;

    public void OnPointerDown(PointerEventData eventData)
    {
        dragStartPosition = eventData.position;
        isDragging = true;
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (robot != null && isDragging)
        {
            Vector2 dragDelta = eventData.position - dragStartPosition;

            // Convert screen coordinates to world coordinates
            Vector3 worldPos = mainCamera.ScreenToWorldPoint(
                new Vector3(eventData.position.x, eventData.position.y,
                           mainCamera.WorldToScreenPoint(robot.transform.position).z));

            // Move robot toward the gesture direction
            robot.transform.position = new Vector3(worldPos.x, robot.transform.position.y, worldPos.z);
        }
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        isDragging = false;
    }
}
```

### Step 4: Creating Status and Feedback Displays

#### Robot Status Panel
1. Create a status panel showing:
   - Current task progress
   - Battery level (with visual indicator)
   - Connection status
   - Error messages or warnings

2. Implement status update script:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotStatusDisplay : MonoBehaviour
{
    public Text positionText;
    public Text batteryText;
    public Text taskText;
    public Slider batterySlider;
    public Image connectionIndicator;

    public GameObject robot;
    private float batteryLevel = 100f;
    private bool isConnected = true;

    void Update()
    {
        UpdatePositionDisplay();
        UpdateBatteryDisplay();
        UpdateTaskDisplay();
        UpdateConnectionStatus();
    }

    void UpdatePositionDisplay()
    {
        if (robot != null && positionText != null)
        {
            Vector3 pos = robot.transform.position;
            positionText.text = $"Position: ({pos.x:F2}, {pos.y:F2}, {pos.z:F2})";
        }
    }

    void UpdateBatteryDisplay()
    {
        if (batteryText != null)
        {
            batteryText.text = $"Battery: {batteryLevel:F1}%";
        }

        if (batterySlider != null)
        {
            batterySlider.value = batteryLevel / 100f;
        }

        // Simulate battery drain
        batteryLevel -= 0.01f * Time.deltaTime;
        if (batteryLevel < 0) batteryLevel = 0;
    }

    void UpdateTaskDisplay()
    {
        if (taskText != null)
        {
            // Update based on robot's current task
            taskText.text = "Task: Idle";
        }
    }

    void UpdateConnectionStatus()
    {
        if (connectionIndicator != null)
        {
            connectionIndicator.color = isConnected ? Color.green : Color.red;
        }
    }
}
```

## Developing Collaborative Task Scenarios

### Understanding Collaborative Robotics

Collaborative tasks involve humans and robots working together to achieve common goals. These scenarios require:

- Clear task division between human and robot
- Effective communication channels
- Safety protocols and emergency procedures
- Adaptive behavior based on human actions

### Step 1: Designing Collaborative Task Framework

#### Task Structure Definition
1. Define task phases:
   - Task assignment and planning
   - Execution phase with role distribution
   - Monitoring and adaptation phase
   - Completion and evaluation phase

2. Create task management system:

```csharp
using UnityEngine;
using System.Collections.Generic;

public enum TaskPhase
{
    Assignment,
    Planning,
    Execution,
    Monitoring,
    Completion,
    Evaluation
}

public class CollaborativeTaskManager : MonoBehaviour
{
    public TaskPhase currentPhase = TaskPhase.Assignment;
    public List<GameObject> humanParticipants;
    public List<GameObject> robotParticipants;
    public string taskDescription;

    private Dictionary<string, object> taskData = new Dictionary<string, object>();

    public void StartTask(string description)
    {
        taskDescription = description;
        currentPhase = TaskPhase.Assignment;
        InitializeTask();
    }

    private void InitializeTask()
    {
        // Set up task-specific parameters
        // Assign roles to participants
        // Initialize communication channels
    }

    public void AdvancePhase()
    {
        switch (currentPhase)
        {
            case TaskPhase.Assignment:
                currentPhase = TaskPhase.Planning;
                break;
            case TaskPhase.Planning:
                currentPhase = TaskPhase.Execution;
                break;
            case TaskPhase.Execution:
                currentPhase = TaskPhase.Monitoring;
                break;
            case TaskPhase.Monitoring:
                currentPhase = TaskPhase.Completion;
                break;
            case TaskPhase.Completion:
                currentPhase = TaskPhase.Evaluation;
                break;
        }
    }

    public void ExecuteTask()
    {
        // Execute current phase logic
        switch (currentPhase)
        {
            case TaskPhase.Execution:
                ExecuteTaskPhase();
                break;
            case TaskPhase.Monitoring:
                MonitorTaskPhase();
                break;
        }
    }

    private void ExecuteTaskPhase()
    {
        // Implement specific task execution logic
    }

    private void MonitorTaskPhase()
    {
        // Monitor task progress and adapt as needed
    }
}
```

### Step 2: Implementing Specific Collaborative Scenarios

#### Scenario 1: Object Transportation Task
1. Set up the environment with objects to transport
2. Define robot and human roles:
   - Robot: Navigate to object, grasp and lift
   - Human: Guide robot to destination, verify placement

3. Create interaction points:
   - Object selection interface
   - Destination selection system
   - Progress tracking

4. Implement the transportation logic:

```csharp
using UnityEngine;

public class ObjectTransportationTask : MonoBehaviour
{
    public GameObject[] objectsToTransport;
    public Transform[] destinationPoints;
    public GameObject robot;
    public GameObject humanAvatar;

    private int currentObjectIndex = 0;
    private int currentDestinationIndex = 0;
    private bool isObjectGrasped = false;

    public void StartTransportationTask()
    {
        if (objectsToTransport.Length > 0 && destinationPoints.Length > 0)
        {
            StartCoroutine(TransportationSequence());
        }
    }

    private System.Collections.IEnumerator TransportationSequence()
    {
        while (currentObjectIndex < objectsToTransport.Length)
        {
            // Robot moves to object
            yield return StartCoroutine(MoveRobotToPosition(objectsToTransport[currentObjectIndex].transform.position));

            // Robot grasps object
            yield return StartCoroutine(GraspObject(objectsToTransport[currentObjectIndex]));

            // Human guides to destination
            yield return StartCoroutine(MoveToDestination());

            // Robot releases object
            yield return StartCoroutine(ReleaseObject());

            currentObjectIndex++;
            currentDestinationIndex = Mathf.Min(currentDestinationIndex + 1, destinationPoints.Length - 1);
        }
    }

    private System.Collections.IEnumerator MoveRobotToPosition(Vector3 targetPosition)
    {
        // Move robot to target position with animation
        float duration = 2.0f;
        Vector3 startPosition = robot.transform.position;
        float elapsedTime = 0;

        while (elapsedTime < duration)
        {
            robot.transform.position = Vector3.Lerp(startPosition, targetPosition, elapsedTime / duration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }
    }

    private System.Collections.IEnumerator GraspObject(GameObject obj)
    {
        // Animation and logic for grasping object
        isObjectGrasped = true;
        obj.transform.SetParent(robot.transform);
        obj.GetComponent<Rigidbody>().isKinematic = true;
        yield return new WaitForSeconds(1.0f);
    }

    private System.Collections.IEnumerator MoveToDestination()
    {
        // Move to selected destination point
        Vector3 destination = destinationPoints[currentDestinationIndex].position;
        yield return StartCoroutine(MoveRobotToPosition(destination));
    }

    private System.Collections.IEnumerator ReleaseObject()
    {
        // Release the object
        GameObject currentObject = objectsToTransport[currentObjectIndex];
        currentObject.transform.SetParent(null);
        currentObject.GetComponent<Rigidbody>().isKinematic = false;
        isObjectGrasped = false;
        yield return new WaitForSeconds(0.5f);
    }
}
```

#### Scenario 2: Assembly Task Collaboration
1. Create assembly environment with components
2. Define collaborative assembly process:
   - Human: Complex manipulation requiring dexterity
   - Robot: Precise positioning and heavy lifting
   - Joint: Quality inspection and verification

3. Implement assembly sequence with safety checks

### Step 3: Creating Task Progress and Communication Systems

#### Progress Tracking Interface
1. Create visual progress indicators:
   - Task completion percentage
   - Current step display
   - Timeline visualization
   - Success/failure indicators

2. Implement progress tracking:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class TaskProgressTracker : MonoBehaviour
{
    public Slider progressSlider;
    public Text taskStepText;
    public Text completionText;
    public Image successIndicator;

    private float totalSteps = 10f;
    private float currentStep = 0f;

    public void UpdateProgress(float stepIncrement = 1f)
    {
        currentStep += stepIncrement;
        float progress = currentStep / totalSteps;

        if (progressSlider != null)
            progressSlider.value = progress;

        if (taskStepText != null)
            taskStepText.text = $"Step {currentStep} of {totalSteps}";

        if (completionText != null)
            completionText.text = $"Completion: {(progress * 100):F1}%";

        if (successIndicator != null)
            successIndicator.color = progress >= 1.0f ? Color.green : Color.yellow;
    }

    public void MarkStepComplete()
    {
        UpdateProgress(1f);
    }

    public void ResetProgress()
    {
        currentStep = 0f;
        UpdateProgress(0f);
    }
}
```

#### Communication Interface
1. Create communication panels for:
   - Text-based commands
   - Status updates
   - Error reporting
   - Acknowledgment systems

2. Implement communication protocols:

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class HRICommunicationSystem : MonoBehaviour
{
    public Text communicationLog;
    public InputField messageInput;
    public Button sendButton;

    private List<string> messageHistory = new List<string>();
    private const int maxHistory = 50;

    private void Start()
    {
        if (sendButton != null)
            sendButton.onClick.AddListener(SendMessage);

        if (messageInput != null)
            messageInput.onEndEdit.AddListener(OnMessageInputEnd);
    }

    private void OnMessageInputEnd(string message)
    {
        if (Input.GetKeyDown(KeyCode.Return))
            SendMessage();
    }

    public void SendMessage()
    {
        if (messageInput != null && !string.IsNullOrEmpty(messageInput.text))
        {
            AddMessage($"Human: {messageInput.text}");

            // Process the message and potentially trigger robot response
            ProcessMessage(messageInput.text);

            messageInput.text = "";
        }
    }

    private void ProcessMessage(string message)
    {
        // Simple response system - in real implementation, this would be more sophisticated
        if (message.ToLower().Contains("hello") || message.ToLower().Contains("hi"))
        {
            AddMessage("Robot: Hello! How can I assist you today?");
        }
        else if (message.ToLower().Contains("move") || message.ToLower().Contains("go"))
        {
            AddMessage("Robot: Acknowledged. Moving to designated position.");
            // Trigger robot movement
        }
        else
        {
            AddMessage("Robot: Message received and understood.");
        }
    }

    private void AddMessage(string message)
    {
        messageHistory.Add(message);
        if (messageHistory.Count > maxHistory)
            messageHistory.RemoveAt(0);

        if (communicationLog != null)
        {
            communicationLog.text = string.Join("\n", messageHistory.ToArray());
        }

        Debug.Log(message);
    }

    public void ClearLog()
    {
        messageHistory.Clear();
        if (communicationLog != null)
            communicationLog.text = "";
    }
}
```

## Implementing Safety Protocols and Emergency Procedures

### Safety System Architecture

Safety is paramount in human-robot interaction scenarios. Implement multiple layers of safety protocols:

- **Physical Safety**: Collision avoidance and emergency stops
- **Operational Safety**: Proper task sequencing and error handling
- **Communication Safety**: Clear status indicators and emergency protocols

### Step 1: Emergency Stop System

1. Create emergency stop buttons in the UI
2. Implement safety zones around the robot
3. Add collision detection and avoidance

```csharp
using UnityEngine;

public class SafetySystem : MonoBehaviour
{
    public GameObject robot;
    public float safetyRadius = 2.0f;
    public LayerMask humanLayer;
    private bool isEmergencyStopActive = false;

    void Update()
    {
        CheckSafetyZones();
    }

    void CheckSafetyZones()
    {
        if (robot != null)
        {
            Collider[] nearbyObjects = Physics.OverlapSphere(robot.transform.position, safetyRadius, humanLayer);

            foreach (Collider col in nearbyObjects)
            {
                if (col.CompareTag("Human") || col.CompareTag("Player"))
                {
                    TriggerSafetyProtocol();
                    return;
                }
            }
        }
    }

    void TriggerSafetyProtocol()
    {
        if (!isEmergencyStopActive)
        {
            isEmergencyStopActive = true;
            // Stop robot movement
            if (robot != null)
            {
                // Implementation depends on your robot's movement system
                Debug.LogWarning("Safety zone violation! Robot movement stopped.");
            }

            // Visual indication of safety stop
            StartCoroutine(ReEnableAfterDelay(5.0f)); // Re-enable after 5 seconds
        }
    }

    System.Collections.IEnumerator ReEnableAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        isEmergencyStopActive = false;
        Debug.Log("Safety system re-enabled.");
    }

    public void EmergencyStop()
    {
        isEmergencyStopActive = true;
        Debug.LogWarning("Emergency stop activated!");
    }

    public void EmergencyResume()
    {
        isEmergencyStopActive = false;
        Debug.Log("Emergency stop cleared. Resuming operations.");
    }
}
```

### Step 2: Safe Interaction Boundaries

1. Define safe interaction zones
2. Implement boundary detection
3. Create visual indicators for safety zones

## Testing Interaction Mechanics with Humanoid Robot Models

### Comprehensive Testing Framework

#### Functional Testing
1. Test all UI elements and controls
2. Verify robot response to commands
3. Validate collaborative task execution
4. Check safety system functionality

#### Performance Testing
1. Monitor frame rate during interaction
2. Test with multiple simultaneous interactions
3. Verify stability under stress conditions
4. Profile resource usage

#### Usability Testing
1. Evaluate interface intuitiveness
2. Test with different user skill levels
3. Gather feedback on interaction quality
4. Iterate based on user experience

### Step 1: Basic Interaction Testing

1. Load your humanoid robot model into the scene
2. Attach interaction scripts to the robot
3. Test basic movement and control commands
4. Verify UI responsiveness and feedback

### Step 2: Collaborative Task Testing

1. Execute collaborative scenarios with multiple participants
2. Test task completion under various conditions
3. Verify safety protocol activation and deactivation
4. Monitor communication system performance

### Step 3: Stress and Edge Case Testing

1. Test with rapid succession of commands
2. Verify behavior when multiple safety systems trigger
3. Test boundary conditions and error states
4. Validate recovery from various failure modes

## Best Practices for Human-Robot Interaction Design

### User Experience Considerations
- Keep interfaces simple and intuitive
- Provide clear feedback for all actions
- Maintain consistent interaction patterns
- Design for users with varying technical expertise

### Safety-First Design
- Implement multiple safety layers
- Provide clear emergency procedures
- Ensure graceful degradation of systems
- Test all safety scenarios thoroughly

### Performance Optimization
- Optimize UI rendering for smooth interaction
- Implement efficient communication protocols
- Use appropriate update frequencies
- Monitor and optimize resource usage

## Tools Required

- Unity Editor with UI and Animation packages
- Unity Robotics packages for integration
- Graphics hardware capable of rendering complex scenes
- Basic understanding of Unity interface and animation system
- Humanoid robot model or placeholder assets

## Summary

In this lesson, you've implemented comprehensive human-robot interaction scenarios in Unity with intuitive user interfaces and collaborative task systems. You've learned to create sophisticated interaction mechanics that enable meaningful collaboration between humans and robots, including advanced UI systems, communication protocols, and safety mechanisms.

The human-robot interaction capabilities developed in this lesson complete the visualization layer of Module 2's digital twin architecture. You now have the tools and knowledge to create immersive, interactive robotics simulations that enable effective human-robot collaboration and testing.

These interaction systems provide the foundation for the final chapter of Module 2, where you'll integrate Unity's visualization and interaction capabilities with Gazebo's physics simulation to create a comprehensive digital twin system that combines visual, physical, and interactive elements for advanced humanoid robotics development.

The skills acquired throughout this chapter—environment setup, high-fidelity rendering, and human-robot interaction—form a complete toolkit for creating sophisticated robotics simulation environments that can support complex research, development, and testing scenarios.