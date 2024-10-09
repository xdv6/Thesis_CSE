Sure! Here’s a README summarizing all the demo scripts in robosuite, including a brief explanation of each script's purpose and functionality:

---

# Robosuite Demo Scripts

This repository contains a collection of demo scripts showcasing the functionalities of **robosuite**. Each demo script highlights different features of the simulation environment, allowing users to explore and understand the modular design and capabilities of robosuite.

## Demo Scripts

### 1. `demo_random_action.py`
- **Purpose:** Serves as the starter demo script.
- **Functionality:** Allows users to create new simulation instances by selecting one environment, one or more robots, and their controllers via command line. It executes uniform random actions from the specified controller's action space.

### 2. `demo_control.py`
- **Purpose:** Demonstrates the various functionalities of each controller available in robosuite.
- **Functionality:** Conducts a perturbation test for each action dimension, sequentially moving and pausing to showcase the expected behavior of different controllers (e.g., OSC_POSE, OSC_POSITION, IK_POSE, etc.).

### 3. `demo_domain_randomization.py`
- **Purpose:** Showcases domain randomization functionality.
- **Functionality:** Utilizes the `DomainRandomizationWrapper` to apply visual variations such as colors, textures, and camera viewpoints in the simulation.

### 4. `demo_sensor_corruption.py`
- **Purpose:** Demonstrates sensor corruption and delay modeling.
- **Functionality:** Uses the `Observable` class to simulate real-world sensor noise, variable delays, and realistic sampling frequencies, improving policy generalization and sim2real transfer.

### 5. `demo_gripper_selection.py`
- **Purpose:** Illustrates how to select different grippers for an environment.
- **Functionality:** Utilizes the `gripper_type` keyword argument to specify the desired gripper from the available options defined in `robosuite.ALL_GRIPPERS`.

### 6. `demo_gripper_interaction.py`
- **Purpose:** Demonstrates importing grippers into a scene and their interaction with objects.
- **Functionality:** Shows how to procedurally generate a scene using MJCF utility functions while integrating actuated grippers.

### 7. `demo_collect_and_playback_data.py`
- **Purpose:** Records and plays back robot rollout trajectory data.
- **Functionality:** Uses the `DataCollectionWrapper` to record environment states in `.npz` format and replay them, allowing analysis of previous trajectories.

### 8. `demo_gym_functionality.py`
- **Purpose:** Adapts robosuite environments to be compatible with OpenAI Gym-style APIs.
- **Functionality:** Demonstrates integration with OpenAI Gym using the `GymWrapper`, enabling usage with learning pipelines like OpenAI Baselines.

### 9. `demo_device_control.py`
- **Purpose:** Shows how to teleoperate robots using input devices.
- **Functionality:** Allows control of the robot's end-effector via keyboard or SpaceMouse, with options for specifying environment, robots, controller types, and additional settings.

### 10. `demo_video_recording.py`
- **Purpose:** Records a video of robot rollouts.
- **Functionality:** Uses offscreen rendering with the `imageio` library to generate qualitative videos of robot behavior, outputting in mp4 format.

### 11. `demo_renderers.py`
- **Purpose:** Demonstrates the use of different renderers in the simulation.
- **Functionality:** Supports rendering options including MuJoCo (default) and NVISII, allowing users to visualize the simulation using different rendering techniques.

### 12. `demo_nvisii_modalities.py`
- **Purpose:** Illustrates how to obtain vision modalities from the NVISII renderer.
- **Functionality:** Enables rendering of specific vision modalities (depth, normal, segmentation, rgb) and allows adjustment of segmentation levels for detailed visualization.

---

## Getting Started
To run any of the demo scripts, use the following command format:

```bash
$ python <script_name.py> [arguments]
```

Make sure to replace `<script_name.py>` with the desired demo script and provide any required arguments as specified in the script documentation.

Explore these demos to familiarize yourself with robosuite's capabilities and enhance your understanding of robotic simulations!

--- 

Feel free to customize or expand upon this README as needed!