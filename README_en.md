# Training Unitree H1 standing by reinforcement learning in Webots simulator

## 0. Introduction

This project uses reinforcement learning (PPO) to train the **Unitree H1** humanoid robot to accomplish the task of **vertical standing** from **various initial postures**.

### Working Log

- **[2024/07/21]** Completed initial modeling and coding.
- **[2024/07/22]** Adjusted the `v1` version reward function.
- **[2024/07/23]** Discovered that the `.urdf` file used included the dexterous hand version, causing unexpected errors in the simulator due to the complexity of the parts. Added a version of the `.urdf` file **without the dexterous hand**, configured the corresponding `.proto` node and `.wbt` world file, and designed the `UnitreeH1StandingV1` interface for the new version.

## 1. Markov Decision Process

During the Markov Decision Process modeling, the 21 (19 in the version without hands) electric joints of the Unitree H1 humanoid robot are utilized, as shown in the table below.

| Body Parts       | Joint Names                |                           |                          |                   |                                                              |
| ---------------- | -------------------------- | ------------------------- | ------------------------ | ----------------- | ------------------------------------------------------------ |
| Left Lower Limb  | left_hip_yaw_joint         | left_hip_roll_joint       | left_hip_pitch_joint     | left_knee_joint   | left_ankle_joint                                             |
| Right Lower Limb | right_hip_yaw_joint        | right_hip_roll_joint      | right_hip_pitch_joint    | right_knee_joint  | right_ankle_joint                                            |
| Middle Part      | torso_joint                |                           |                          |                   |                                                              |
| Left Upper Limb  | left_shoulder_pitch_joint  | left_shoulder_roll_joint  | left_shoulder_yaw_joint  | left_elbow_joint  | left_hand_joint **(not included in versions without hands)** |
| Right Upper Limb | right_shoulder_pitch_joint | right_shoulder_roll_joint | right_shoulder_yaw_joint | right_elbow_joint | right_hand_joint **(not included in versions without hands)** |

During the modeling of the Markov decision process, one entity of the Unitree H1 robot is used, as shown in the table below.

| Body Parts | Names                         | Comments                                           |
| ---------- | ----------------------------- | -------------------------------------------------- |
| Head       | head (shown in `.proto` file) | The height from the `pelvis_visual` is 0.7 meters. |

### Observation Space

The state space consists of two parts: the current angle values of each joint in the above table, each with 1 dimension; and the current 3-dimensional position from the head sensor, totaling 24 (22 in the version without hands) dimensions.

### Action Space

The action space consists of the angle increments of each joint in the above table, measured in radians, within the range $[-\Delta\theta, \Delta\theta]$. Here, $\Delta\theta$ represents the maximum value of the joint angle increment, meaning the angle increment for each joint is constrained within the range $[-\Delta\theta, \Delta\theta]$. The joint increment expression is: $\theta_{t+1}=\theta_{t}+\Delta\theta$.

In the designed neural network, the activation function of the final layer is the `softmax` activation function, which restricts the output mean $\mu$ and standard deviation $\sigma$ values of the network to the range $[−1,1]$ . Subsequently, the standard deviation $\sigma$ is exponentiated to obtain the final standard deviation, constructing a normal distribution as $\alpha \sim N(\mu, \sigma^{2})$ . To obtain a normal distribution within the range $[-\Delta\theta, \Delta\theta]$ , the normal distribution is linearly scaled as follows: $\alpha^{\prime}=\Delta\theta \cdot \alpha \sim N(\Delta\theta \mu, \Delta\theta^{2} \sigma^{2})$ .

### Reward Function

#### v0 version

The reward function used in the following three images is the `v0` version: the goal is for the Unitree H1 robot's head to **raise its height** and **approach its own height** (1.75 meters) after applying actions. Therefore, the reward decreases as the head distance from 1.75 meters increases. The reward function is given by $\text{reward} = -1 \times |z - 1.75|^{2}$ . During debugging, some clipping was applied: if the head's distance from 1.75 meters exceeds 10 meters, resulting in a reward lower than -100, the reward is capped at -100 to prevent adversely affecting learning performance.

#### v1 version

The issue with the `v0` version is that the reward decreases as the height approaches 1.75 meters, which is strongly related to the PPO algorithm. Consequently, very small reward values lead to smaller gradients of the policy, resulting in the strategy not learning new content effectively. This is evident as the early episodes show large fluctuations, while later episodes "suddenly" approach a reward of 0.

Considering that the goal of reinforcement learning is to maximize cumulative rewards and that the task involves standing up from the ground, the head height should increase away from the ground. Therefore, the reward function should use the difference between the head height and the ground as its variable. Additionally, the possibility of the robot being thrown off due to improper torque needs to be considered.

Thus, the reward function is designed as $\text{reward} = \exp^{2 \times z}$ , where $z$ is the distance between the head height and the ground. When $z > 2.0$ , the reward is capped at 0 .

### Ending Flag of An Episode 

To determine if an episode has ended for automatic data collection in reinforcement learning, there are two conditions for ending the episode:

1. The current time step exceeds the predefined episode length.
2. The Unitree H1 robot lifts off the ground by 3 meters, which can occur due to changes in joint angles and applied torques causing the robot to "bounce" off the ground.

## 2. Algorithm

### PPO

Source: [Written by myself in 2023](https://github.com/JinbiaoZhu/BasicReinforcementLearning.git) .

> **Note:** The following performance display might contain errors and is currently being adjusted.

Performance and loss values are displayed below:

![Total Reward Per Episode](https://github.com/JinbiaoZhu/Unitree_H1_Webots/blob/main/controllers/results/ppo/UnitreeH1_Standing-20240722122807.png?raw=true)

![Loss function Per Episode](https://github.com/JinbiaoZhu/Unitree_H1_Webots/blob/main/controllers/results/ppo/UnitreeH1_Standing-20240722104334.png-loss.png?raw=true)

![GIF file](https://github.com/JinbiaoZhu/Unitree_H1_Webots/blob/main/controllers/record/ppo/actor_20240722134709.gif?raw=true)

### PlaNet

Developing...

### Dreamer v1

Developing...

### Dreamer v2

Developing...

## 3. Webots Structure

```bash
.
├── basic
│   └── h1_with_hand.urdf
├── controllers
│   ├── base_declaration.py
│   ├── base_network.py
│   ├── buffer.py
│   ├── ckpts
│   ├── continuous_critic.py
│   ├── continuous_policy.py
│   ├── evaluation.py
│   ├── PPO_algo.py
│   ├── PPO_config.py
│   ├── PPO_run.py
│   ├── __pycache__
│   ├── results
│   ├── tools.py
│   └── UnitreeH1StandingEnv.py
├── libraries
├── plugins
│   ├── physics
│   ├── remote_controls
│   └── robot_windows
├── protos
│   └── H1.proto
├── README.md
└── worlds
    └── UnitreeH1_Standing_RLTraining.wbt

12 directories, 15 files
```

The default Webots project template includes the following directories: **controllers** for environment interfaces, algorithm implementations, and saving metrics and models; **libraries** for additional libraries; **plugins** for any plugins used in the simulation; **protos** for describing robot nodes and **worlds** for describing the worlds.

I additionally include: **basic** for storing Unitree’s original `.urdf` files and **README(_en).md** for project documentation.

