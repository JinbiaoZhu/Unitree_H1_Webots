# Training Unitree H1 standing by reinforcement learning in Webots simulator

## 0. Introduction

这个项目使用强化学习 (PPO) 训练宇树 H1 机器人实现**在不同姿态作为初始状态下**完成**竖直站立**任务.

## 1. Markov Decision Process

在建模马尔科夫决策过程中, 使用宇树 H1 机器人的 21 个电机关节, 如下表所示.

| 躯体部分   | 关节名                     |                           |                          |                   |                   |
| ---------- | -------------------------- | ------------------------- | ------------------------ | ----------------- | ----------------- |
| 左下肢部分 | left_hip_yaw_joint         | left_hip_roll_joint       | left_hip_pitch_joint     | left_knee_joint   | left_ankle_joint  |
| 右下肢部分 | right_hip_yaw_joint        | right_hip_roll_joint      | right_hip_pitch_joint    | right_knee_joint  | right_ankle_joint |
| 中间部分   | torso_joint                |                           |                          |                   |                   |
| 左上肢部分 | left_shoulder_pitch_joint  | left_shoulder_roll_joint  | left_shoulder_yaw_joint  | left_elbow_joint  | left_hand_joint   |
| 右上肢部分 | right_shoulder_pitch_joint | right_shoulder_roll_joint | right_shoulder_yaw_joint | right_elbow_joint | right_hand_joint  |

在建模马尔科夫决策过程中, 使用宇树 H1 机器人 1 个部位实体, 如下表所示.

| 躯体部位 | 名称                          | 备注                                 |
| -------- | ----------------------------- | ------------------------------------ |
| 头部     | head (在 `.proto` 文件中显示) | 距离 `pelvis_visual` 的高度是 0.7 米 |

### 状态空间

状态空间包括两部分: 上述表格中各个关节的当前时刻角度信息, 各 1 维; 头部传感器的当前时刻 3 维位置信息; 总共 24 维度.

### 动作空间

动作空间是上述表格中各个关节的角度增量, 采用弧度制, 范围是 $[-\Delta\theta, \Delta\theta]$ 之间. 这里 $\Delta\theta$ 的含义是关节角度增量的最大值, 即对每个关节的角度增量都不容许超过 $[-\Delta\theta, \Delta\theta]$ 范围. 关节增量表达式: $\theta_{t+1}=\theta_{t}+\Delta\theta$ .

在设计的神经网络中, 最后层的激活函数是 `softmax` 激活函数, 将网络输出的均值 $\mu$ `mean` 和方差 $\sigma$  `std` 数值限制在 $[-1,1]$ 之间, 随后方差 $\sigma$ 经过指数 $e$ 计算后得到最终的方差, 由此构建正态分布是 $\alpha\sim N(\mu,\sigma^{2})$ . 为了得到范围是 $[-\Delta\theta, \Delta\theta]$ 之间的正态分布, 这里对正态分布进行线性放缩: $\alpha^{\prime}=\Delta\theta\cdot\alpha\sim N(\Delta\theta\mu,\Delta\theta^{2}\sigma^{2})$ .

### 奖励函数

目前设计的奖励函数是 `v0` 版本: 希望宇树 H1 的头部机器人施加动作后**头部高度会上升**且**接近于自身高度值** ( 1.75 米) , 因此头部距离 1.75 米越远, 获得的奖励就越小. 由此得到奖励函数 $\text{reward} = -1\times|z - 1.75|^{2}$ . 最后在调试中做了一些限幅处理: 当头部距离 1.75 米超过 10 米时, 获得的奖励函数低于 -100 数值, 就将奖励限制在 -100 这个值, 更远就会影响学习效果.

### Episode 结束标志

获取一个 episode 是否结束的标志, 用于强化学习自动结束一个 episode 数据收集. 自动结束的标志有两个: 1. 当前的时间步骤大于既定的 episode 长度; 2. 宇树 H1 机器人飞离地面 3 米, 这是因为关节角的变化和力矩的施加导致机器人会 "弹" 出去.

## 2. 算法

### PPO 算法

代码来源: [我自己 2023 年写的](https://github.com/JinbiaoZhu/BasicReinforcementLearning.git) .

效果表现和损失函数展示:

![Total Reward Per Episode](https://github.com/JinbiaoZhu/Unitree_H1_Webots/tree/main/controllers/results/ppo/UnitreeH1_Standing-20240722122807.png)

---

![Loss function Per Episode](https://github.com/JinbiaoZhu/Unitree_H1_Webots/tree/main/controllers/results/ppo/UnitreeH1_Standing-20240722104334.png-loss.png)

---



### PlaNet

待开发

### Dreamer v1

待开发

### Dreamer v2

待开发

## 3. Webots 项目结构

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

Webots 默认项目模板是: `controllers` (环境接口、算法实现和指标与模型的保存) , `libraries` , `plugins` , `protos` (用于描述 Webots 中的机器人节点) 和 `worlds` (描述 Webots 中世界) 文件; `basic` (存放宇树原本的 `.urdf` 文件) 和 `README.md` 是我额外加的.

