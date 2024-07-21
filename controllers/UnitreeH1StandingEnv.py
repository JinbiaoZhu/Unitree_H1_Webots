from __future__ import annotations

import math
from typing import Dict, Any, Tuple, List

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, RenderFrame
from numpy import ndarray, dtype

from controller import Supervisor, Motor, PositionSensor, GPS


# 奖励函数设计区域
def reward_function_v0(state=None, action=None, next_state=None):
    """
    奖励函数设置: 密集奖励函数
    希望 Unitree H1 的头部机器人施加动作后头部高度会上升且接近于自身高度值(假定是 1.75m )
    --> 头部距离 1.75m 越远, 获得的奖励就越小
    --> reward = -1 * | next_state的最后一维度 - 1.75米 |
    """
    return -1 * math.exp(math.fabs(next_state[-1] - 1.75))


class UnitreeH1StandingV0(gymnasium.Env):
    # 这个元数据字典用于存储类内全局和仿真器连接的超参数
    metadata = {
        "episode_length": 100,
        "p": 1,
        "timestep": 1,
        "threshold": math.pi * 2 / 100,
        "fly": 3.0,
    }

    # 初始化在任务中需要用到的关节角度名称
    joint_namelist = [
        # 左下肢部分
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_joint",
        # 右下肢部分
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_joint",
        # 中间部分
        "torso_joint",
        # 左上肢部分
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_hand_joint",
        # 右上肢部分
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_hand_joint"
    ]

    # 初始化在任务中需要用到的部位名称
    body_namelist = [
        "head"
    ]

    # 初始化传感器后缀字符串
    sensor_suffix = "_sensor"

    def __init__(self):
        # 初始化一个 "超级监督者" 用于读取/设置整个仿真器的信息
        self.Supervisor = Supervisor()

        # 获取 Unitree H1 全部 ["关节名称" - ["关节名称, 关节控制句柄, 角度最小值, 角度范围"]] 作为键值对的字典
        self.joint_infos = self.get_joint_infos()

        # 获取 Unitree H1 的头部三维位置传感器, 用于计算奖励函数
        self.body_infos = self.get_body_infos()

        # 获取状态空间相关的信息: 状态空间 = {各个关节的当前时刻角度信息, 各 1 维} + {头部传感器的当前时刻位置信息, 各 3 维}
        jnt_obs_shape = sum([self.joint_infos[name]["shape"] for name in self.joint_namelist])
        bdy_obs_shape = sum([self.body_infos[name]["shape"] for name in self.body_namelist])
        self.observation_space = spaces.Box(-1.0 * math.inf, math.inf, (jnt_obs_shape + bdy_obs_shape,))

        # 获取动作空间的相关信息: 动作空间是各个关节的角度增量, 弧度制, 范围是 [-1, 1] 之间, 便于神经网络输出
        self.action_space = spaces.Box(-1.0, 1.0, (jnt_obs_shape,))

        # 获取奖励函数, 从外部获得
        self.reward_function = reward_function_v0

        # 初始化每一时刻的步数记录
        self.current_step = 0

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> ndarray[Any, dtype[Any]]:
        # 使用 gym.Env 环境基类提供的随机数生成器, 需要调用以下代码以确保 seed 正确
        super().reset(seed=seed)

        # 监视器重启仿真环境
        self.Supervisor.simulationReset()

        # 初始化时间步骤记录
        self.current_step = 0

        # 初始化智能体的行为, 具体在各个电机中赋值运动范围内的随机数
        for name, properties in self.joint_infos.items():
            properties["controller"].setPosition(
                self.np_random.random() * (properties["controller_max"] - properties["controller_min"]) + properties[
                    "controller_min"])
            self.Supervisor.step(time_step=self.metadata["timestep"])

        # 调用 .get_obs() 方法获取当前关节位置和头部三维位置并返回
        return np.asarray(self.get_obs())

    def step(
            self, action: ActType
    ) -> Tuple[List[Any], List[Any], float, bool, Dict[str, int | Any]]:

        modified_action = []
        for an_action, jnt_name in zip(action, self.joint_namelist):
            # 对 an_action 原本是 [-1, 1] 范围进行放缩
            an_action *= self.metadata["threshold"]
            # 将放缩完的动作存储至 modified_action 列表中, 用于存放至经验回访池中
            modified_action.append(an_action)
            # 读取当前 jnt_name 关节的数值
            current = self.joint_infos[jnt_name]["sensor"].getValue()
            # 增量式计算
            current += an_action
            # 对关节角度进行限幅
            current = self.clip(current, jnt_name)
            # 设置关节角度值
            self.joint_infos[jnt_name]["controller"].setPosition(current)
        # 获取施加完动作的下一个状态观测
        next_obs = self.get_obs()
        # 更新时间步骤
        self.current_step += 1
        # 获得奖励函数, 这里设置密集奖励函数
        reward = self.reward_function(None, modified_action, next_obs)
        # 获取是否自动结束的标志
        done = self.get_done()
        # 获取当前时间步骤相关的信息
        info = self.get_info(reward=reward)

        return modified_action, next_obs, reward, done, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """
        Webots 自带的渲染器渲染, 目前此处暂时不开发
        这个 .render() 方法不会出现在循环体内, 会出现在 while 循环条件内
        :return: None
        """
        # 进行仿真环境的渲染
        render_result = self.Supervisor.step(time_step=self.metadata["timestep"])
        return render_result

    def close(self):
        """
        关闭环境: 底层实现是重载整个世界
        """
        self.Supervisor.worldReload()
        pass

    def get_joint_infos(self) -> Dict:
        """
        获取用于本控制任务的关节信息, 包括关节名字、关节控制器、关节传感器和关节值上下限.
        :return: 字典格式数据, 键是关节名字, 值是对应的属性字典.
        """
        # 初始化一个用于汇总所有信息的空字典
        temp_dict = {}
        # 根据 joint_namelist 内的命名顺序填充字典
        for name in self.joint_namelist:
            # 初始化一个空字典用于保存当前关节的所有信息: 关节名, 关节控制器, 控制下限, 控制上限, 关节传感器
            # 注意: 根据 .proto 文件所述, 所有关节传感器名字 = 关节控制器名字 + "_sensor" 后缀组成
            controller = self.Supervisor.getDevice(name)
            assert isinstance(controller, Motor), "Wrong"
            min_pos, max_pos = controller.getMinPosition(), controller.getMaxPosition()
            sensor = self.Supervisor.getDevice(name + self.sensor_suffix)
            assert isinstance(sensor, PositionSensor), "Wrong"
            # 注意: 实例化关节传感器后需要对传感器使能
            sensor.enable(p=self.metadata["p"])
            infos = {
                "name": name,
                "controller": controller,
                "controller_min": min_pos, "controller_max": max_pos,
                "sensor": sensor,
                "shape": 1
            }
            # 将 infos 字典汇总到 temp_dict 中
            temp_dict[name] = infos

        return temp_dict

    def get_body_infos(self):
        """
        获取用于本控制任务的部位信息, 包括部位名字、部位的位置传感器.
        :return: 字典格式数据, 键是部位名字, 值是对应的属性字典.
        """
        # 初始化一个用于汇总所有信息的空字典
        temp_dict = {}
        # 根据 joint_namelist 内的命名顺序填充字典
        for name in self.body_namelist:
            # 初始化一个空字典用于保存当前部位的所有信息: 部位名, 部位位置传感器
            gps = self.Supervisor.getDevice("head")
            assert isinstance(gps, GPS), "Wrong"
            gps.enable(p=self.metadata["p"])
            infos = {"name": name, "sensor": gps, "shape": 3}
            temp_dict[name] = infos

        return temp_dict

    def get_obs(self):
        """
        获取观测信息, 也就是各个关节传感器的信息, 头部部位位置信息
        :return: 一个列表格式数据, 前半部分是各个关节的当前值, 顺序是 joint_namelist ;
        后半部分是位置传感器读取的数值, 顺序是 body_namelist ;
        """
        tmp_dict = []
        for name in self.joint_namelist:
            tmp_dict.append(self.joint_infos[name]["sensor"].getValue())
        for name in self.body_namelist:
            tmp_dict += self.body_infos[name]["sensor"].getValues()
        return tmp_dict

    def get_done(self):
        """
        获取一个 episode 是否结束的标志, 用于强化学习自动结束一个 episode 数据收集
        自动结束的标志有两个:
        1. 当前的时间步骤大于既定的 episode 长度
        2. Unitree H1 机器人飞离地面 3 米, 这是因为关节角的变化和力矩的施加导致机器人会弹出去
        """
        step_varification = self.current_step >= self.metadata["episode_length"]
        # 获取机器人中心
        robot_center = self.Supervisor.getFromDef("H1")
        trans_field = robot_center.getField("translation")
        z_verification = trans_field.getSFVec3f()[2] > self.metadata["fly"]
        return step_varification or z_verification

    def get_info(self, **kwargs):
        """
        获取当前时间步骤额外的信息
        :param kwargs: 需要获取的信息, 此处默认先获取当前时刻的步数和奖励, 可后续继续开发
        :return: 字典格式数据
        """
        info_dict = {"current": self.current_step}
        if "reward" in kwargs.keys():
            info_dict["reward"] = kwargs["reward"]
        else:
            pass
        return info_dict

    def clip(self, value, jnt_name):
        """
        根据 jnt_name 字符串检索 self.joint_infos 属性内的关节范围, 并对当前数值进行限幅
        :param value: 当前关节数值
        :param jnt_name: 特定的关节字符串
        :return: 限幅后的关节数值
        """
        minValue, maxValue = self.joint_infos[jnt_name]["controller_min"], self.joint_infos[jnt_name]["controller_max"]
        if value <= minValue:
            value = minValue * 0.999  # 0.999 的目的是避免 value 恰好等于临界值时候产生的警告
        elif value >= maxValue:
            value = maxValue * 0.999  # 0.999 的目的是避免 value 恰好等于临界值时候产生的警告
        else:
            value = value
        return value


if __name__ == "__main__":

    env = UnitreeH1StandingV0()

    for _ in range(10):

        obs = env.reset()

        while env.render() != -1:
            action = env.action_space.sample()

            modified_action, next_obs, reward, done, info = env.step(action)
            print(info)

            if done:
                break

    env.close()
