"""环境抽象基类 — 定义单环境的标准接口约定。

所有具体环境（如 DoorInteractionEnv）继承本基类，实现 reset / step / close。

接口数据流：
    reset(params)  → (actor_obs_dict, critic_obs_dict)
    step(action)   → (actor_obs_dict, critic_obs_dict, reward, done, info)

其中：
    - actor_obs_dict  结构由 ActorObsBuilder.build() 定义
    - critic_obs_dict 结构由 CriticObsBuilder.build() 定义
    - action ∈ R^12   是绝对物理力矩（N·m），由策略直接输出
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EnvConfig:
    """环境配置参数。

    汇集仿真步长、控制频率、判定阈值等所有环境级超参。
    """

    # ── 仿真参数 ─────────────────────────────────────────────────────
    # 物理引擎积分步长（秒）
    physics_dt: float = 1.0 / 120.0
    # 策略决策间隔 = decimation × physics_dt
    decimation: int = 2
    # 单 episode 最大策略步数
    max_episode_steps: int = 500

    # ── 任务判定参数 ──────────────────────────────────────────────────
    # episode 成功终止角度阈值（rad）
    door_angle_target: float = 1.57
    # 杯体脱落检测距离（m）
    cup_drop_threshold: float = 0.15

    # ── 接触过滤参数 ──────────────────────────────────────────────────
    # 接触力过滤阈值（N），低于此值的微碰撞被忽略
    contact_force_threshold: float = 0.1

    # ── 机器人规格 ────────────────────────────────────────────────────
    # 单臂关节数
    joints_per_arm: int = 6
    # 双臂总关节数
    total_joints: int = 12
    # 各关节力矩安全限幅（N·m），长度 = total_joints
    # None 时由 URDF 读取；此处给出合理默认值
    effort_limits: np.ndarray | None = None

    # ── 奖励配置 ──────────────────────────────────────────────────────
    # 完整的 reward 配置字典，传给 RewardManager
    # 默认值对齐 rewards/README.md 中定义的推荐超参
    reward_cfg: dict = field(default_factory=lambda: {
        "task": {
            "w_delta": 10.0,
            "alpha": 0.3,
            "k_decay": 0.5,
            "w_open": 50.0,
            "theta_target": 1.2,
        },
        "stability": {
            "w_zero_acc": 1.0,
            "lambda_acc": 2.0,
            "w_zero_ang": 0.5,
            "lambda_ang": 1.0,
            "w_acc": 0.5,
            "w_ang": 0.3,
            "w_tilt": 0.3,
            "w_smooth": 0.1,
            "w_reg": 0.01,
        },
        "safety": {
            "beta_self": 5.0,
            "beta_limit": 1.0,
            "mu": 0.1,
            "beta_vel": 0.5,
            "beta_torque": 0.01,
            "w_drop": 100.0,
        },
        "scaling": {
            "s_min": 0.1,
            "n_anneal": 500_000,
        },
    })

    # ── 观测配置 ──────────────────────────────────────────────────────
    action_history_length: int = 3
    acc_history_length: int = 10

    @property
    def control_dt(self) -> float:
        """策略决策步长（秒）。"""
        return self.physics_dt * self.decimation

    def get_effort_limits(self) -> np.ndarray:
        """获取关节力矩限幅数组。"""
        if self.effort_limits is not None:
            return self.effort_limits
        # 默认限幅：Z1 机械臂典型值
        return np.array(
            [33.5, 33.5, 33.5, 33.5, 33.5, 33.5,   # 左臂 6 关节
             33.5, 33.5, 33.5, 33.5, 33.5, 33.5],   # 右臂 6 关节
            dtype=np.float64,
        )


class BaseEnv(ABC):
    """仿真环境抽象基类。

    子类必须实现 ``reset()``、``step()``、``close()`` 三个方法。
    """

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()

    @abstractmethod
    def reset(
        self,
        *,
        domain_params: dict[str, Any] | None = None,
        door_type: str = "push",
        left_occupied: bool = False,
        right_occupied: bool = False,
    ) -> tuple[dict, dict]:
        """重置环境，返回初始观测。

        Parameters
        ----------
        domain_params : dict | None
            来自 DomainRandomizer.sample_episode_params() 的物理参数。
            包含 cup_mass, door_mass, door_damping, base_pos。
        door_type : str
            门类型，由 CurriculumManager 决定。
        left_occupied : bool
            左臂是否持杯。
        right_occupied : bool
            右臂是否持杯。

        Returns
        -------
        actor_obs : dict
            由 ActorObsBuilder.build() 生成的 actor 观测字典。
        critic_obs : dict
            由 CriticObsBuilder.build() 生成的 critic 观测字典。
        """
        ...

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> tuple[dict, dict, float, bool, dict[str, Any]]:
        """执行一步仿真。

        Parameters
        ----------
        action : (12,) ndarray
            绝对物理力矩（N·m），策略直接输出。
            环境层仅做 effort limit 截断。

        Returns
        -------
        actor_obs : dict
        critic_obs : dict
        reward : float
        done : bool
        info : dict
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """释放仿真资源。"""
        ...

    @property
    def action_dim(self) -> int:
        """动作空间维度。"""
        return self.cfg.total_joints
