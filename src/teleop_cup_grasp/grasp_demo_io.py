from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GraspDemo:
    t: np.ndarray
    q_arm: np.ndarray
    q_gripper: np.ndarray
    joint_names: np.ndarray
    cup_world_pos: np.ndarray
    cup_world_quat_wxyz: np.ndarray
    robot_initial_q: np.ndarray
    support_center_world: np.ndarray
    support_scale: np.ndarray
    remove_support_time: float

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=np.float64)
        q_arm = np.asarray(self.q_arm, dtype=np.float64)
        q_gripper = np.asarray(self.q_gripper, dtype=np.float64)
        joint_names = np.asarray(self.joint_names)
        cup_world_pos = np.asarray(self.cup_world_pos, dtype=np.float64)
        cup_world_quat_wxyz = np.asarray(self.cup_world_quat_wxyz, dtype=np.float64)
        robot_initial_q = np.asarray(self.robot_initial_q, dtype=np.float64)
        support_center_world = np.asarray(self.support_center_world, dtype=np.float64)
        support_scale = np.asarray(self.support_scale, dtype=np.float64)

        if t.ndim != 1:
            raise ValueError("t must be 1D")
        if q_arm.ndim != 2 or q_arm.shape[1] != 6:
            raise ValueError("q_arm must have shape (T, 6)")
        if q_arm.shape[0] != t.shape[0]:
            raise ValueError("q_arm length must match t")
        if q_gripper.shape != t.shape:
            raise ValueError("q_gripper length must match t")
        if joint_names.shape != (7,):
            raise ValueError("joint_names must have shape (7,)")
        if cup_world_pos.shape != (3,):
            raise ValueError("cup_world_pos must have shape (3,)")
        if cup_world_quat_wxyz.shape != (4,):
            raise ValueError("cup_world_quat_wxyz must have shape (4,)")
        if robot_initial_q.shape != (7,):
            raise ValueError("robot_initial_q must have shape (7,)")
        if support_center_world.shape != (3,):
            raise ValueError("support_center_world must have shape (3,)")
        if support_scale.shape != (3,):
            raise ValueError("support_scale must have shape (3,)")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "q_arm", q_arm)
        object.__setattr__(self, "q_gripper", q_gripper)
        object.__setattr__(self, "joint_names", joint_names.astype(str))
        object.__setattr__(self, "cup_world_pos", cup_world_pos)
        object.__setattr__(self, "cup_world_quat_wxyz", cup_world_quat_wxyz)
        object.__setattr__(self, "robot_initial_q", robot_initial_q)
        object.__setattr__(self, "support_center_world", support_center_world)
        object.__setattr__(self, "support_scale", support_scale)
        object.__setattr__(self, "remove_support_time", float(self.remove_support_time))


def save_grasp_demo_npz(path: str | Path, demo: GraspDemo) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=demo.t,
        q_arm=demo.q_arm,
        q_gripper=demo.q_gripper,
        joint_names=demo.joint_names,
        cup_world_pos=demo.cup_world_pos,
        cup_world_quat_wxyz=demo.cup_world_quat_wxyz,
        robot_initial_q=demo.robot_initial_q,
        support_center_world=demo.support_center_world,
        support_scale=demo.support_scale,
        remove_support_time=np.array(demo.remove_support_time, dtype=np.float64),
    )


def load_grasp_demo_npz(path: str | Path) -> GraspDemo:
    with np.load(Path(path), allow_pickle=False) as data:
        return GraspDemo(
            t=data["t"],
            q_arm=data["q_arm"],
            q_gripper=data["q_gripper"],
            joint_names=data["joint_names"],
            cup_world_pos=data["cup_world_pos"],
            cup_world_quat_wxyz=data["cup_world_quat_wxyz"],
            robot_initial_q=data["robot_initial_q"],
            support_center_world=data["support_center_world"],
            support_scale=data["support_scale"],
            remove_support_time=float(data["remove_support_time"]),
        )


def sample_joint_targets(demo: GraspDemo, query_time: float) -> tuple[np.ndarray, float]:
    query_time = float(np.clip(query_time, demo.t[0], demo.t[-1]))
    if query_time <= demo.t[0]:
        return demo.q_arm[0].copy(), float(demo.q_gripper[0])
    if query_time >= demo.t[-1]:
        return demo.q_arm[-1].copy(), float(demo.q_gripper[-1])

    idx = int(np.searchsorted(demo.t, query_time, side="right"))
    left = idx - 1
    right = idx
    t0 = demo.t[left]
    t1 = demo.t[right]
    alpha = 0.0 if t1 == t0 else (query_time - t0) / (t1 - t0)
    q_arm = (1.0 - alpha) * demo.q_arm[left] + alpha * demo.q_arm[right]
    q_gripper = (1.0 - alpha) * demo.q_gripper[left] + alpha * demo.q_gripper[right]
    return q_arm, float(q_gripper)


def due_events(
    prev_time: float,
    cur_time: float,
    event_times: dict[str, float | None],
) -> list[str]:
    lower = min(prev_time, cur_time)
    upper = max(prev_time, cur_time)
    return [
        name
        for name, event_time in sorted(event_times.items(), key=lambda item: item[1] if item[1] is not None else np.inf)
        if event_time is not None and lower < float(event_time) <= upper
    ]
