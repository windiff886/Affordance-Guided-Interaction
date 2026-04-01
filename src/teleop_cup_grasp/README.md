# Teleoperated Cup Grasping Task (水杯遥操作抓取任务)

## 任务背景 (Background)
本任务在强化学习（RL）训练正式开始前进行，目的是通过人工使用 UI 滑块控制机械臂左臂执行一次抓取动作，从而获取可供基准复现、模仿学习或奖励函数校准的基线参考数据（Demonstration）。水杯在机械臂左臂正前方的固定位置生成，并且放置在一个用于托举的初始无碰撞小平面上。抓住水杯后通过按键消除该平面，进而验证抓取是否成功。

## 任务目标 (Objectives)
1. **场景搭建**: 在 Isaac Sim 环境中加载机械臂模型，并在其左臂正前方固定位置生成水杯和一个托举用的小平面。
2. **UI 控制**: 提供一个 UI 界面（包含滑块），操作者可通过调整滑块来控制机械臂夹爪的末端位姿及开合度，使其移动并抓住水杯。
3. **数据记录**: 程序需具备后台录制能力，以设定的频率自动记录操作过程的详细内容（时间、关节状态、末端姿态、抓握命令以及水杯位姿）。
4. **按键交互**: 当操作者认为已成功抓取水杯时，输入特定的按键命令触发逻辑：将水杯下方的小平面隐身（或取消物理碰撞），结束记录并将完整交互过程保存至文件（供未操作状态下的程序自动复现）。

## 系统架构与实现细节 (Implementation Details)

### 1. 场景配置 (`scene_setup`)
*   **Robot**: 导入相关的机器人模型（如 Z1-Dingo 或是当前的主机体），聚焦于其左侧机械臂和夹爪组件。
*   **Object**: 使用 Isaac Sim 的基础形状或特定的 USD 资产生成水杯。位置依据其相对于机器人基座的坐标静态设定。
*   **Support Plane**: 在水杯正下方生成平面支撑物。

### 2. 用户界面滑块 (`teleop_ui`)
*   利用 Isaac Sim 内部的 `omni.ui` 模块开发悬浮面板。
*   提供如“末端前伸 (Reach)”、“高度调整 (Height)”和“夹爪开合 (Gripper Closed)”等相关的 `FloatSlider` 控制参数。在后端绑定至 Inverse Kinematics (IK) 解算器，计算对应的各关节目标角度。

### 3. 数据记录模块 (`data_recorder`)
*   挂载 `SimulationContext.add_physics_callback` 周期性回调。
*   每次物理步进时将感兴趣的观测维度封装入特定的字典中，如：
    ```json
    {
      "time": 0.0,
      "joint_positions": [...],
      "end_effector_pose": [...],
      "slider_commands": [...],
      "cup_pose": [...]
    }
    ```

### 4. 键盘事件监听 (`keyboard_listener`)
*   使用 `omni.appwindow.get_default_app_window().get_keyboard()` 捕获特定按下的按键（例如 `Space` 或 `Enter` 键）。
*   一旦按键被按下：
    *   取消 Support Plane 的物理碰撞（如关闭其物理 API 状态）。
    *   将收集到的交互序列保存为 `grasp_demo.json` 或 `grasp_demo.npz` 文件。
    *   输出 “Recording Saved” 提示。

## 待确定的设定 (Open Questions / Configurations)
- **输入映射边界：** 滑块的滑动区间物理限度分别是多少（比如向前滑动的最大米数）？
- **动作频度：** 记录轨迹数据的评率（例如是以控制的频率 50Hz 记录，还是物理渲染的更高频率记录）？

---
*Created per request for pre-training setup planning.*
