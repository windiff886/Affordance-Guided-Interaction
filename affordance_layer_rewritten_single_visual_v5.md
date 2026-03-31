# 基于 Affordance 引导的双臂持杯门交互：问题定义与技术方案

## 1. 我们要解决什么问题

### 1.1 问题背景

现有很多 manipulation 工作默认：机器人主要通过 **end-effector (EE)**，也就是 gripper / hand，直接与环境交互完成任务。这种设定对于抓取、放置、按按钮等任务是自然的，但在更真实的场景中，这种设定存在一个明显限制：

> 当末端已经被占用时，机器人是否仍然能够利用身体其他区域完成环境交互，并同时保持已持物体的稳定？

这正是本文希望研究的问题。

我们关注的场景是：在 **Isaac Sim** 中使用一个由 **双臂 Z1 + RealSense 深度相机** 组成的双臂操作平台，研究机器人在**持杯**与**非持杯**两种上下文下，如何完成与“门”有关的交互任务。本文在当前实验中只关注**双臂操作能力**以及来自 **RealSense 相机** 的 RGB-D 观测。

在这一设定下，本文关心的不只是“某个单臂末端如何接触门”，而是：当一侧操作臂末端已经持杯时，系统能否综合利用**另一侧操作臂**以及**非持杯侧局部 link**，完成 door-related interaction。但这里的“开门”不应被过早简化为“固定的 push door”。因为在真实环境中，开门方式本身是多样的，例如：

- 某些门可以直接推动门板；
- 某些门需要先按压按钮，再推动门；
- 某些门需要触碰或下压门把手；
- 某些门的有效作用区域不是整块门板，而是局部的可操作区域。

因此，我们真正关心的问题不是：

> “如何训练一个策略去推某一类固定的门？”

而是：

> “机器人能否在多种 door-related interaction 中，根据环境的 **affordance** 以及自身当前的身体约束，自主形成合适的交互方式？”

这意味着，问题的核心不再只是 motion control，也不只是 door opening，而是：

> **affordance-guided embodied interaction under end-effector occupancy constraints**

也就是说，系统既要理解环境“哪里可以做什么”，又要理解自己“当前适合用哪里去做”。

---

### 1.2 我们真正想研究的核心科学问题

这个工作本质上不是一个简单的固定门任务 benchmark，而是一个关于 **object affordance、机械臂隐式可作用性表征，以及约束感知控制** 的统一问题。

更具体地说，我们关心以下几个核心问题：

#### （1）门交互本身具有多样的环境 affordance
“开门”不是单一动作，而是一组可能的交互方式。例如：

- pushing affordance：门板某些区域可直接推动；
- pressing affordance：按钮区域可按压；
- handle-affordance：某些把手区域可触碰、下压或施加力；
- sequential affordance：必须先完成某个局部动作，再完成下一步门交互。

因此，系统不能只学“门在哪里”，而需要学：
- 当前环境中有哪些候选交互区域；
- 每个区域支持什么类型的交互；
- 在当前任务目标下，哪类交互更值得优先尝试。

这与 **Where2Act** 一类工作的核心精神是一致的：学习对象局部区域“哪里可以 act，以及 act 之后会产生什么效果”。

#### （2）机械臂自身的 affordance 更可能以内隐方式存在于 policy 中
相比 object affordance 已经有较多工作，机械臂本体的 affordance 研究仍然明显不足。尤其在“末端被占用、还要满足稳定性约束”的场景中，我们通常没有一个现成的标签体系去显式告诉系统：

- 哪个 link 在什么情况下适合承担主要接触；
- 哪个局部区域在持杯时更适合与环境交互；
- 在不同任务目标和不同约束下，身体各部分的交互价值如何变化。

因此，这里的关键不只是学习动作映射，而是让 policy 在交互过程中逐渐形成一种 **implicit body-affordance representation**，用于判断：
- 当前哪些身体局部更适合接近某类 affordance 区域；
- 当前哪些接触方式会带来更大的末端扰动；
- 当前在任务成功与持物稳定之间，哪种接触分配更优。

#### （3）持杯稳定性是任务约束的一部分，而不是附属要求
在很多 manipulation 任务中，“手里拿着东西”只是状态的一部分；  
但在这里，持杯意味着系统必须同时满足：

- 任务要完成；
- 杯体尽量保持稳定；
- 末端姿态变化不能太剧烈；
- 接触冲击不能通过机械臂传导到杯体造成明显晃动。

因此，我们的目标不是单纯 task completion，而是：

> **constraint-aware interaction success**

即：在上肢稳定性约束下，完成由 affordance 决定的环境交互。


### 1.3 我们希望建立的研究范式

基于上述分析，本文希望提出的不是一个 narrowly-defined push-door method，而是一个更一般的研究范式：

> 在 door-related interaction 场景中，先利用 affordance 学习或 affordance 表示给出环境中的候选交互区域及其交互类型，再结合机器人当前的末端占用状态、稳定性要求和身体几何条件，学习一个能够执行这些交互的策略。

这个范式的核心在于区分两件事：

#### （1）环境 affordance 是显式提供的任务语义
它描述对象局部“哪里可以做什么”，例如：
- 哪些区域可推动；
- 哪些区域可按压；
- 哪些区域可触碰 / 下压；
- 哪些区域属于顺序交互中的关键步骤。

#### （2）机械臂 affordance 不预先写成显式规则
我们不预先规定“持杯时必须用哪个部位接触”，而是让策略在交互和优化过程中，自己形成对身体可作用性的判断。换句话说，环境 affordance 提供的是外部任务线索，而机械臂 affordance 更可能通过 policy 的内部表示被隐式学习出来。

因此，本文想建立的不是一个“把某一种门推开”的专用方法，而是一个更一般的：

> **affordance-conditioned door interaction framework**

在这个框架中，系统面对的不是单一的 push-door 场景，而是一组 door-related interaction：
- push 门板；
- press 按钮；
- 作用于把手或压杆；
- 按照一定顺序完成局部触发与后续开门。

这样，研究重点就从“为某个门实例拟合动作”转向：

> “在环境 affordance 已知或可预测的前提下，让策略根据当前身体资源与任务约束，自主决定如何完成交互。”


### 1.4 本文的问题定义

因此，本文最终要研究的问题可以定义为：

> 在 Isaac Sim 中，针对一个由 **双臂 Z1 与 RealSense 深度相机** 组成的双臂机器人平台，和一组具有不同交互方式的 door-related objects，构建一个 affordance-guided interaction framework。系统首先表示或预测环境中的候选交互 affordance（如 push region、press region、handle region 等），再结合机器人当前的末端占用状态、持杯稳定性约束，以及双臂的可达性条件，学习一个统一的策略去执行这些交互；当某个操作末端空闲时，系统可以自然使用该 gripper；当一侧末端持杯时，系统应尽量抑制持杯侧末端剧烈扰动，并根据双臂几何与接触可达性，选择更合适的执行臂或身体局部区域完成交互。

这里强调一点：本文并不要求预先显式规定“持杯时必须用肘部”或“必须用前臂”。相反，我们希望系统在 affordance 与约束共同作用下，自然学出合理的身体接触分配。

---

## 2. 我们要如何解决这个问题

## 2.1 总体思路

我们的总体方法是：

> 构建一个 **affordance-guided interaction architecture**。  
> 上层接收原始 **RGB-D** 观测，先从视觉输入中提取门相关点云，再通过视觉 encoder 将其压缩为统一的视觉表征；下层再结合机器人当前状态、末端占用情况和持杯稳定性信息，输出具体动作。

在当前平台设定下，原始视觉输入具体来自 **RealSense RGB-D 相机**；而“机器人当前状态”主要由**双臂状态**构成。因此，本文的整体框架在当前实验中是面向**双臂操作平台**而非固定底座单臂来定义的。

这样做的目的，是避免把问题过早收缩成“为某一种门显式写规则的控制器设计”，而是建立一套能够覆盖多种 door-related interaction 的统一视觉表示与控制框架。

为了让结构更清晰，我们将整体方案分成两个层面：

### 第一层：Affordance 视觉表征层
这一层以原始 **RGB-D** 观测作为输入，完成以下处理：

1. 从 RGB 图像中识别门相关区域；
2. 结合深度图反投影得到门相关点云；
3. 将门点云输入视觉 encoder，得到统一的视觉表征。

我们将这一层的输出记为：

- $z_{\text{aff},t}$：当前时刻的统一视觉 affordance 表征。

这里的 $z_{\text{aff},t}$ 不再被拆分为“任务进展表示”和“对象 affordance 表示”两个组分，而是作为一个单一的视觉 latent，统一编码当前场景中与门交互有关的几何与外观信息。

### 第二层：Constraint-aware 执行层
这一层接收：
- 机器人自身状态；
- 末端是否被占用；
- 当前稳定性要求；
- 来自上层的统一视觉 affordance 表征。

然后输出真正的控制动作。

整个系统的关键是：

> 上层负责把原始 RGB-D 观测转换为可供控制使用的门相关视觉表征，  
> 下层负责在身体约束存在时，把这一视觉表征变成可执行动作。


## 2.2 问题建模：Affordance-Conditioned Interaction MDP

我们将任务建模为一个包含上下文和视觉 affordance 条件的决策问题：

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{C}, r, \gamma)
$$

其中：

- $\mathcal{S}$：机器人状态、环境局部几何以及交互相关观测；
- $\mathcal{A}$：控制动作；
- $\mathcal{C}$：任务上下文；
- $r$：奖励函数；
- $\gamma$：折扣因子。

这里的重点不是显式建模环境动力学，而是学习一个在视觉表征和交互反馈基础上优化得到的策略。

我们定义的 context $c \in \mathcal{C}$ 主要包括：

- `occupied ∈ {0,1}`：末端是否持杯；
- `task_goal`：当前交互目标或任务类型。

与前一版不同，这里不再要求上层模块同时显式输出“任务进展表示”和“对象 affordance 表示”。  
相反，上层只负责从原始 **RGB-D** 输入中构造统一的视觉 affordance 表征：

- $z_{\text{aff},t}$：当前时刻的视觉 affordance latent。

于是策略可以写成：

$$
a_t = \pi_\theta(o_t, h_t, c_t, z_{\text{aff},t})
$$

其中：
- $o_t$：当前机器人观测；
- $h_t$：历史隐状态；
- $c_t$：任务上下文；
- $z_{\text{aff},t}$：统一视觉 affordance 表征；
- $a_t$：动作。

这种写法的意义是：

> 策略不需要直接处理高维 RGB-D 或原始点云，  
> 而是基于上层提供的门相关视觉表征，在身体约束下学习如何执行交互。


## 2.3 Affordance 层如何实现

在当前版本中，Affordance 层的功能被重新明确为一条**单一的视觉表征流水线**：

> 接收原始 **RGB-D** 输入，提取其中的门相关点云，再通过视觉 encoder 将该点云编码成统一的视觉表征 $z_{\text{aff}}$，供后续 policy 使用。

这意味着，当前版本的 Affordance 层**不再划分为两个组分**，也不再分别输出 `task-progress representation` 和 `object-affordance representation`。相反，它只输出一个统一的视觉 latent，用于表示当前场景中与 door-related interaction 最相关的几何与视觉信息。

### 2.3.1 当前版本的设计目标

当前版本的 Affordance 层需要满足以下要求：

1. **输入直接来自 RealSense 的原始 RGB-D 观测**；
2. **能够从输入中提取门相关点云**，而不是要求策略直接处理整幅场景点云；
3. **门点云的提取和编码尽量复用现成模型或固定算法**，避免额外训练；
4. **最终输出是一个固定维度的视觉特征向量**，便于直接并入 actor observation。

因此，当前版本的重点不再是把上层做成多个子模块，而是构造一条尽可能直接的：

```text
RGB-D -> door point cloud extraction -> visual encoder -> z_aff
```

### 2.3.2 整体处理流程

在当前版本中，我们推荐将 Affordance 层实现为如下处理管线：

```text
Input:
  RGB_t, Depth_t

Affordance Layer:
  1. Door region detection / segmentation
  2. Depth back-projection
  3. Door point cloud extraction
  4. Point cloud preprocessing
  5. Visual encoder

Output:
  z_aff_t
```

其中：

- 第 1 步负责从 RGB 图像中找到门相关区域；
- 第 2 步负责将对应深度像素反投影为三维点；
- 第 3 步得到门点云；
- 第 4 步对点云做采样与清理；
- 第 5 步将门点云编码成一个固定维度的视觉特征向量 $z_{\text{aff},t}$。

### 2.3.3 从 RGB-D 中提取门点云

在当前版本中，门点云的提取流程如下：

1. 输入当前时刻的 RGB 图像 $RGB_t$ 与深度图 $Depth_t$；
2. 使用现成的开集视觉检测/分割模型，在 RGB 图像中识别 `door` 区域；
3. 根据得到的门 mask，在深度图中保留对应像素；
4. 使用相机内参将这些像素反投影到三维空间，得到门点云 $P_{\text{door},t}$；
5. 对得到的门点云进行简单几何预处理。

在当前版本中，可直接复用的现成模型包括：

- **LangSAM**
- **Grounded-SAM 2**
- **Grounding DINO + SAM / SAM 2**

它们都可以直接下载现成权重，因此符合当前版本“门点云视觉识别任务不需要自行训练”的要求。

如果像素坐标为 $(u,v)$，深度值为 $d$，相机内参为 $(f_x, f_y, c_x, c_y)$，则反投影公式为：

$$
x = (u-c_x)d/f_x, \qquad
y = (v-c_y)d/f_y, \qquad
z = d
$$

由此可得到当前时刻可见的门局部点云，而不是完整的门 CAD 模型。对于本文的交互任务而言，这种“当前视角下可见的门点云”已经足以支持后续控制。

### 2.3.4 门点云预处理

在将门点云送入视觉 encoder 之前，建议先做一轮轻量级预处理，以提升编码稳定性。当前版本可采用如下处理：

- voxel downsampling
- statistical outlier removal
- farthest point sampling 或随机采样
- 统一点数到固定规模（如 256 / 512 / 1024）
- 如有必要，估计法向量并附加为点特征

在这一阶段，本文不要求对门把手、按钮等局部再单独构造独立表征；这些信息可以在后续版本中扩展。当前版本的重点是：

> 从 RGB-D 中稳定地得到门点云，并将其整理成视觉 encoder 可直接消费的标准输入格式。

### 2.3.5 视觉 encoder 的角色

在当前版本中，Affordance 层的最后一步是使用视觉 encoder 对门点云进行编码，得到统一的视觉表征 $z_{\text{aff}}$。

这里的视觉 encoder 作用不是直接输出动作，也不是显式预测不同 affordance 类型，而是把门点云压缩成一个固定维度的 latent，用于表示：

- 当前门的局部几何形状；
- 当前视角下可见的门区域分布；
- 与后续门交互有关的整体视觉结构。

对于视觉 encoder，有两种实现思路：

#### （1）固定几何摘要编码
不额外使用学习型 3D backbone，而是直接从门点云中提取低维几何特征，例如：

- 门点云中心；
- 门点云主方向或平面法向；
- 门点云包围盒尺寸；
- 当前门点云与 active gripper 的相对距离。

然后将这些量拼接为一个固定维度向量，直接作为 $z_{\text{aff}}$。

#### （2）冻结的现成点云 encoder
如果希望保留更丰富的点云表征，则可以使用一个**现成预训练、并在当前任务中冻结使用**的点云 encoder，例如：

- Point-MAE
- ULIP / ULIP-2

此时，门点云经过 encoder 后直接输出 embedding，并作为 $z_{\text{aff}}$ 或其一部分。

在当前版本中，这两种方式都满足“编码任务不需要自己训练”的要求。若以工程可控性为优先，则推荐先从固定几何摘要编码开始；若更强调表征能力，则可以使用冻结的现成点云 encoder。

### 2.3.6 当前版本中 $z_{\text{aff}}$ 的定义

综合上述流程，当前版本的上层输出统一定义为：

$$
z_{\text{aff},t} = Enc_{vis}(P_{\text{door},t})
$$

其中：

- $P_{\text{door},t}$ 表示由当前 RGB-D 观测提取出的门点云；
- $Enc_{vis}(\cdot)$ 表示视觉 encoder；
- $z_{\text{aff},t}$ 表示最终送入 policy 的统一视觉表征。

也就是说，在当前版本中，Affordance 层并不再承担“任务进展估计 + affordance 类型分解”这类多头输出职责，而是只负责：

> **把原始 RGB-D 压缩成一个以门点云为核心的统一视觉特征。**

### 2.3.7 当前版本的整体接口形式

基于以上设计，当前版本的 Affordance 层可以写为：

```text
Input:
  RGB_t, Depth_t
  (from RealSense RGB-D camera)

Affordance Layer:
  1. Door detection / segmentation
  2. Depth back-projection
  3. Door point cloud extraction
  4. Point cloud preprocessing
  5. Visual encoder

Output:
  z_aff_t
```

于是当前版本的设计重点就变为：

- 用现成视觉模型解决“门在哪里”；
- 用深度反投影解决“门在三维中如何表示”；
- 用视觉 encoder 解决“门点云如何变成 policy 可消费的 observation”。

这使得本文当前版本中的 Affordance 层成为一个**单一、直接、且尽量不依赖额外训练**的上层视觉表征模块。


## 2.4 执行策略的输入：只使用现实中相对可获得的信息

### 2.4.1 设计原则

执行策略（actor）的输入必须尽量满足：

1. **部署时可以获得**
2. **不依赖环境隐藏物理参数**
3. **不要求对机械臂人为划分语义部位**
4. **既能表达任务上下文，也能表达稳定性约束**

因此，不适合作为 actor 必需输入的量包括：
- `cup_mass`
- `cup_fill_ratio`
- `door_mass`
- `door_damping`

这些量可以在训练中随机化，或在 critic 中作为 privileged information 使用，但不应要求 deployment 时精确已知。

---

### 2.4.2 Actor 观测组成

#### A. 机器人本体观测（proprioception）
对于当前的双臂平台，机器人本体观测不再只是单臂关节状态，而应至少包括：

- 左臂 Z1 关节位置与速度
- 右臂 Z1 关节位置与速度
- joint torques / motor currents（如果可用）
- previous actions $a_{t-k:t-1}$

#### B. 明确可定义的末端状态
在当前版本中，**优先使用当前执行臂或持杯臂对应的 gripper / end-effector frame 作为主要参考坐标系**。

这样做的原因是：
- 本文关心的稳定性本质上是“杯体是否稳定”，而杯体与持杯 gripper 的关系比与 wrist 更直接；
- gripper frame 更适合表达末端相对重力方向的倾斜；
- 在双臂平台上，显式区分“执行臂”和“持杯臂”有助于把交互任务与稳定性约束分配到正确的末端参考系；
- 如果同时保留 gripper 和 wrist 两套末端位姿与速度表示，会带来较强冗余；
- 对 SoFTA 风格的稳定性奖励而言，参考坐标系本身也是定义的一部分，因此不宜随意将 end-effector frame 替换为 wrist frame。

因此，在当前设定下，低层策略优先接收：
- active gripper pose / velocity
- cup-carrying gripper pose / velocity（当持杯臂与执行臂不同时可单独保留）

如果后续实验发现某些动态 proxy 用 wrist rigid body 更稳定，wrist 信息可以作为可选辅助量加入，但当前版本不将其作为主参考坐标系。

#### C. 上层模块输出
- 当前统一视觉 affordance 表征 $z_{\text{aff}}$

也就是说，低层策略不直接消费原始视觉输入中的全部信息，而是主要消费经过上层压缩后的统一视觉表征。

#### D. 任务上下文
- `occupied`

#### E. 稳定性 proxy
相比直接输入难测的液体参数，我们输入现实中更可获得的稳定性指标，并尽量围绕 **gripper / cup** 参考系来定义：

- cup up-vector relative to gravity
- gripper linear velocity
- gripper linear acceleration
- gripper angular velocity
- gripper angular acceleration（可由差分估计）
- recent acceleration history
- jerk proxy

#### F. 可选接触反馈
如果仿真或真实系统中容易获得，还可以加入较粗粒度的接触反馈，例如：
- 是否发生非末端接触；
- 当前接触冲击大小；
- 某些 link 的接触标志摘要。

但这些信息是**可选增强项**，不是该框架成立的前提。

---

### 2.4.3 推荐的 actor 输入汇总

最终，actor 可以表示为：

```text
Actor Obs =
{
  left-arm q, dq,
  right-arm q, dq,
  tau(optional), past actions,
  active gripper pose/vel,
  cup-carrying gripper pose/vel(optional),
  z_aff,
  occupied,
  cup-tilt-to-gravity proxy,
  gripper linear acc,
  gripper angular vel,
  gripper angular acc(optional),
  recent motion history,
  optional contact summaries
}
```

这个表示的关键在于：
- 不依赖人工定义 forearm / elbow；
- 不依赖精确环境动力学参数；
- 覆盖双臂平台在当前实验中实际参与控制与建模的可观测状态；
- 使用 active gripper / cup-carrying gripper 的 end-effector frame 作为主要末端参考坐标系；
- 由上层模块先把视觉输入转换为统一的视觉 affordance 表征；
- 低层策略再围绕这一视觉表征决定具体动作。

---

## 2.5 Critic 的输入：使用 asymmetric observation

为了提高训练稳定性，我们使用 **asymmetric actor-critic**。

### 2.5.1 原因

在训练阶段，simulator 可以提供一些真实部署中难以直接获得的信息。  
因此：

- **actor**：只看现实可得观测；
- **critic**：可以额外看 privileged information。

这样既有利于 value learning，也避免 actor 依赖 oracle 信息。

### 2.5.2 Critic 可额外访问的信息

critic 可额外输入：
- 精确对象状态；
- 精确接触状态；
- 各 link 与各 affordance 区域之间的精确距离；
- 精确外力传递信息；
- 隐藏环境参数：
  - `cup_mass`
  - `cup_fill_ratio`
  - `door_mass`
  - `door_damping`
- 若使用更高保真流体代理模型，也可输入更精确的 slosh surrogate state。

但这些量只在训练中服务于 critic，不是 actor 的 deployment input。

---

## 2.6 训练中的 latent parameter randomization

如果训练总是在固定门、固定按钮、固定把手、固定杯体参数下进行，那么策略很容易只学会某个特定实例，而不是一种可推广的交互范式。

因此，下列参数应在训练中随机化：
- 杯体质量；
- 液体填充率；
- 物体局部阻力；
- 门的阻尼 / 惯性；
- 按钮刚度；
- 把手触发阈值；
- 观测噪声；
- 接触摩擦；
- 机器人初始位姿扰动。

这些变量的作用不是让 actor 直接读取，而是让策略通过历史反馈隐式适应。

---

## 2.7 动作空间与控制方式

在本文中，我们选择采用 **关节力矩（joint torque）控制** 的形式。

## 2.8 奖励函数设计：奖励“在约束下正确交互”

奖励函数的目标不是奖励“用某个特定身体部位”，而是奖励：

1. 利用与当前任务相关的有效 affordance；
2. 推动当前任务进展；
3. 在持杯时维持稳定；
4. 减少无意义碰撞和高冲击动作。

---

### 2.8.1 主任务奖励

主任务奖励依赖当前任务目标和任务进展。

例如：

#### push-affordance
当目标是推动某个门板区域时，可奖励：
- 有效接触后门状态进展；
- 门角度 / 位移朝目标方向变化。

#### press-affordance
当目标是按压按钮时，可奖励：
- 接触点进入按钮区域；
- 按压深度达到阈值；
- 触发状态成功改变。

#### handle-affordance
当目标是作用于把手时，可奖励：
- 接触位置接近把手 affordance 区域；
- 施力方向与所需交互方向一致；
- 把手状态成功变化，或后续门状态开始变化。

因此，主任务奖励不再绑定“推门”这一种操作，而是绑定到：

> “当前目标对应的关键任务进展是否被满足”。


### 2.8.2 持杯稳定奖励（Object-carrying stability term）

这一部分参考 **SoFTA / Hold My Beer** 中针对 **end-effector stabilization** 的 reward 设计思路进行改写。该工作在稳定性项中主要使用了三类量：

1. 惩罚高线加速度与高角加速度；
2. 用指数型奖励鼓励“接近零线加速度 / 零角加速度”；
3. 惩罚重力在 end-effector 局部坐标系横向平面上的投影，也就是 end-effector frame 下的 tilt。

在本文中，我们沿用这一思路，但将参考坐标系明确设为 **gripper / end-effector frame**，因为当前任务关心的是持杯末端本身的稳定性，而不是 wrist 作为上游刚体的近似状态。

需要强调的是，这里的稳定性项不是始终激活的全局约束，而是一个 **object-carrying stability term**。  
也就是说：

> **只有当 `occupied = 1` 时，该稳定性项才参与奖励计算；当 `occupied = 0` 时，该项关闭。**

这样设计的原因是：
- 当末端没有持有杯子时，我们并不需要策略额外满足“末端必须平稳”的专门约束；
- 如果在空手场景中仍然强行施加这一类稳定性项，策略会倾向于形成过于保守的行为，从而削弱空手情况下本应具有的直接、高效交互能力；
- 将其写成条件激活项，能够使同一个策略在 **occupied = 0** 和 **occupied = 1** 两种上下文下自然呈现出不同表现。

因此，当 `occupied = 1` 时，可以加入如下稳定性项：

#### （1）末端线加速度惩罚
$$
r_{\text{acc}} = -\alpha_1 \|\ddot p_{\text{EE}}\|^2
$$

#### （2）末端角加速度惩罚
$$
r_{\text{ang-acc}} = -\alpha_2 \|\dot \omega_{\text{EE}}\|^2
$$

#### （3）零线加速度奖励
$$
r_{\text{zero-acc}} = \alpha_3 \exp\left(-\lambda_{acc}\|\ddot p_{\text{EE}}\|^2\right)
$$

#### （4）零角加速度奖励
$$
r_{\text{zero-ang-acc}} = \alpha_4 \exp\left(-\lambda_{ang}\|\dot \omega_{\text{EE}}\|^2\right)
$$

#### （5）末端坐标系下的重力倾斜惩罚
参考 SoFTA 的 gravity tilt 项，可写为：
$$
r_{\text{grav-xy}} = -\alpha_5 \left\|P_{xy}(R_{\text{EE}}^T g)\right\|^2
$$

其中：
- $R_{\text{EE}}$ 表示 gripper / end-effector 的旋转矩阵；
- $g$ 为重力方向；
- $P_{xy}(\cdot)$ 表示投影到 end-effector 局部坐标系的横向平面。

这个量本质上是在惩罚“末端所携带杯体偏离竖直方向”的程度，而这正是 SoFTA 中 $r_{grav-xy}$ 的核心含义。

#### （6）力矩变化平滑项
由于本文采用 joint torque control，还应额外抑制力矩输出的剧烈跳变：
$$
r_{\text{torque-smooth}} = -\alpha_6 \|\tau_t - \tau_{t-1}\|^2
$$

#### （7）力矩幅值正则项
$$
r_{\text{torque-reg}} = -\alpha_7 \|\tau_t\|^2
$$

因此，可以将持杯稳定总项写为：

$$
r_{\text{carry-stability}} =
r_{\text{acc}}
+ r_{\text{ang-acc}}
+ r_{\text{zero-acc}}
+ r_{\text{zero-ang-acc}}
+ r_{\text{grav-xy}}
+ r_{\text{torque-smooth}}
+ r_{\text{torque-reg}}
$$

并通过 occupancy mask 使其只在持杯场景中激活：

$$
m_{\text{occ}} =
\begin{cases}
1, & \text{if occupied}=1 \\
0, & \text{if occupied}=0
\end{cases}
$$

于是，奖励中的这一部分实际写为：

$$
m_{\text{occ}} \cdot r_{\text{carry-stability}}
$$

这样，空手时策略不会受到专门的末端平稳约束；而持杯时，这一项会显式约束策略去学习低加速度、低倾斜、低冲击的交互方式。

这样的写法有三个好处：

1. **更贴近 SoFTA 的 reward 结构**  
   不是只惩罚大扰动，而是同时显式鼓励“接近零扰动”的状态。

2. **与当前 torque control 设定更一致**  
   除了末端稳定项之外，还额外约束了力矩输出的平滑性与幅值，避免策略通过剧烈瞬时输出换取短期任务进展。

3. **能够自然地区分空手与持杯两种行为模式**  
   同一个策略在 `occupied = 0` 时可以更自由地完成交互，而在 `occupied = 1` 时则会因为该项被激活而更倾向于稳态、低冲击的动作风格。


### 2.8.3 接触与安全项

我们不写“必须用某个 anchor 接触”，但可以写：

- 对错误 affordance 区域的无效碰撞进行惩罚；
- 对高冲击末端碰撞进行惩罚；
- 对能带来任务进展的有效接触给予小奖励；
- 对 self-collision、关节限位、过大速度或过大力矩给予惩罚。

特别是在 `occupied = 1` 时，可额外惩罚：
- gripper 高冲击接触；
- 导致杯体明显晃动的接触方式。

---

## 2.9 训练算法

建议采用：

- **PPO**
- recurrent actor（GRU / LSTM）
- asymmetric critic
- parallel simulation rollout

原因：
- PPO 在连续控制任务中成熟稳定；
- recurrent policy 有助于隐式辨识隐藏环境参数；
- asymmetric critic 能提升训练效率。

---

## 2.10 Curriculum 设计

为了避免一开始就学习过于复杂的 door interaction，可采用逐步 curriculum：

### Stage 1：单一 affordance 任务
- 只包含一种简单交互，例如 push patch
- 不持杯
- 学会最基础的 affordance-conditioned control

### Stage 2：持杯稳定
- 不要求复杂门交互
- 学会维持杯体稳定

### Stage 3：两类 affordance
- 例如 push patch 与 press patch 共存
- 任务随机选择
- 学会区分不同交互类型

### Stage 4：加入复杂 door-related objects
- 包括 button + door
- handle-like region + door
- sequence-style interaction

### Stage 5：统一混合训练
- occupied / unoccupied 混合
- 多种 affordance 类型混合
- 多种 task_goal 混合
- 多种稳定性要求混合

通过这种方式，最终得到的不是“会推一种门的策略”，而是“会在多种门交互方式中，根据 affordance 和约束决定如何交互的策略”。

---

