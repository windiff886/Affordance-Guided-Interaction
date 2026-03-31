# 基于 Affordance 引导的单臂持杯门交互：问题定义与技术方案

## 1. 我们要解决什么问题

### 1.1 问题背景

现有很多 manipulation 工作默认：机器人主要通过 **end-effector (EE)**，也就是 gripper / hand，直接与环境交互完成任务。这种设定对于抓取、放置、按按钮等任务是自然的，但在更真实的场景中，这种设定存在一个明显限制：

> 当末端已经被占用时，机器人是否仍然能够利用身体其他区域完成环境交互，并同时保持已持物体的稳定？

这正是本文希望研究的问题。

我们关注的场景是：在 **Isaac Sim** 中使用 **Franka 单臂**，研究机器人在**持杯**与**非持杯**两种上下文下，如何完成与“门”有关的交互任务。但这里的“开门”不应被过早简化为“固定的 push door”。因为在真实环境中，开门方式本身是多样的，例如：

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

> 在 Isaac Sim 中，针对 Franka 单臂和一组具有不同交互方式的 door-related objects，构建一个 affordance-guided interaction framework。系统首先表示或预测环境中的候选交互 affordance（如 push region、press region、handle region 等），再结合机器人当前的末端占用状态与持杯稳定性约束，学习一个统一的策略去执行这些交互；当末端空闲时，系统可以自然使用 gripper；当末端持杯时，系统应尽量抑制末端剧烈扰动，并根据身体几何与接触可达性，选择更合适的身体局部区域完成交互。

这里强调一点：本文并不要求预先显式规定“持杯时必须用肘部”或“必须用前臂”。相反，我们希望系统在 affordance 与约束共同作用下，自然学出合理的身体接触分配。

---

## 2. 我们要如何解决这个问题

## 2.1 总体思路

我们的总体方法是：

> 构建一个 **affordance-guided interaction architecture**。  
> 上层接收原始视觉输入以及当前任务目标，并将其转换为两类中间表示：一类用于描述当前任务的完成程度或交互进展，另一类用于描述环境中的对象 affordance；下层再结合机器人当前状态、末端占用情况和持杯稳定性信息，输出具体动作。

这样做的目的，是避免把问题过早收缩成“为某一种门显式写规则的控制器设计”，而是建立一套能够覆盖多种 door-related interaction 的统一表示与控制框架。

为了让结构更清晰，我们将整体方案分成两个层面：

### 第一层：Affordance / Progress 表示层
这一层以原始视觉输入和任务目标作为输入，输出两类量：

1. **任务进展表示（task-progress representation）**  
   用于描述当前任务完成到了什么程度，或者当前交互是否已经触发了关键状态变化。  
   例如：
   - 门是否已经被部分打开；
   - 按钮是否已经被按下；
   - 把手是否已经被触发；
   - 顺序交互中的前置步骤是否已经完成。

2. **对象 affordance 表示（object-affordance representation）**  
   用于描述环境中“哪里可以做什么”，例如：
   - pushable patch
   - pressable patch
   - handle-like patch
   - ordered interaction candidate

这一层的功能不是直接输出动作，而是把原始视觉输入转化为更适合决策和控制使用的结构化表征。

### 第二层：Constraint-aware 执行层
这一层接收：
- 机器人自身状态；
- 末端是否被占用；
- 当前稳定性要求；
- 来自上层的任务进展表示；
- 来自上层的对象 affordance 表示。

然后输出真正的控制动作。

整个系统的关键是：

> 上层负责把“看到的场景”和“当前要完成的目标”转换成结构化交互表示，  
> 下层负责在身体约束存在时，把这些表示变成可执行动作。


## 2.2 问题建模：Affordance-Conditioned Interaction MDP

我们将任务建模为一个包含上下文和 affordance 条件的决策问题：

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
- `stability_level`：当前任务对末端稳定性的要求强度。

与前一版不同，这里不再将 `task_goal` 直接作为执行策略的 context 输入。  
相反，**task goal 被送入上层 affordance 模块**，与原始视觉输入一起生成两类表示：

- $z_{\text{prog},t}$：当前任务进展表示（task-progress representation）
- $z_{\text{aff},t}$：当前对象 affordance 表示（object-affordance representation）

于是策略可以写成：

$$
a_t = \pi_\theta(o_t, h_t, c_t, z_{\text{prog},t}, z_{\text{aff},t})
$$

其中：
- $o_t$：当前机器人观测；
- $h_t$：历史隐状态；
- $c_t$：任务上下文；
- $z_{\text{prog},t}$：任务进展表示；
- $z_{\text{aff},t}$：对象 affordance 表示；
- $a_t$：动作。

这种写法的意义是：

> 策略不需要直接从原始视觉和任务目标中自己分离“当前任务做到哪一步”和“环境哪里可以操作”，  
> 而是基于上层提供的结构化表示，在身体约束下学习如何执行交互。


## 2.3 Affordance 模块如何实现

在当前版本中，我们优先选择一条**尽量不额外训练、可直接复用现成预训练模型**的实现路线。其核心思想不是直接将整幅原生点云输入低层策略，而是先利用现成的开集视觉分割模型从 RGB-D 观测中提取与 door-related interaction 相关的对象区域，再结合深度反投影得到对应的局部点云，最后将这些几何信息压缩为结构化的 affordance 表示 $z_{\text{aff}}$，并与任务进展表示 $z_{\text{prog}}$ 一起提供给执行策略。

这样做有三个直接好处：

1. **不需要自行训练门分割网络或 3D affordance 网络**，能够尽快完成系统原型搭建；
2. **避免低层 actor 直接处理高维噪声点云**，更符合本文“上层表示、下层执行”的架构设计；
3. **便于后续逐步升级**：当系统已经能够工作后，再考虑从 object-level 提取扩展到 point-level affordance 预测。

因此，本节将当前版本的 Affordance 模块具体化为一条“现成模型 + 几何处理 + 结构化编码”的实现路径。

### 2.3.1 当前版本的设计原则

当前版本的 Affordance 模块应满足以下原则：

1. **尽量使用现成模型与现成权重**，避免额外的大规模监督训练；
2. **优先从 RGB 上做开集检测 / 分割，再借助 depth 得到 3D 点云**，而不是直接从原始点云中端到端学习门分割；
3. **上层输出应为任务相关的低维结构化表示**，而不是未经压缩的整场景点云；
4. **优先关注可交互局部区域，而非门对象的完整重建**，因为策略真正需要的是“哪里可以做什么”，而不仅仅是“门在哪里”。

基于这些原则，本文当前版本不采用“从场景原始点云端到端训练 3D 实例分割网络”的路线，而采用更现成、工程代价更低的 RGB-D 分割与投影方案。

### 2.3.2 最现成的实现路线：开集分割 + 深度反投影

在当前版本中，我们推荐使用如下上层处理管线：

```text
RGB-D observation + task_goal
    -> open-vocabulary detection / segmentation
    -> door / handle / button mask
    -> depth back-projection
    -> local point clouds
    -> geometric summarization / optional frozen encoder
    -> z_aff
```

其中，最现成的候选模型包括：

- **LangSAM**
- **Grounded-SAM 2**
- **Grounding DINO + SAM / SAM 2**

这些模型都可以直接下载现成权重，并通过文本提示词对图像中的对象进行开集检测或分割。因此，在不自行训练 perception 模块的前提下，我们可以使用如下文本提示：

- `"door"`
- `"door handle"`
- `"button"`
- `"push bar"`

从当前 RGB 图像中直接获得对应的 mask 或 bounding box。然后结合深度图，将 mask 区域内的像素反投影到三维空间，得到：

- 门板点云 $P_{\text{door}}$
- 把手点云 $P_{\text{handle}}$
- 按钮点云 $P_{\text{button}}$

这种实现方式的关键优点在于：

- 使用的是**现成视觉模型**而非自训网络；
- 开集文本提示与 door-related interaction 的多样性天然匹配；
- 容易扩展到更多对象部件与交互区域；
- 在 Isaac Sim 中也便于和深度图、实例信息做对照验证。

### 2.3.3 从 mask 到门相关点云

在从现成分割模型获得 mask 之后，使用相机内参将深度图反投影为点云。设像素坐标为 $(u,v)$，深度为 $d$，相机内参为 $(f_x, f_y, c_x, c_y)$，则反投影公式为：

$$
x = (u-c_x)d/f_x, \qquad
y = (v-c_y)d/f_y, \qquad
z = d
$$

在此基础上，当前版本的门相关点云提取流程如下：

1. 根据 `door` 的 mask 保留门板区域像素，得到 $P_{\text{door}}$；
2. 根据 `door handle` 的 mask 保留把手区域像素，得到 $P_{\text{handle}}$；
3. 根据 `button` 的 mask 保留按钮区域像素，得到 $P_{\text{button}}$；
4. 对上述点云做简单的几何清理，例如：
   - voxel downsampling
   - statistical outlier removal
   - 半径滤波或邻域滤波
5. 如有必要，再对门板点云做平面拟合，估计门平面法向与局部参考坐标系。

需要强调的是，当前版本不要求将整扇门的所有点都输入后续网络。与其保留完整门对象的高维点集，更推荐只保留：

- 与当前相机视角中可见的门局部；
- 与当前任务目标相关的候选交互区域；
- 与机器人当前末端或前臂距离较近的局部区域。

也就是说，在当前框架中，点云提取的重点不是“完整重建门”，而是“保留对交互决策最有用的几何信息”。

### 2.3.4 当前版本中 $z_{\text{aff}}$ 的推荐构造方式

在当前版本中，我们不建议将原始点云直接拼接到 actor observation 中。更合适的做法是：先将门相关点云转为**低维几何摘要（geometric summary）**，再将其作为对象 affordance 表示的一部分输入策略。

推荐的几何摘要包括：

- 门板中心 $c_{\text{door}}$
- 门平面法向 $n_{\text{door}}$
- 门板包围盒尺寸 $b_{\text{door}}$
- 把手中心 $c_{\text{handle}}$
- 按钮中心 $c_{\text{button}}$
- gripper 到门平面的距离
- gripper 到把手的距离
- gripper 到按钮的距离
- 当前检测 / 分割置信度
- 当前候选交互类型 one-hot

因此，当前版本推荐将对象 affordance 表示写为：

$$
z_{\text{aff}} =
[c_{\text{door}}, n_{\text{door}}, b_{\text{door}}, c_{\text{handle}}, c_{\text{button}},
 d_{g,door}, d_{g,handle}, d_{g,button},
 \text{affordance\_type}, \text{confidence}]
$$

其中：

- $d_{g,door}$ 表示 gripper 到门平面的距离；
- $d_{g,handle}$ 表示 gripper 到把手中心的距离；
- $d_{g,button}$ 表示 gripper 到按钮中心的距离；
- `affordance_type` 用于表示当前关注的是 push / press / handle 等哪一类交互；
- `confidence` 表示来自上层分割模型的置信度或可见性质量估计。

这种表示的优点是：

1. **不需要训练额外的 3D 编码器**；
2. **对 RL 更稳定**，因为输入维度更低、语义更明确；
3. **与当前执行策略的 observation 结构更一致**；
4. **便于分析策略到底在利用哪些 affordance 线索**。

### 2.3.5 可选增强：冻结的现成点云编码器

如果后续希望在不自行训练 3D perception 的前提下，进一步引入更丰富的点云表征，则可以在上述几何摘要之外，再加入一个**冻结的现成点云编码器**作为辅助分支。例如：

- 使用现成预训练的 **Point-MAE** 编码门局部点云；
- 使用现成预训练的 **ULIP / ULIP-2** 编码门局部点云；
- 将其输出的 embedding 与前述几何摘要拼接，作为扩展后的 $z_{\text{aff}}$。

此时可写为：

$$
z_{\text{aff}} = [z_{\text{geom}}, z_{\text{pc-embed}}]
$$

其中：

- $z_{\text{geom}}$ 表示低维几何摘要；
- $z_{\text{pc-embed}}$ 表示冻结点云基础模型输出的 embedding。

但在当前版本中，这一分支应视为**可选增强项**，而非系统成立的必要条件。原因是：

- 通用点云基础模型未必针对 door-related interaction 的局部几何进行优化；
- 如果一开始同时引入大模型 embedding 和 RL，系统调试复杂度会明显上升；
- 从工程可控性出发，先使用几何摘要往往更稳妥。

因此，当前版本推荐的优先级是：

> **几何摘要优先，冻结点云编码器可选。**

### 2.3.6 当前版本中 $z_{\text{prog}}$ 的实现建议

由于本文的上层模块不仅要输出对象 affordance 表示，还要输出任务进展表示，因此当前版本也需要给出一个尽量现成的 $z_{\text{prog}}$ 实现方式。

为了避免额外训练进展识别网络，当前版本推荐采用两类较低成本方式：

#### （1）仿真状态驱动的 progress 表示
在 Isaac Sim 中，可以直接利用环境状态构造任务进展特征，例如：

- 门的开合角度或位移；
- 按钮是否被按下；
- 把手是否被触发；
- 顺序任务中的前置步骤是否已完成。

这种方案虽然带有 simulator 提供的便利，但非常适合作为当前版本的 progress 基线。

#### （2）视觉 / 几何阈值驱动的 progress 表示
如果希望减少对 simulator oracle 的依赖，也可以用较简单的阈值式视觉逻辑近似构造，例如：

- 门平面相对初始位姿是否已发生明显变化；
- 按钮区域是否出现持续接触并产生位移；
- 把手局部是否发生触发状态改变。

在当前版本中，$z_{\text{prog}}$ 可以是一个低维向量，用于表示当前关键状态是否被触发，以及当前任务完成进度是否推进。

### 2.3.7 当前版本的整体接口形式

基于以上设计，当前版本的 Affordance 模块可写为：

```text
Input:
  RGB_t, Depth_t, task_goal_t

Affordance Module:
  1. Open-vocabulary segmentation
  2. Door / handle / button mask extraction
  3. Depth back-projection
  4. Door-related local point cloud construction
  5. Geometric summarization
  6. Optional frozen point-cloud encoder
  7. Progress feature construction

Output:
  z_prog_t, z_aff_t
```

于是当前版本的设计重点就变为：

- 用现成模型解决“门及其关键局部在哪里”；
- 用几何处理解决“这些局部在 3D 中如何表示”；
- 用结构化编码解决“哪些信息应该真正进入策略 observation”。

这使得本文当前版本的 Affordance 层不再是一个待定模块，而是一个**可直接落地、且尽量不依赖额外训练的上层感知与表示组件**。


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
- joint positions $q_t$
- joint velocities $\dot q_t$
- joint torques / motor currents（如果可用）
- previous actions $a_{t-k:t-1}$

#### B. 明确可定义的末端状态
在当前版本中，**优先使用 gripper / end-effector frame 作为主要参考坐标系**。

这样做的原因是：
- 本文关心的稳定性本质上是“杯体是否稳定”，而杯体与 gripper 的关系比与 wrist 更直接；
- gripper frame 更适合表达末端相对重力方向的倾斜；
- 如果同时保留 gripper 和 wrist 两套末端位姿与速度表示，会带来较强冗余；
- 对 SoFTA 风格的稳定性奖励而言，参考坐标系本身也是定义的一部分，因此不宜随意将 end-effector frame 替换为 wrist frame。

因此，在当前设定下，低层策略优先接收：
- gripper pose / velocity

如果后续实验发现某些动态 proxy 用 wrist rigid body 更稳定，wrist 信息可以作为可选辅助量加入，但当前版本不将其作为主参考坐标系。

#### C. 上层模块输出
- 当前任务进展表示 $z_{\text{prog}}$
- 当前对象 affordance 表示 $z_{\text{aff}}$

也就是说，低层策略不直接消费原始视觉输入中的全部信息，而是主要消费经过上层压缩后的结构化表示。

#### D. 任务上下文
- `occupied`
- `stability_level`

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
  q, dq, tau(optional), past actions,
  gripper pose/vel,
  z_prog,
  z_aff,
  occupied,
  stability_level,
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
- 使用 gripper / end-effector frame 作为主要末端参考坐标系；
- 由上层模块先把视觉输入转换为任务进展表示和对象 affordance 表示；
- 低层策略再围绕这些结构化表示决定具体动作。

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

如果进一步考虑不同程度的稳定性要求，还可以通过 `stability_level` 去调节该项的整体权重，即：

$$
\lambda_{\text{stab}}(\text{stability\_level}) \cdot m_{\text{occ}} \cdot r_{\text{carry-stability}}
$$

其中：
- `occupied` 决定该项是否激活；
- `stability_level` 决定激活后该项有多强。

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

