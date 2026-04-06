# policy — 推门策略网络（目标设计说明）

> 文档状态：目标设计导向
>
> 本文档描述的是 `policy/` 层的目标形态，也就是我们希望最终训练系统中的策略模块如何工作。
> 它以 `docs/training_pipeline_detailed.md` 中已经确认的策略规格为准，
> 并以 `observations/README.md` 作为上游观测接口的参考。
>
> 如果当前代码实现与本文档不一致，应以本文档作为后续重构方向。

---

## 1. 本层做什么

`policy/` 是训练系统中的决策核心。

它只负责一件事：

> 根据当前观测，输出双臂 12 维关节力矩策略。

在系统中的位置如下：

```text
在线视觉
  RGB-D -> 2D识别 -> 3D点云提取 -> Point-MAE编码 -> door_embedding
                                                     │
                                                     ▼
observations/
  原始物理状态 -> actor_obs / critic_obs
                          │
                          ▼
policy/
  actor -> 输出动作
  critic -> 输出价值
                          │
                          ▼
envs/
  执行动作，推进物理仿真
```

本层不负责：

- 不处理原始 `RGB`、深度图或原始点云
- 不计算 reward
- 不决定课程阶段
- 不直接读取仿真内部 oracle 给 actor
- 不规定“哪只手必须推门”

---

## 2. Actor 输入

### 2.1 总体接口

actor 的正式输入定义为：

```text
o_actor,t = {
  proprio,
  ee,
  context,
  stability,
  visual
}
```

这五个分支构成同一套策略在四种正式场景中共享的输入接口：

- 无持杯
- 左臂持杯
- 右臂持杯
- 双臂持杯

本设计要求这套接口在所有课程阶段都保持稳定，不允许因为阶段变化而动态增删字段。

### 2.2 `proprio`

`proprio` 描述机器人自身的关节状态，至少包含：

- 双臂 12 个关节位置
- 双臂 12 个关节速度
- 双臂 12 个当前执行力矩或最近测得力矩
- 最近一步动作 `a_(t-1)`

如果后续实验表明确有收益，也允许保留一个很短的动作历史窗口，但目标设计不依赖长 history stacking，因为时序建模主要由 recurrent actor 负责。

`proprio` 回答的是：

> 机器人自己的身体现在处于什么状态，刚才输出了什么控制。

### 2.3 `ee`

`ee` 描述左右末端在 `base_link` 坐标系下的状态，至少包含：

- 左末端位置、姿态、线速度、角速度、线加速度、角加速度
- 右末端位置、姿态、线速度、角速度、线加速度、角加速度

这里统一使用 `base_link` 而不是世界坐标系，原因很直接：

- 训练中允许存在基座初始位姿扰动
- 若直接给世界坐标，策略会学到与底盘绝对摆放相关的脆弱表示
- 使用 `base_link` 后，策略看到的是与机器人自身一致的局部几何关系

`ee` 回答的是：

> 两个末端现在在哪里、以什么姿态和速度在运动。

### 2.4 `context`

`context` 只负责描述当前回合的持杯上下文：

- `left_occupied ∈ {0, 1}`
- `right_occupied ∈ {0, 1}`

这两个量在 episode 开始时确定，在回合内保持不变。

它们的作用不是直接规定动作，而是告诉策略：

> 当前哪一侧处在“需要兼顾持杯稳定性”的约束模式下。

本设计明确不把左臂/右臂重命名为“执行臂”或“持杯臂”。  
策略始终面对固定的 `left/right` 输入顺序，再通过 `context` 自己学会角色分配。

### 2.5 `stability`

`stability` 只保留最核心的持杯稳定性信息：

- 左侧倾斜程度
- 右侧倾斜程度

它的职责非常单一：

> 告诉策略当前末端姿态是否正在破坏“杯口朝上”的稳定性。

注意这里的设计边界：

- 与末端运动有关的速度、加速度信息已经放在 `ee` 中
- `stability` 不再重复承载线加速度、角加速度等高维 proxy
- 这样 actor 可以清楚地区分“运动状态”和“持杯稳定性状态”

### 2.6 `visual`

`visual` 分支只包含一个正式视觉结果：

```text
door_embedding ∈ R^768
```

它来自固定频率更新的在线视觉链路：

```text
RGB-D -> 2D视觉识别 -> 3D点云提取 -> 冻结 Point-MAE 编码 -> door_embedding
```

actor 不直接接收：

- 原始 `RGB`
- 原始深度图
- 原始点云

因此，`policy/` 看到的是已经压缩好的门相关视觉表征，而不是视觉前端本身。

### 2.7 Actor 明确不能看到的信息

actor 必须与部署可得信息保持一致，因此它不能直接接收：

- 门铰链真实角度
- 门铰链真实角速度
- 门板精确世界位姿
- 杯体精确位姿和速度
- `door_mass`、`door_damping`、`cup_mass` 等隐藏物理参数
- 任何仿真内部 oracle 标签

actor 对门动力学和环境变化的适应，必须来自：

- 在线视觉
- 本体状态
- 末端状态
- 倾斜稳定性信号
- 时序交互历史

---

## 3. Critic 输入

### 3.1 总体接口

critic 的正式输入定义为：

```text
o_critic,t = {
  actor_obs,
  privileged
}
```

其中：

- `actor_obs` 与 actor 完全一致
- `privileged` 只在训练时提供

critic 的目标不是替 actor 做决策，而是提供更准确的 value estimate。

### 3.2 `privileged`

`privileged` 至少包含：

- 门铰链真实角度与角速度
- 门板或关键门体刚体的精确位姿
- 当前回合采样到的关键域随机化参数
- `cup_dropped ∈ {0,1}` 事件标志

这里的关键域随机化参数，至少包括：

- `door_mass`
- `door_damping`
- `cup_mass`

本设计有一个明确约束：

> critic 不需要看到杯体的精确位姿、姿态和速度。

原因是这里假设杯体在未掉落前被稳定抓持，value 估计真正需要知道的是：

- 门当前推进到了哪里
- 当前回合的动力学隐藏参数是什么
- 是否已经发生掉杯这一终止事件

因此，`cup_dropped` 对 critic 是有价值的；但连续的杯体刚体状态不是本设计的必要输入。

---

## 4. Actor 模型

### 4.1 总体结构

目标 actor 采用多分支编码加循环主干的结构：

```text
proprio   -> encoder -> f_proprio
ee        -> encoder -> f_ee
context   -> direct concat
stability -> lightweight encoder -> f_stab
visual    -> encoder -> f_vis

concat([f_proprio, f_ee, context, f_stab, f_vis])
    -> recurrent backbone
    -> Gaussian action head
    -> tau ∈ R^12
```

形式化写成：

```text
f_t = concat(
  E_proprio(proprio_t),
  E_ee(ee_t),
  context_t,
  E_stab(stability_t),
  E_vis(visual_t)
)

h_t = RNN(f_t, h_(t-1))

(mu_t, sigma_t) = Head(h_t)

a_t ~ N(mu_t, diag(sigma_t^2))
```

### 4.2 分支编码器

#### `E_proprio`

`proprio` 是高密度本体状态，适合使用 2 层 MLP 进行编码。

它的职责是提取：

- 双臂关节构型
- 当前运动趋势
- 最近控制意图

#### `E_ee`

`ee` 包含左右末端的位姿和运动状态，同样适合使用 2 层 MLP 编码。

它的职责是提取：

- 末端相对基座的几何关系
- 接近门时的运动学模式
- 持杯侧与非持杯侧的动作动态

#### `context`

`context` 只有两个二值量，因此不需要复杂网络。

目标设计推荐直接拼接进融合特征，而不是单独堆叠深层 MLP。  
这样可以保持语义清晰，也避免对低维上下文做过度变换。

#### `E_stab`

`stability` 输入维度很小，只包含左右两侧的倾斜程度，因此应使用一个很轻量的 encoder。

它的职责不是学习复杂几何，而是给后续主干一个明确的稳定性通道。

#### `E_vis`

`visual` 输入是 `door_embedding ∈ R^768`，需要通过视觉投影 MLP 压缩到适合策略融合的维度。

它的职责是提取：

- 门相关空间结构
- 推门相关局部几何线索
- 与当前交互阶段相关的视觉语义

需要强调：

- `policy/` 不负责训练 Point-MAE
- 视觉前端是冻结的
- actor 只消费 `door_embedding`

### 4.3 Recurrent Backbone

多分支特征融合后，送入一个循环主干网络。

目标设计要求：

- 默认使用 **GRU**
- 可选切换为 **LSTM**
- 每个并行环境都维护独立隐状态

之所以需要 recurrent backbone，是因为本任务本质上是部分可观测交互问题：

- actor 看不到门的真实铰链状态
- actor 看不到隐藏动力学参数
- actor 需要通过连续交互的反馈去推断“门是不是更重”“当前施力是否有效”

因此，时序信息不应主要依赖长 observation 堆叠，而应由 RNN 隐状态承担。

### 4.4 Action Head

Action Head 负责把循环特征映射为连续动作分布。

目标设计采用对角高斯参数化：

- 输出动作均值 `mu_t ∈ R^12`
- 输出每一维的标准差 `sigma_t ∈ R^12`

在训练中：

```text
a_t ~ N(mu_t, diag(sigma_t^2))
```

在评估或部署推理中：

```text
a_t = mu_t
```

这让 PPO 训练阶段保留探索，而评估阶段使用确定性动作。

---

## 5. Critic 模型

### 5.1 总体结构

critic 采用非对称结构，不参与部署推理。

目标设计如下：

```text
actor-side branches
  -> actor-side encoders

privileged
  -> privileged encoder

concat(actor_features, privileged_features)
  -> critic MLP
  -> V(s)
```

形式化写成：

```text
g_t = concat(
  E_proprio^c(proprio_t),
  E_ee^c(ee_t),
  context_t,
  E_stab^c(stability_t),
  E_vis^c(visual_t),
  E_priv(privileged_t)
)

V_t = MLP(g_t)
```

### 5.2 为什么 critic 不用循环结构

目标基线中，critic 不要求使用 recurrent structure，原因是：

- critic 已经能看到比 actor 更完整的状态
- 它的主要任务是稳定地拟合 value，而不是在部署约束下隐式辨识环境
- 保持 critic 为纯 MLP，更利于训练稳定性和实现简洁性

如果未来实验充分证明 recurrent critic 有明确收益，可以再单独扩展；但这不属于当前目标设计的默认方案。

### 5.3 Actor-Critic 的职责分离

这套设计的关键不是“让 critic 更强”，而是：

- actor 在受限信息下学会决策
- critic 在训练中提供更准确的价值梯度

因此 privileged 信息只能停留在 critic 内部，不能泄漏到 actor 正式输入。

---

## 6. 输出语义

### 6.1 动作定义

策略输出始终定义为：

```text
a_t = tau_t ∈ R^12
```

也就是双臂 12 维关节力矩。

其中：

- 前 6 维对应一侧机械臂
- 后 6 维对应另一侧机械臂

### 6.2 Raw Torque 语义

actor 输出的是原始力矩意图，而不是已经过安全裁剪的执行值。

这意味着：

- `policy/` 负责产生 raw torque
- `envs/` 可在执行前根据硬件上限做 clip
- 安全惩罚中的“力矩超限”应基于 raw torque，而不是 clip 之后的结果

否则策略的超限控制意图会被环境裁剪掩盖。

### 6.3 不做动作屏蔽

本设计明确不引入基于持杯上下文的动作 mask。

也就是说：

- 即使左臂持杯，左臂仍可输出动作
- 即使双臂都持杯，双臂仍都可参与交互

系统不通过硬编码告诉策略“哪只手不能动”，而是让策略自己在 reward 和上下文约束下学会保守或主导的分工方式。

---

## 7. 时序语义与 Reset

### 7.1 隐状态

每个并行环境都维护一份 actor 循环隐状态。

在 episode 内：

- 隐状态持续传递
- 策略利用它积累交互记忆

在 episode reset 时：

- 对应环境的隐状态必须清零
- 对应环境的动作历史缓存必须同步清零

### 7.2 视觉缓存与 policy 的关系

视觉缓存本身不属于 `policy/` 的隐藏状态，但它会影响 actor 的输入。

因此在 reset 后必须保证：

- 在线视觉先完成一次 warm start
- actor 在新回合起点就拿到有效的 `door_embedding`

否则策略的第一步会以失真的视觉输入启动。

---

## 8. 本层不做的事

为了让模块边界长期清晰，`policy/` 不承担以下职责：

- 不直接处理原始相机数据
- 不在策略内部跑点云编码器
- 不生成 reward
- 不决定课程切换
- 不读取 deployment 不可得的 oracle 给 actor
- 不硬编码“左手推门 / 右手持杯”之类的规则
- 不在策略层执行力矩 clip

换句话说，`policy/` 只做：

> 接收结构化观测，输出动作分布和价值估计。

---

## 9. 本节结论

`policy/` 的目标设计可以压缩成三句话：

1. actor 接收 `proprio + ee + context + stability + visual`，输出 12 维 raw torque。
2. critic 接收 `actor_obs + privileged`，只在训练时提供 value estimate。
3. 模型结构是多分支 encoder + recurrent actor + asymmetric critic，不做动作 mask，不处理原始视觉。
