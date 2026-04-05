# Affordance-Guided Interaction 训练循环流程

> 本文档聚焦于训练循环的完整执行流程，说明每一步做了什么、数据如何流转。各模块的内部实现细节请参阅对应源码。

---

## 目录

1. [启动与初始化](#1-启动与初始化)
2. [训练主循环总览](#2-训练主循环总览)
3. [Step 1: 轨迹采集](#3-step-1-轨迹采集)
4. [Step 2: GAE 优势估计](#4-step-2-gae-优势估计)
5. [Step 3: PPO 参数更新](#5-step-3-ppo-参数更新)
6. [Step 4–5: 计数与指标更新](#6-step-45-计数与指标更新)
7. [Step 6: 课程阶段跃迁判定](#7-step-6-课程阶段跃迁判定)
8. [Step 7–8: 日志输出](#8-step-78-日志输出)
9. [Step 9: Checkpoint 保存](#9-step-9-checkpoint-保存)
10. [Step 10: Buffer 清空](#10-step-10-buffer-清空)
11. [终止与收尾](#11-终止与收尾)

---

## 1. 启动与初始化

```bash
# 典型启动命令
python scripts/train.py --config configs/training/default.yaml --num-envs 2
```

`train.py` 的 `main()` 在进入训练循环前，按以下顺序完成初始化：

```
load_config(config_path)
  → 合并 6 个 YAML (training / env / policy / task / curriculum / reward)
  │
  ├→ build_env_config(cfg)       → EnvConfig
  ├→ VecDoorEnv(n_envs, cfg)     → N 个并行环境
  ├→ build_models(cfg, device)   → Actor, Critic
  ├→ build_ppo_trainer(...)      → PPOTrainer (含两个 Adam optimizer)
  ├→ build_collector(...)        → RolloutCollector + RolloutBuffer
  ├→ CurriculumManager(...)      → 课程阶段管理
  ├→ DomainRandomizer(seed)      → 域随机化采样器
  ├→ PerceptionRuntime(...)      → 视觉 embedding 缓存
  └→ SummaryWriter(log_dir)      → TensorBoard (可选)
```

若指定 `--resume`，则从 checkpoint 恢复所有状态（模型权重、优化器动量、课程阶段、全局步数）。

---

## 2. 训练主循环总览

```
for iteration in range(start_iter, max_iterations):

    Step 1:  collector.collect(envs, n_steps, ...)     # 轨迹采集
    Step 2:  buffer.compute_gae(γ, λ, last_values)     # GAE 优势估计
    Step 3:  ppo_trainer.update(buffer)                 # PPO 参数更新
    Step 4:  global_steps += n_envs × n_steps           # 更新全局步数
    Step 5:  metrics.update_ppo(losses...)              # 指标更新
    Step 6:  curriculum.report_epoch(success_rate)       # 课程跃迁判定
    Step 7:  print(...)                                 # 控制台日志
    Step 8:  writer.add_scalar(...)                     # TensorBoard
    Step 9:  save_checkpoint(...)                       # 定期保存 (每50轮)
    Step 10: buffer.clear()                             # 清空 Buffer

    if global_steps >= total_steps: break
```

**终止条件**: `global_steps >= total_steps (10M)` 或 `Ctrl+C`。无论如何终止，都会保存 `ckpt_final.pt`。

---

## 3. Step 1: 轨迹采集

`RolloutCollector.collect()` 在 $N$ 个并行环境中同时推演 $T$ 步，将所有 transition 写入预分配的 `RolloutBuffer`。

### 单步循环 (重复 T=128 次)

```
for step in range(T):

    1. 观测展平
       actor_obs_list → batch_flatten_actor_obs() → actor_branches (dict of Tensors)
       critic_obs_list → batch_flatten_priv() → privileged_flat (Tensor)

    2. 缓存当前 RNN 隐状态
       cached_hidden = hidden.detach().clone()

    3. Actor 前向推理
       action, log_prob, entropy, hidden_new = actor.forward(actor_branches, hidden)

    4. Critic 前向推理
       value = critic.forward(actor_branches, privileged_flat)

    5. 环境 step
       actor_obs', critic_obs', reward, done, info = envs.step(action)

    6. 写入 Buffer
       buffer.add(step, obs, action, log_prob, value, reward, done, cached_hidden)

    7. 处理 episode 结束
       对 done=True 的环境: 重置 RNN 隐状态为零

    8. 视觉缓存更新
       PerceptionRuntime.prepare_batch(obs', force_refresh=done)
       → done 的环境强制刷新 embedding，其余按 refresh_interval=4 周期缓存
```

### 自动重置

当某个环境 `done=True` 时:
- `VecDoorEnv` 内部自动 `reset()`，返回新 episode 的初始观测
- `CurriculumManager` 提供新 episode 的上下文分布（决定是否持杯、哪只手持杯）
- `DomainRandomizer` 采样新的物理参数（杯质量、门阻尼等）
- 该环境的 RNN 隐状态清零
- 该 step 的 transition 正常写入 Buffer（不丢弃）

### 采集结束后

额外做一次 Critic 前向，获取最后一步的 `last_values`（用于下一步的 bootstrap）。

---

## 4. Step 2: GAE 优势估计

在整段 rollout 数据上，从后向前一次性计算 GAE（不在采集时逐步递增）：

```
buffer.compute_gae(gamma=0.99, lam=0.95, last_values, last_dones)
```

计算过程 (从 t=T-1 反向到 t=0):

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - d_{t+1}) - V(s_t)
Â_t = δ_t + γλ · (1 - d_{t+1}) · Â_{t+1}
R_t = Â_t + V(s_t)     # 目标 return
```

- `last_values` 提供 $V(s_T)$ 用于 bootstrap
- `last_dones` 屏蔽已终止 episode 的 bootstrap（避免跨 episode 传递价值）
- 计算完成后 Buffer 中包含: 观测、动作、log_prob、value、**advantage**、**return_target**

---

## 5. Step 3: PPO 参数更新

```
update_info = ppo_trainer.update(buffer)
```

### 数据切分

Buffer 中 `(T=128, N=64)` 的轨迹按 TBPTT 序列长度 $L=16$ 切分:

```
每个环境: 128 步 → 8 个长度为 16 的片段
总计: 8 × 64 = 512 个片段
随机打乱后分为 N_mb=4 个 mini-batch，每个含 128 个片段
```

每个片段附带从 Buffer 恢复的初始 RNN 隐状态 $h_{t_0}$。

### 优化循环

```
for epoch in range(K=5):                    # K 轮遍历全部数据
    for batch in mini_batches(N_mb=4):      # 每轮 4 个 mini-batch

        # Actor: 在片段内逐步展开 RNN (TBPTT)
        hidden = h_init.detach()
        for t in range(L=16):
            log_prob_t, entropy_t, hidden = actor.evaluate_actions(obs_t, act_t, hidden)

        # Critic: 无 RNN，直接展平为 (B×L, dim) 批量前向
        values = critic(obs_flat, priv_flat)

        # 计算三项损失
        actor_loss  = PPO-Clip surrogate loss
        critic_loss = clipped value loss
        entropy     = 策略熵 (正则化项)
        total_loss  = actor_loss + 0.5 × critic_loss - 0.01 × entropy

        # 反向传播 + 梯度裁剪 + 参数更新
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(actor.parameters(), 1.0)
        clip_grad_norm_(critic.parameters(), 1.0)
        actor_optimizer.step()
        critic_optimizer.step()
```

每轮迭代共执行 $K \times N_{mb} = 5 \times 4 = 20$ 次梯度更新。

---

## 6. Step 4–5: 计数与指标更新

```
# Step 4: 累计全局步数
global_steps += n_envs × n_steps    # 64 × 128 = 8,192 steps/iter

# Step 5: 更新滑动指标
metrics.update_ppo(
    actor_loss, critic_loss, entropy, clip_fraction, approx_kl, explained_variance
)
metrics.update_rollout(
    mean_reward, completed_episodes, successful_episodes, per_context_success
)
```

---

## 7. Step 6: 课程阶段跃迁判定

```
stage_changed = curriculum.report_epoch(success_rate)
```

课程管理器维护一个长度 $M=50$ 的滑动窗口，记录每轮迭代的成功率。当窗口均值 $\geq 0.8$ 时触发阶段跃迁:

```
Stage 1 (none: 1.0)
    ↓ 窗口均值 ≥ 0.8
Stage 2 (left_only: 0.5, right_only: 0.5)
    ↓ 窗口均值 ≥ 0.8
Stage 3 (none/left/right/both: 各 0.25)
```

**跃迁时的操作**:
- 更新环境的上下文采样分布（决定新 episode 的持杯配置）
- 清空滑动窗口，从零积累新阶段成功率
- **不重置**模型权重或优化器状态

---

## 8. Step 7–8: 日志输出

```
# Step 7: 控制台 (每 log_interval=1 轮)
[Iter   42] steps=  10,240 | fps=3200 | a_loss=0.0234 | c_loss=0.5123 |
            ent=1.23 | clip=0.082 | r̄=0.1234 | succ=0.650 | stage=1 | ETA=4.2h

# Step 8: TensorBoard (每轮)
writer.add_scalar("train/actor_loss", ...)
writer.add_scalar("collect/episode_success_rate", ...)
writer.add_scalar("curriculum/stage", ...)
...
```

---

## 9. Step 9: Checkpoint 保存

每 `checkpoint_interval=50` 轮保存一次:

```
save_checkpoint(
    path = ckpt_dir / f"ckpt_iter_{iteration}.pt",
    内容 = {
        iteration, global_steps,
        actor_state_dict, critic_state_dict,
        trainer_state_dict (两个 optimizer),
        curriculum_state_dict (阶段 + 滑动窗口),
        best_success_rate,
    }
)
```

恢复训练时通过 `load_checkpoint()` 还原所有状态，从中断处无缝继续。

---

## 10. Step 10: Buffer 清空

```
buffer.clear()
```

释放本轮的 transition 数据，为下一轮采集腾出空间。Buffer 结构保留（预分配的张量不重新分配）。

---

## 11. 终止与收尾

```
训练循环结束 (global_steps ≥ 10M 或 Ctrl+C)
    │
    ├→ save_checkpoint(ckpt_dir / "ckpt_final.pt", ...)
    ├→ writer.close()     # 关闭 TensorBoard
    └→ envs.close()       # 释放仿真资源
```

训练完成后可用 `evaluate.py` 评估、`export_policy.py` 导出、`rollout_demo.py` 可视化。
