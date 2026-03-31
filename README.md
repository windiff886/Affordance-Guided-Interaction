# Affordance-Guided-Interaction

本项目用于构建一个面向 door-related interaction 的研究与代码框架，核心关注
affordance 引导下的交互决策，以及末端持物约束下的稳定执行。

当前仓库还处于框架搭建阶段，重点是先把模块边界、目录结构和后续扩展入口整理出来，
具体算法、Isaac Sim 接入和训练细节仍保留为占位实现。

## 项目结构

```text
Affordance-Guided-Interaction/
├── README.md
├── project_architecture.md
├── configs/
├── docs/
├── scripts/
├── src/
├── tests/
└── .gitignore
```

## 目录说明

- `project_architecture.md`
  面向工程落地的项目架构设计文档。
- `configs/`
  配置骨架，按环境、任务、策略、奖励、训练、课程等维度拆分。
- `docs/`
  目前主要存放实现计划文档。
- `scripts/`
  训练、评估、rollout 演示和策略导出入口脚本。
- `src/affordance_guided_interaction/`
  核心源码目录，当前已拆分为 `envs`、`perception`、`observations`、
  `policy`、`rewards`、`training`、`evaluation`、`common`、`utils`。
- `tests/`
  当前用于验证包结构、观测构建和奖励聚合等最小骨架能力。

## 当前状态

目前已经完成的是：

- 基础目录和模块分层
- 观测构建与奖励聚合的最小实现
- 可训练的 `assets/grasp_objects/` 持物资产骨架（无把杯 + 托盘）
- 训练/评估脚本入口占位
- 配置与测试骨架

目前仍然留白的是：

- Isaac Sim 场景与环境接入
- affordance/progress 模块的真实实现
- actor/critic 网络与 PPO 训练逻辑
- 真实任务奖励和评估指标

如果后续继续开发，建议优先补 `envs + observations + rewards + training`
之间的最小闭环，再逐步替换上层表示模块。

## Isaac Sim 持物资产

新增的 `assets/grasp_objects/` 目录用于放置训练期持物相关资产定义：

- `cup/carry_cup.usda`
- `tray/carry_tray.usda`
- `catalog.json`

在 Isaac Sim 里当前有两种启动语义：

```bash
python3 scripts/load_scene.py --grasp-object cup
python3 scripts/load_scene.py --grasp-object tray --grasp-arm right
```

- `cup`: 走基座相对 scripted pickup 流程，会在 `base_link` 相对坐标下生成支撑台和竖直杯子，然后驱动左臂执行 `pregrasp -> approach -> closure -> settle -> lift -> retreat`
- `tray`: 仍然只是把托盘生成到场景里，不执行抓取序列
- `--attach-grasp-object`: 已废弃；为了兼容旧命令保留，但不会再创建 fixed joint
