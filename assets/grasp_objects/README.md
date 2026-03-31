# Grasp Object Assets

该目录存放用于 Isaac Sim 持物任务初始化的训练资产。

当前包含：

- `cup/carry_cup.usda`: 无把杯训练资产
- `tray/carry_tray.usda`: 托盘训练资产
- `catalog.json`: 资产清单、默认变体、pickup/scene-spawn 配置和随机化范围

设计原则：

- 优先保证训练时的物理稳定性和可随机化能力
- 允许后续把公开来源网格作为视觉层替换进来
- 杯子资产根 prim 采用“杯底中心 + Z 向上”的语义，便于做竖直抓取与抬升判定

后续如果导入公开网格，建议只替换视觉层，不直接替换训练碰撞体。

## `load_scene.py` 使用方法

可以通过 `scripts/load_scene.py` 的 grasp 参数，在场景启动时加载这里的资产。

常用参数：

- `--grasp-object {cup,tray}`: 选择要加载的持物资产类型
- `--grasp-variant <name>`: 指定 catalog 里的变体名称；当前默认变体分别是 `carry_cup` 和 `carry_tray`
- `--grasp-arm {left,right}`: 指定使用哪一侧夹爪；当前 `cup` 第一版只支持 `left`
- `--gripper-closed-position <value>`: 覆盖默认夹爪闭合目标值

当前逻辑说明：

- `cup` 会执行“基座相对 scripted pickup”：先在 `base_link` 坐标系下生成支撑台和竖直杯子，再按 `pregrasp -> approach -> closure -> settle -> lift -> retreat` 顺序运行左臂抓取脚本
- `cup` 抓取成功与否会在运行时按抬升高度、倾斜角、线速度和与抓取参考系的距离做判定；失败会直接打印 payload 并退出
- `tray` 目前仍然只是在场景中生成，不执行 scripted pickup
- `--attach-grasp-object` 已废弃；为了兼容旧命令仍可保留，但对 `cup` 不会再创建 fixed joint

示例：

```bash
# 左臂执行基座相对 scripted pickup
python3 scripts/load_scene.py --grasp-object cup

# 右夹爪把托盘生成到场景中
python3 scripts/load_scene.py --grasp-object tray --grasp-arm right

# 指定变体并覆盖夹爪闭合值
python3 scripts/load_scene.py \
  --grasp-object cup \
  --grasp-variant carry_cup \
  --gripper-closed-position -0.20

# 兼容旧命令；不会再创建 fixed joint
python3 scripts/load_scene.py --grasp-object cup --attach-grasp-object
```

`cup` 当前的 pickup 目标点来自 `catalog.json` 里的 `pickup.pickup_point_xyz_m`，它是相对于 `/World/Robot/base_link` 的固定位置，不是世界坐标，也不是直接生成在夹爪内部。

如果想查看完整参数列表，可以运行：

```bash
python3 scripts/load_scene.py --help
```
