"""
Uni-Dingo 双臂机器人 Isaac Sim 导入脚本

使用方式:
  方式 A - 在 Isaac Sim 的 Script Editor 中运行（推荐）:
    1. 打开 Isaac Sim
    2. Window > Script Editor
    3. 复制本脚本内容粘贴，修改 URDF_PATH 路径后运行

  方式 B - 作为 standalone 脚本运行:
    isaacsim-python /path/to/import_to_isaacsim.py

前提条件:
  - 已经通过 generate_urdf.sh 生成了 uni_dingo_dual_arm.urdf
  - 或者手动用 xacro 展开了 URDF 文件
"""

import os
import omni.kit.commands
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools

# =========================================================
#  配置区域 - 请修改以下路径为你的实际路径
# =========================================================

# URDF 文件路径 (请替换为你的实际路径)
URDF_PATH = os.path.expanduser(
    "~/Code/Affordance-Guided-Interaction/asserts/robot/urdf/uni_dingo_dual_arm_absolute.urdf"
)

# 导入后的 USD 保存路径 (可选)
USD_OUTPUT_PATH = os.path.expanduser(
    "~/Code/Affordance-Guided-Interaction/asserts/robot/usd/uni_dingo_dual_arm.usd"
)

# =========================================================
#  导入配置
# =========================================================

def import_robot():
    """导入 Uni-Dingo 双臂机器人到 Isaac Sim"""
    
    if not os.path.exists(URDF_PATH):
        print(f"❌ URDF 文件不存在: {URDF_PATH}")
        print("   请先运行 generate_urdf.sh 生成 URDF 文件")
        return None
    
    print(f"📂 正在导入 URDF: {URDF_PATH}")
    
    # 创建导入配置
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    
    # === 关键参数说明 ===
    
    # 是否合并固定关节 (False = 保留所有 link 用于仿真精确性)
    import_config.merge_fixed_joints = False
    
    # 是否使用凸分解生成碰撞体 (True = 更精确但更慢)
    import_config.convex_decomp = False
    
    # 是否导入 URDF 中定义的惯性张量 (True = 使用 URDF 中精确的惯性参数)
    import_config.import_inertia_tensor = True
    
    # 是否固定底座 (False = 底盘可自由移动; True = 底盘固定在世界坐标)
    # 对于移动机器人，通常设为 False
    import_config.fix_base = False
    
    # 从视觉模型生成碰撞体 (如果 URDF 已有碰撞定义则设为 False)
    import_config.collision_from_visuals = False
    
    # 距离缩放因子 (URDF 通常使用米为单位，Isaac Sim 也使用米)
    import_config.distance_scale = 1.0
    
    # 是否自动创建物理场景
    import_config.create_physics_scene = True
    
    # 执行导入
    status, stage_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=import_config,
    )
    
    if not status:
        print("❌ URDF 导入失败!")
        return None
    
    print(f"✅ 机器人已导入到: {stage_path}")
    
    # 获取 stage
    stage = omni.usd.get_context().get_stage()
    
    # 设置物理场景重力
    scene_path = "/physicsScene"
    if not stage.GetPrimAtPath(Sdf.Path(scene_path)).IsValid():
        scene = UsdPhysics.Scene.Define(stage, Sdf.Path(scene_path))
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)
        print("✅ 物理场景已创建 (重力: -Z, 9.81 m/s²)")
    
    # 添加地面
    ground_path = "/World/groundPlane"
    if not stage.GetPrimAtPath(Sdf.Path(ground_path)).IsValid():
        PhysicsSchemaTools.addGroundPlane(
            stage, ground_path, "Z", 1500,
            Gf.Vec3f(0, 0, 0),
            Gf.Vec3f(0.5)
        )
        print("✅ 地面已添加")
    
    # 添加环境光
    light_path = "/World/DistantLight"
    if not stage.GetPrimAtPath(Sdf.Path(light_path)).IsValid():
        distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path(light_path))
        distantLight.CreateIntensityAttr(500)
        print("✅ 环境光已添加")
    
    # 保存 USD 文件 (可选)
    if USD_OUTPUT_PATH:
        usd_dir = os.path.dirname(USD_OUTPUT_PATH)
        if not os.path.exists(usd_dir):
            os.makedirs(usd_dir)
        stage.GetRootLayer().Export(USD_OUTPUT_PATH)
        print(f"💾 USD 已保存到: {USD_OUTPUT_PATH}")
    
    print("")
    print("🎉 导入完成!")
    print("   - 按 Play 按钮开始仿真")
    print("   - 机器人应出现在原点位置")
    print("")
    
    return stage_path


# =========================================================
#  运行导入
# =========================================================
if __name__ == "__main__":
    import_robot()
else:
    # 在 Script Editor 中运行
    import_robot()
