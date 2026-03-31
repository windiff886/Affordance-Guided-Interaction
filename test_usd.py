from pxr import Usd, UsdGeom
stage = Usd.Stage.Open("assets/robot/uni_dingo_dual_arm.usd")
for p in stage.Traverse():
    if "d455" in p.GetName() or "head" in p.GetName() or "pan_tilt" in p.GetName():
        print(p.GetPath())
