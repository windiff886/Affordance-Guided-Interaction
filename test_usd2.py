from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni
from pxr import Usd, UsdGeom
import sys
stage = omni.usd.get_context().get_stage()
# load usd
from pxr import Sdf
stage = Usd.Stage.Open("assets/robot/uni_dingo_dual_arm.usd")
paths = []
for p in stage.Traverse():
    if "d455" in p.GetName() or "camera" in p.GetName():
        paths.append(str(p.GetPath()))
for path in paths:
    print(path)
app.close()
