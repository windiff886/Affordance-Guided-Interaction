from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from omni.isaac.sensor import Camera
import numpy as np

cam = Camera("/World/Cam", resolution=(640, 480))
cam.set_focal_length(1.93)
cam.set_horizontal_aperture(3.8) # arbitrary
print("SUCCESS")
app.close()
