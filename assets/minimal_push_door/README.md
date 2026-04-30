# Minimal Push Door Asset

Handle-free push-only door for the door-traversal training task.

## Files

- `solid_push_door.usda` — single rigid door panel with hinge revolute joint
- `minimal_push_door.usda` — full scene with room, door frame, and lighting
- `room_shell.usda` — room enclosure
- `door_side_walls.usda` — wall segments flanking the door frame

## Door Parameters

| Parameter | Nominal | Training Range |
|---|---:|---:|
| Door width | 0.90 m | [0.8, 1.0] m |
| Door thickness | 0.04 m | [0.02, 0.06] m |
| Door mass | 25 kg | [15, 75] kg |
| Hinge resistance | 5 Nm | [0, 30] Nm (0.2 prob zero) |
| Hinge air damping | 0 | [0, 4] Nms^2 |
| Hinge closer damping scale | — | alpha in [1.5, 3.0] (0.4 prob zero) |

## Door Structure

- Door panel body: `DoorLeaf`
- Single hinge revolute joint (no handle joint)
- Push-only: door opens in the positive-angle direction
- No handle prim, no contact patches, no grasp targets

## Notes

- No handle geometry. The entire front face is a valid push surface.
- Hinge joint is the only actuated/articulated door joint.
- Door angle range: [-0.05, ~1.57] rad.
