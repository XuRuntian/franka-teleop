# franka-teleop

Standalone SpaceMouse teleoperation for Franka through the
`serl_franka_controllers` Cartesian impedance controller.

## Setup

```
git clone https://github.com/XuRuntian/franka-teleop.git && cd franka-teleop
```

Install dependencies:

```
pip install uv
uv sync
```

## Run Teleop

Start the HTTP/ROS bridge first:

```bash
source .venv/bin/activate
python franka_robot_server/franka_server.py --robot_ip=172.16.0.2 --gripper_type=Robotiq
```

Then start the decoupled SpaceMouse client:

```bash
python scripts/spacemouse_teleop.py --config configs/spacemouse_teleop.json
```

The client talks to `franka_server.py` through HTTP only and does not depend on
Gym/RL wrappers.

For a changed tool/TCP, edit `tool.ee_T_tcp` in
`configs/spacemouse_teleop.json`. The client applies teleop motion at the TCP
and converts the command back to the EE pose expected by the existing impedance
controller.

## Master Device Interface

SpaceMouse is implemented as one `MasterDevice`. Other master devices should
provide the same methods:

```python
class MasterDevice:
    def start(self) -> None: ...
    def close(self) -> None: ...
    def get_state(self) -> MasterState: ...
```

`MasterState.motion` is a relative 6D command:

```text
[x, y, z, roll, pitch, yaw]
```

The controller exposes `compute_command(robot_state, master_state)` for code
that needs the parsed command without sending it to the robot. The returned
`TeleopCommand` includes raw master input, scaled/clipped deltas, current TCP
pose, target TCP pose, and target EE pose.
