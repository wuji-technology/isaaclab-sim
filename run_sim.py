"""
WujiHand Isaac Sim demonstration with trajectory tracking.

Usage:
    python run_sim.py
    python run_sim.py --side left
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="WujiHand demonstration")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--side", choices=["left", "right"], default="right")
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path
import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import saturate

SIDE = args_cli.side
PKG_DIR = Path(__file__).resolve().parent
DESC_DIR = PKG_DIR / "wuji_hand_description"

HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(DESC_DIR / "usd" / SIDE / "wujihand.usd"),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={f"{SIDE}_finger.*_joint1": 0.06},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[f"{SIDE}_finger.*_joint.*"],
            stiffness=None,
            damping=None,
        ),
    },
)


def main():
    """Set up scene with Wuji Hand and run trajectory tracking loop."""
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 100, device=args_cli.device)
    )
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg.func("/World/light", cfg)

    prim_utils.create_prim("/World/hand", "Xform")
    hand = Articulation(cfg=HAND_CFG.replace(prim_path="/World/hand/WujiHand"))

    sim.reset()
    sim_dt = sim.get_physics_dt()
    count = 0

    trajectory = np.load(PKG_DIR / "data" / "wave.npy")
    mujoco_joints = [
        f"{SIDE}_finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)
    ]
    joint_name_to_idx = {name: idx for idx, name in enumerate(hand.joint_names)}
    mj_indices = []
    hand_indices = []
    for mj_idx, name in enumerate(mujoco_joints):
        if name in joint_name_to_idx:
            mj_indices.append(mj_idx)
            hand_indices.append(joint_name_to_idx[name])
    hand_idx_t = torch.tensor(hand_indices, device=args_cli.device)
    joint_pos_target = torch.zeros(len(hand.joint_names), device=args_cli.device)

    print("WujiHand simulation running...")
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            default_pos = hand.data.default_joint_pos.clone()
            hand.set_joint_position_target(default_pos)
            hand.write_joint_state_to_sim(default_pos, hand.data.default_joint_vel.clone())
            hand.reset()
            print("[INFO] Resetting robot state...")

        joint_pos_target.zero_()
        traj_data = trajectory[count % len(trajectory)]
        joint_pos_target[hand_idx_t] = torch.from_numpy(
            traj_data[mj_indices].astype(np.float32)
        ).to(device=args_cli.device)

        clamped = saturate(
            joint_pos_target,
            hand.data.soft_joint_pos_limits[0, :, 0],
            hand.data.soft_joint_pos_limits[0, :, 1],
        )
        hand.set_joint_position_target(clamped)
        hand.write_data_to_sim()

        sim.step()
        count += 1
        hand.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
