import argparse
from isaaclab.app import AppLauncher

# Launch simulator
parser = argparse.ArgumentParser(description="WujiHand demonstration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from pathlib import Path
import omni.usd
from pxr import UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import saturate
import isaacsim.core.utils.prims as prim_utils
from wuji_hand import get_wujihand_config

HAND_SIDE = "right"


def design_scene():
    """Setup scene with WujiHand."""
    # Ground and lighting
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg.func("/World/light", cfg)

    # Create hand
    prim_utils.create_prim("/World/hand", "Xform")
    hand_cfg = get_wujihand_config("wuji_hand_description/urdf/", HAND_SIDE).replace(
        prim_path="/World/hand/WujiHand"
    )
    hand = Articulation(cfg=hand_cfg)

    # Filter collisions between palm and finger link2
    stage = omni.usd.get_context().get_stage()
    palm_prim = stage.GetPrimAtPath("/World/hand/WujiHand/palm_link")
    filtered_api = UsdPhysics.FilteredPairsAPI.Apply(palm_prim)
    for i in range(1, 6):
        finger_prim = stage.GetPrimAtPath(f"/World/hand/WujiHand/finger{i}_link2")
        filtered_api.CreateFilteredPairsRel().AddTarget(finger_prim.GetPath())

    return hand


def run_simulator(sim, hand):
    """Run simulation with trajectory tracking"""
    sim_dt = sim.get_physics_dt()
    count = 0

    trajectory = np.load(Path(__file__).parent / "data/wave.npy")
    mujoco_joints = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            joint_pos = hand.data.default_joint_pos.clone()
            joint_vel = hand.data.default_joint_vel.clone()
            hand.set_joint_position_target(joint_pos)
            hand.write_joint_state_to_sim(joint_pos, joint_vel)
            hand.reset()
            print("[INFO]: Resetting robot state...")

        # Load trajectory and map to joint order
        joint_pos_target = torch.zeros(len(hand.joint_names), device=args_cli.device)
        traj_data = trajectory[count % len(trajectory)]
        for mujoco_idx, joint_name in enumerate(mujoco_joints):
            if joint_name in hand.joint_names:
                joint_pos_target[hand.joint_names.index(joint_name)] = traj_data[
                    mujoco_idx
                ]

        # Apply joint limits and set targets
        joint_pos_target = saturate(
            joint_pos_target,
            hand.data.soft_joint_pos_limits[..., 0],
            hand.data.soft_joint_pos_limits[..., 1],
        )
        hand.set_joint_position_target(joint_pos_target)
        hand.write_data_to_sim()

        sim.step()
        count += 1
        hand.update(sim_dt)


def main():
    """Main simulation loop."""
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 100, device=args_cli.device)
    )
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_entities = design_scene()
    sim.reset()
    print("WujiHand simulation running...")
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
