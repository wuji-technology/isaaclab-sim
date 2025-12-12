import argparse
from isaaclab.app import AppLauncher

# Launch simulator
parser = argparse.ArgumentParser(description="WujiHand demonstration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaacsim.core.utils.prims as prim_utils
from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.utils.math import saturate
from wuji_hand import get_wujihand_config
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import omni.usd
from pxr import UsdPhysics

side = "right"

def design_scene():
    """Setup scene with WujiHand."""
    # Ground and lighting
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg.func("/World/light",cfg)
    
    # Create hand
    prim_utils.create_prim("/World/hand", "Xform")
    hand_cfg = get_wujihand_config("wujihand-urdf/urdf/", side).replace(prim_path="/World/hand/WujiHand")
    hand = Articulation(cfg=hand_cfg)

    robot_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/hand/WujiHand/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=True,
    )
    contact_sensor = ContactSensor(cfg=robot_contact_sensor_cfg)

    return {"hand": hand, "contact_sensor": contact_sensor}


def run_simulator(sim, entities):
    """Run simulation with trajectory tracking and sin motion on joint2."""
    sim_dt = sim.get_physics_dt()
    count = 0

    hand = entities["hand"]
    contact_sensor = entities["contact_sensor"]
    trajectory = np.load(Path(__file__).parent / "data/wave.npy")
    mujoco_joints = [f"finger{i}_joint{j}" for i in range(1, 6) for j in range(1, 5)]

    # Find joint2 indices
    joint2_indices = [
        hand.joint_names.index(f"finger{i}_joint2")
        for i in range(1, 6)
        if f"finger{i}_joint2" in hand.joint_names
    ]

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            joint_pos = hand.data.default_joint_pos.clone()
            joint_vel = hand.data.default_joint_vel.clone()
            hand.set_joint_position_target(joint_pos)
            hand.write_joint_state_to_sim(joint_pos, joint_vel)
            hand.reset()
            contact_sensor.reset()
            print("[INFO]: Resetting robot state...")

        # Load trajectory and map to joint order
        joint_pos_target = torch.zeros(len(hand.joint_names), device=args_cli.device)
        traj_data = trajectory[count % len(trajectory)]
        for mujoco_idx, joint_name in enumerate(mujoco_joints):
            if joint_name in hand.joint_names:
                joint_pos_target[hand.joint_names.index(joint_name)] = traj_data[
                    mujoco_idx
                ]

        # Add sin motion to joint2
        sin_offset = 0.3 * torch.sin(
            torch.tensor(2 * np.pi * 0.5 * count * sim_dt, device=args_cli.device)
        )
        for joint2_idx in joint2_indices:
            joint_pos_target[joint2_idx] += sin_offset * (2 * (joint2_idx % 2) - 1)

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
        contact_sensor.update(sim_dt)

        # Check contacts
        if contact_sensor.data.net_forces_w is not None:
            force_mags = torch.norm(contact_sensor.data.net_forces_w, dim=-1)[0]
            contacting = force_mags > 0.01

            if contacting.any() and count % 100 == 0:
                indices = torch.where(contacting)[0]
                links = [contact_sensor.body_names[i] for i in indices.cpu().tolist()]
                print(
                    f"[INFO]: Contact detected: {len(indices)} bodies, max force: {force_mags.max():.3f} N"
                )
                print(f"[INFO]: Contacting links: {', '.join(links)}")


def main():
    """Main simulation loop."""
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/100, device=args_cli.device))
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    scene_entities = design_scene()
    sim.reset()
    print("WujiHand simulation running...")
    run_simulator(sim, scene_entities)

if __name__ == "__main__":
    main()
    simulation_app.close()