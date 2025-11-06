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
from model.wuji_hand import get_wujihand_config

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
    hand_cfg = get_wujihand_config("model/", side).replace(prim_path="/World/hand/WujiHand")
    hand = Articulation(cfg=hand_cfg)
    return {"hand": hand}

def run_simulator(sim, entities):
    """Run simulation with trajectory tracking."""
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Load trajectory
    traj_path = Path(__file__).parent / "data/wave.npy"
    trajectory = np.load(traj_path)
    
    # Joint mapping from MuJoCo to IsaacLab
    mujoco_joints = [f"finger{i}_joint{j}" for i in range(1,6) for j in range(1,5)]
    
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            for index, robot in enumerate(entities.values()):
                # set joint positions
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.set_joint_position_target(joint_pos) # you should ensure that target pos match the joint state writen into sim
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")

        for robot in entities.values():
            # Get target from trajectory
            joint_pos_raw = torch.from_numpy(trajectory[count % len(trajectory)])
            joint_pos_target = torch.zeros_like(joint_pos_raw).to(args_cli.device)
            
            # Map joints from MuJoCo to IsaacLab order
            for mujoco_idx, joint_name in enumerate(mujoco_joints):
                isaac_idx = robot.joint_names.index(joint_name)
                joint_pos_target[isaac_idx] = joint_pos_raw[mujoco_idx]
            
            # Apply action
            joint_pos_target = saturate(joint_pos_target, 
                                      robot.data.soft_joint_pos_limits[..., 0], 
                                      robot.data.soft_joint_pos_limits[..., 1])
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()
        
        sim.step()
        count += 1
        
        for robot in entities.values():
            robot.update(sim_dt)

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