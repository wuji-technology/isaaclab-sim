import os
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["ROS_PACKAGE_PATH"] = f"{current_dir}:" + os.environ.get(
    "ROS_PACKAGE_PATH", ""
)


def get_wujihand_config(model_base_dir, hand_side):
    # File paths
    urdf_path = f"{model_base_dir}{hand_side}.urdf"
    usd_dir = f"{current_dir}/usd"

    # PD control gains
    kp = {
        "finger(1|2|3|4|5)_joint(1|2)": 2,
        "finger(1|2|3|4|5)_joint3": 1,
        "finger(1|2|3|4|5)_joint4": 0.8,
    }
    kd = {
        "finger.*_joint(1|2)": 0.05,
        "finger.*_joint(3|4)": 0.03,
    }

    # Torque limits for each joint (Nm)
    effort_limits = {
        "finger(1|2|3|4|5)_joint(1|2)": 3,
        "finger(1|2|3|4|5)_joint3": 1.5,
        "finger(1|2|3|4|5)_joint4": 1,
    }

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            asset_path=urdf_path,
            usd_dir=usd_dir,
            usd_file_name="wujihand",
            force_usd_conversion=True,
            # Physics properties
            fix_base=True,
            root_link_name="palm_link",
            link_density=1,
            # Collision settings
            collider_type="convex_hull",
            self_collision=True,
            activate_contact_sensors=True,
            # Joint control - PD gains for implicit solver
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=kp,
                    damping=kd,
                ),
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=20,
                solver_velocity_iteration_count=10,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["finger.*_joint.*"],
                effort_limit_sim=effort_limits,
                stiffness=kp,
                damping=kd,
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
