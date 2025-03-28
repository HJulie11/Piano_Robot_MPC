import numpy as np
from dm_control import mjcf
import mujoco
import mujoco_utils as mjcf_utils
import sys
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

from robopianist.models.piano import piano_constants as piano_consts
from robopianist.models.piano import piano_mjcf as piano_mjcf
from robopianist.models.hands import shadow_hand_constants as hand_consts
import mink
from loop_rate_limiters import RateLimiter

WHITE_KEY_INDICES = [
        0,
        2,
        3,
        5,
        7,
        8,
        10,
        12,
        14,
        15,
        17,
        19,
        20,
        22,
        24,
        26,
        27,
        29,
        31,
        32,
        34,
        36,
        38,
        39,
        41,
        43,
        44,
        46,
        48,
        50,
        51,
        53,
        55,
        56,
        58,
        60,
        62,
        63,
        65,
        67,
        68,
        70,
        72,
        74,
        75,
        77,
        79,
        80,
        82,
        84,
        86,
        87,
    ]

def create_joint_limits_dict(joint_names, action_spec, tendon_mapping=None):
    """
    Create a dictionary of joint limits from the action spec and joint names, accounting for tendon coupling (J0's).
    
    Parameters:
    - joint_names: list of strings
    - action_spec: list of tuples of floats

    Returns:
    - joint_limits_dict: dictionary
    """

    min_limits = action_spec.minimum[:-1]
    max_limits = action_spec.maximum[:-1]
    joint_limits = {}

    if tendon_mapping is None:
        # Case 1: Action spec includes all joints (22 DOF)
        if len(joint_names) != len(min_limits) or len(joint_names) != len(max_limits):
            raise ValueError(
                f"Mismatch between number of joint names ({len(joint_names)}) and "
                f"action spec dimensions (min: {len(min_limits)}, max: {len(max_limits)})"
            )
        
        for joint_name, min_val, max_val in zip(joint_names, min_limits, max_limits):
            joint_limits[joint_name] = (min_val, max_val)
    
    else:
        # Case 2: Action spec includes reduced DOF due to tendons
        # Create a mapping from action spec indices to joint names, accounting for tendons
        action_idx = 0
        action_joint_names = [] # List of joint/tendon names corresponding to action spec indices
        
        # Iterate over joint_names and map them to the action spec indices
        for joint_name in joint_names:
            is_coupled = False
            for tendon_name, coupled_joints in tendon_mapping.items():
                if joint_name in coupled_joints:
                    # Use the tendon name for coupled joints
                    action_joint_names.append(tendon_name)
                    joint_limits[joint_name] = (min_limits[action_idx], max_limits[action_idx])
                    is_coupled = True

                    # Increment the action spec index
                    if joint_name == coupled_joints[-1]:
                        action_idx += 1
                    break
            if not is_coupled:
                action_joint_names.append(joint_name)
                joint_limits[joint_name] = (min_limits[action_idx], max_limits[action_idx])
                action_idx += 1
        
        if action_idx != len(min_limits):
            raise ValueError(
                f"Action spec dimensions ({len(min_limits)}) do not match the number of joint names ({len(joint_names)})"
            )
    
    return joint_limits

def piano_key_geom_id(env, key_number):
    """
    Convert a site name to the corresponding geom name in the environment.
    """
    # return env.task.piano.keys[key_number].geom[0].
    geom_name = env.task.piano.keys[key_number].geom[0].name
    return env.physics.model.geom("piano/" + geom_name).id

def piano_key_site_id(env, key_number):
    """
    Convert a geom name to the corresponding site name in the environment.
    """
    site_name = env.task.piano.keys[key_number].site[0].name
    return env.physics.model.geom("piano/" + site_name).id

def add_mocap(env):
    physics = env.physics
    model = physics.model._model
    data = physics.data._data

    piano = env.task.piano

    mjcf_root = piano.mjcf_model

    keys = piano.keys
    sites = piano._sites
    mocap_bodies = []

    for key, site in zip(keys, sites):
        key_num = int(key.name.split("_")[-1])

        mocap_body = mjcf_root.worldbody.add(
            "body",
            name=f"key_mocap_{key_num}",
            mocap = "true",
            pos = site.pos,
            quat=[1, 0, 0, 0],
        )
        mocap_body.add(
            "site",
            name=f"key_mocap_site_{key_num}",
            pos=[0, 0, 0],
            size = [0.005, 0.005, 0.005],
            rgba=[1, 0, 0, 1],
        )
        mocap_body.add(mocap_body)

        mjcf_root.equality.add(
            "weld",
            name=f"weld_key_{key_num}",
            body1=mocap_body.name,
            body2=key.name,
            relpose = [0, 0, 0, 1, 0, 0, 0],
            solref = [0.02, 1],
            solimp = [0.9, 0.95, 0.001],
        )
    
    new_model = mjcf.compile(mjcf_root)
    new_data = mujoco.MjData(new_model)

    new_data.qpos[:min(len(data.qpos), len(new_data.qpos))] = data.qpos[:min(len(data.qpos), len(new_data.qpos))]
    new_data.qvel[:min(len(data.qvel), len(new_data.qvel))] = data.qvel[:min(len(data.qvel), len(new_data.qvel))]

    env.physics = mjcf.Physics.from_mjcf_model(new_model, data=new_data)
    physics = env.physics  # Update the physics reference
    model = physics.model._model
    data = physics.data._data

    piano._mjcf_root = mjcf_root
    piano._mocap_bodies = tuple(mocap_bodies)  # Store the new mocap bodies
    piano._parse_mjcf_elements()  # Re-parse MJCF elements if necessary

    return env

def construct_model(env):
    # piano root: env.task.piano.mjcf_model

    # root = mjcf.RootElement()
    root = env.task.root_entity.mjcf_model
    root.statistic.meansize = 0.08
    getattr(root.visual, "global").azimuth = 90
    getattr(root.visual, "global").elevation = -90

    # base = root.worldbody.add("body", name="hand_base")
    # width, height, depth = 0.1, 0.1, 0.1
    # base.add(
    #     "geom",
    #     type="box",
    #     size=[width, height, depth],
    #     density=1e-3,
    #     rgba=".9 .8 .6 1",
    # )

    # body = root.worldbody.add("body", name="base_target", mocap=True)
    # body.add(
    #     "geom",
    #     type="box",
    #     size=".03 .03 .03",
    #     contype="0",
    #     conaffinity="0",
    #     rgba=".3 .3 .6 .5",
    # )

    body = root.worldbody.add("body", name="th_target", mocap=True)
    body.add(
        "geom",
        type="sphere",
        size=".01 .01 .01",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="ff_target", mocap=True)
    body.add(
        "geom",
        type="sphere",
        size=".01 .01 .01",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="mf_target", mocap=True)
    body.add(
        "geom",
        type="sphere",
        size=".01 .01 .01",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="rf_target", mocap=True)
    body.add(
        "geom",
        type="sphere",
        size=".01 .01 .01",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    body = root.worldbody.add("body", name="lf_target", mocap=True)
    body.add(
        "geom",
        type="sphere",
        size=".01 .01 .01",
        contype="0",
        conaffinity="0",
        rgba=".3 .6 .3 .5",
    )

    return root
    # return mujoco.MjModel.from_xml_string(root.to_xml_string(), root.get_assets())    

def solve_ik(env, fingertip_site, target_key_site, rate, press=True):
    env._physics_proxy = env.task.root_entity.physics = mjcf.Physics.from_mjcf_model(construct_model(env))

    configuration = mink.Configuration(env._physics_proxy.model._model)

    model = env._physics_proxy.model._model
    data = env._physics_proxy.data._data
    
    tasks = [
        fingertip_task := mink.FrameTask(
            frame_name=fingertip_site,
            frame_type="site",
            position_cost = 1.0,
            orientation_cost = 1.0,
            gain = 1.0,
            lm_damping = 0.0,
        ),
    ]

    if "th" in fingertip_site:
        target_mocap_name = "th_target"
        target_mocap_id = model.body("th_target").mocapid[0]
    elif "ff" in fingertip_site:
        target_mocap_name = "ff_target"
        target_mocap_id = model.body("ff_target").mocapid[0]
    elif "mf" in fingertip_site:
        target_mocap_name = "mf_target"
        target_mocap_id = model.body("mf_target").mocapid[0]
    elif "rf" in fingertip_site:
        target_mocap_name = "rf_target"
        target_mocap_id = model.body("rf_target").mocapid[0]
    elif "lf" in fingertip_site:
        target_mocap_name = "lf_target"
        target_mocap_id = model.body("lf_target").mocapid[0]
    else:
        raise ValueError(f"Unknown fingertip site: {fingertip_site}")
        
    print(f"target_mocap_name: {target_mocap_name}, target_mocap_id: {target_mocap_id}")
        
    target_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target_key_site)
    target_key_geom_name = f"piano/{env.task.piano.keys[target_key_id].geom[0].name}"
    target_key_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, target_key_geom_name)
    solver = "quadprog"

    # for idx, key in enumerate(target_key_sites):
    transform_fingertip_target_to_world = (
        configuration.get_transform_frame_to_world(target_key_geom_name, "geom")
    )

    key_translation = transform_fingertip_target_to_world.translation()
    fingertip_task.set_target(transform_fingertip_target_to_world)
    mink.move_mocap_to_frame(model, data, target_mocap_name, fingertip_site, "site")

    data.mocap_pos[target_mocap_id] = key_translation
    if not press:
        data.mocap_pos[target_mocap_id][2] += 0.02
        data.mocap_pos[target_mocap_id][0] += 0.05
    data.mocap_pos[target_mocap_id][2] -= 0.02
    data.mocap_pos[target_mocap_id][0] += 0.05
    fingertip_task.set_target(mink.SE3.from_mocap_id(data, target_mocap_id))

    vel=mink.solve_ik(configuration, tasks, rate.dt, solver, damping=1e-3)
    configuration.integrate_inplace(vel, rate.dt)
    # mujoco.mj_camlight(model, data)

    return configuration.q.copy()
    # data.qpos[:] = configuration.q
    # mujoco.mj_step(model, data)

    # viewer.sync()
    # rate.sleep()

def jointlimitscost(joint_limits, qpos):
        '''Return cost of joint limits'''
        return np.sum([max(0.0,low-q) + max(0.0,q-high) for q,(low,high) in zip(qpos, joint_limits)])

def jointlimitsviolated(joint_limits, qpos):
    '''Return true if config not in joint limits'''
    print(f"Joint limits cost: {jointlimitscost(joint_limits, qpos)}")
    return jointlimitscost(joint_limits, qpos) > 0.0


def projecttojointlimits(joint_limits, qpos):
    '''Project config to joint limits'''
    # return np.clip(qpos, self.JOINT_LIMITS[:,0], self.JOINT_LIMITS[:,1])
    return np.minimum(np.maximum(qpos, joint_limits[:,0]), joint_limits[:,1])

def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier

def get_fingertip_z(phi: np.ndarray, lift_height: float = 0.08) -> np.ndarray:
    """Get the fingertip z-coordinate given the finger joint angles."""
    x = (phi + np.pi) / (2 * np.pi) # Normalize to [0, 1]
    x = np.clip(x, 0, 1)
    return np.where(
        x <= 0.5,
        cubic_bezier_interpolation(0, lift_height, 2 * x), # Lift
        cubic_bezier_interpolation(lift_height, 0, 2 * (x - 0.5)), # Lower
    )

def fingertip_mocap(fingertip_list):
    fingertip_mocap_name = []
    for fingertip in fingertip_list:
        if "th" in fingertip:
            target_mocap_name = "th_target"
        elif "ff" in fingertip:
            target_mocap_name = "ff_target"
        elif "mf" in fingertip:
            target_mocap_name = "mf_target"
        elif "rf" in fingertip:
            target_mocap_name = "rf_target"
        elif "lf" in fingertip:
            target_mocap_name = "lf_target"
        else:
            raise ValueError(f"Unknown fingertip site: {fingertip}")

        fingertip_mocap_name.append(target_mocap_name)
    
    return fingertip_mocap_name

def get_key_pos(env, key_number, fingertip):
    white_key_inds = WHITE_KEY_INDICES
    site = env.task.piano._sites[key_number]
    if "white" in site.name and "th" in fingertip:
        key_inds = white_key_inds.index(key_number)
        x = white_key_inds[key_inds + 1]
        site = env.task.piano._sites[x]
        key_pos = env.physics.bind(site).xpos.copy()
        key_pos[2] -= 0.01
        # key_pos[0] -= 0.005
        # key_pos[1] -= 0.03
    else:
        key_pos = env.physics.bind(site).xpos.copy()
        key_pos[2] -= 0.02
        # key_pos[1] -= 0.05
    return key_pos
    
