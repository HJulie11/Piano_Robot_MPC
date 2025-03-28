import mujoco
import numpy as np
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation as R
import pinocchio as pin
from dm_control import mjcf

import sys
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

import robopianist
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand

def compute_jacobian(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_name: str, # "rh_shadow_hand/thdistal_site"
):
    site_id = model.site(site_name).id
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack((jacp, jacr))


def compute_jacobian_multi(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        site_names: list,
):
    J_multi = []
    for site_name in site_names:
        site_id = model.site(site_name).id
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack((jacp, jacr))
        J_multi.extend(J)
    return J_multi

def check_collision(env, model, data, excluded_geom_pairs=None):
    """
    Checks for collision betwee geoms in the MuJoCo simulation.
    
    Parameters:
        env: The simulation environment.
        model: The MuJoCo model.
        data: The MuJoCo data structure.
        excluded_geom_pairs (set of tuple, optional): Geom ID pairs to ignore during collision detection.
    
    Returns:
        True if a collision is detected, False otherwise.
    """
    mujoco.mj_forward(model, data)
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id, geom2_id = contact.geom1, contact.geom2

        if excluded_geom_pairs and (geom1_id, geom2_id) in excluded_geom_pairs:
            continue

        geom1_name = model.geom_id2name(geom1_id)
        geom2_name = model.geom_id2name(geom2_id)

        if "finger" in geom1_name and "finger" in geom2_name:
            print(f"Collision detected between {geom1_name} and {geom2_name}")
            return True
    
    return False

def compute_fk(model, data, site_name):
    """Get forward kinematics (world position) of a body."""
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id
    return np.array(data.site_xpos[site_id].copy()), R.from_matrix(data.site_xmat[site_id].copy().reshape(3, 3)).as_quat()

def compute_fk_multiple_fingertips(model, data, all_fingertip_names):
    """Get forward kinematics (world position) of the fingertips."""
    mujoco.mj_forward(model, data)
    site_inds = [model.site(name).id for name in all_fingertip_names]
    
    return np.array(data.site_xpos[site_inds].copy()), [R.from_matrix(data.site_xmat[i].reshape(3, 3)).as_quat() for i in site_inds]

# geom instead of site
def compute_fk_multiple_geom(model, data, geom_inds):
    """Get forward kinematics (world position) of the fingertips."""
    mujoco.mj_forward(model, data)
    # geom_inds = [model.geom(name).id for name in all_fingertip_names]
    
    return np.array(data.geom_xpos[geom_inds].copy()), [R.from_matrix(data.geom_xmat[i].reshape(3, 3)).as_quat() for i in geom_inds]

def compute_fk_geom(model, data, geom_id):
    """Get forward kinematics (world position) of a body."""
    mujoco.mj_forward(model, data)
    # site_id = model.site(site_name).id
    return np.array(data.geom_xpos[geom_id].copy()), R.from_matrix(data.geom_xmat[geom_id].copy().reshape(3, 3)).as_quat()

def distance_finger_to_key(env, fingertip_site_id, geom_id):
    fingertip_pos = env.physics.data.site_xpos[fingertip_site_id].copy()

    key_geom_pos = env.physics.data.geom_xpos[geom_id].copy()

    fingertip_site_name = env.physics.model.site(fingertip_site_id).name
    
    key_geom_pos[-1] += 2.0 * env.physics.model.geom_size[geom_id, 2].copy()
    key_geom_pos[0] += 1.0 * env.physics.model.geom_size[geom_id, 0].copy()
    
    diff = key_geom_pos -  fingertip_pos
    distance = float(np.linalg.norm(diff))
    print(f"distance between {fingertip_site_name} and key: {distance}")
    return distance

def distance_z_surface(env, fingertip_site_id, geom_id):
    fingertip_z = env.physics.data.site_xpos[fingertip_site_id][-1].copy()  # Extract Z coordinate of the fingertip
    key_geom_z = env.physics.data.geom_xpos[geom_id][-1].copy()  # Extract Z coordinate of the key

    # Compute absolute difference in Z-axis only
    z_distance = abs(fingertip_z - key_geom_z)
    
    print(f"Z-distance between fingertip {env.physics.model.site(fingertip_site_id).name} and key: {z_distance}")
    return z_distance

    

# def collision_check(env, geom_name):
#     return env.physics.model.geom(geom_name) in env.physics.data.contact

def compute_goal_orientation():
    """Define the desired quaternion for piano key pressing."""
    # Align fingertip's z-axis with negative world z-axis
    desired_rot = np.array([[1, 0, 0], # X-axis remains the same
                            [0, 1, 0], # Y-axis remains the same
                            [0, 0, -1]]) # Z-axis points downwards
    return R.from_matrix(desired_rot).as_quat() # Convert to quaternion

def compute_goal_base_pos(env, model, data, target_qpos, qpos_inds):
    """Compute the desired base position for the given joint configuration."""
    hand_base = env.task._hand.root_body
    hand_base_id = model.body("rh_shadow_hand/" + hand_base.name).id
    print(f"hand pose before update: {data.xpos[hand_base_id]}")
    data.qpos[qpos_inds] = target_qpos
    mujoco.mj_forward(model, data)
    # hand_base_pos = np.array(env.task._hand.root_body.pos)
    
    hand_base_pos = data.xpos[hand_base_id]
    print(f"hand pose after update: {hand_base_pos}")
    return hand_base_pos

def compute_orientation_error(current_quat, goal_quat):
    """Compute the orientation error between current and goal quaternion."""
    # Convert to rotation objects
    current_rot = R.from_quat(current_quat)
    goal_rot = R.from_quat(goal_quat)

    rot_diff = goal_rot * current_rot.inv() # or inv(current_rot)

    rot_vector = rot_diff.as_rotvec() # Shape: (3,)
    return rot_vector

def compute_goal_quat(env, model, data, target_pos, qpos_inds):
    """Compute the desired base position for the given end effector position (for five fingers and x, y, z for each of the end effectors)."""
    site_names = ["rh_shadow_hand/thdistal_site", "rh_shadow_hand/ffdistal_site", "rh_shadow_hand/mfdistal_site", "rh_shadow_hand/rfdistal_site", "rh_shadow_hand/lfdistal_site"]
    site_inds = [model.site(name).id for name in site_names]
    goal_quats = []
    for site_id in site_inds:
        data.site_xpos[site_inds] = target_pos
        mujoco.mj_forward(model, data)
        goal_quats.append(R.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_quat())
    return goal_quats

def compute_ik_multiple_keys(env, model, data, q_current, fingertip_names, key_numbers, q_target, qpos_inds):
    """Find a collision-free qpos for mulitple keys - recursively call compute_ik."""
    # q_current[-1] -= 1.0 # Move the hand backwards (toward the player) a bit
    # q_current[0] -= 0.1 # wrist slightly inwards
    data.qpos[qpos_inds] = q_current
    mujoco.mj_forward(model, data)
    
    # goal_orientation = np.array(goal_orientation)
    site_inds = [model.site(name).id for name in fingertip_names]

    # Find geom ids for the keys
    geoms = [env.task.piano.keys[number].geom[0] for number in key_numbers]
    geom_ids = [env.physics.model.geom(f"piano/{geom.name}").id for geom in geoms]

    goal_quats = compute_goal_quat(env, model, data, q_target, qpos_inds)
    print(f"goal quats: {goal_quats}")

    for i in range(500):
        print(f"Iteration {i}")
        # Get current end-effector position
        pos, quats = compute_fk_multiple_fingertips(model, data, fingertip_names) 
        # pos, quats = compute_fk_multiple_geom(model, data, geom_ids)         

        # Compute error
        pos_error = q_target - pos # Position error
        rot_errors = []
        # for idx, quat, goal_quat in enumerate(zip(quats, goal_quats)):
        for quat, goal_quat in zip(quats, goal_quats):
            rot_error = compute_orientation_error(quat, goal_quat) # Orientation error
            rot_errors.append(rot_error)

        errors = np.hstack((pos_error, np.array(rot_errors)))
        
        if all(distance_finger_to_key(env, site_id, geom_id) in np.arange(0.1, 0.101) for site_id, geom_id in zip(site_inds, geom_ids)):
            return data.qpos[qpos_inds].copy(), True

        J_multi = compute_jacobian_multi(model, data, fingertip_names)
        dq = pinv(J_multi) @ errors.flatten()
        print(f"dq: {dq}")
        data.qpos[:] += dq.flatten() * 5e-2 # small step update

        mujoco.mj_forward(model, data)
    
    return data.qpos[qpos_inds].copy(), False # Return final pose with success flag

def compute_ik(env, model, data, q_current, geom_id, q_target, key_number, site_name, qpos_inds):
    """ Find a collision-free grasping pose using MuJoCo. """
    data.qpos[qpos_inds] = q_current
    mujoco.mj_forward(model, data)

    goal_quat = compute_goal_orientation()

    for i in range(300):
        # print(f"Iteration {i}")
        # Get currend end-effector position
        pos, quat = compute_fk(model, data, site_name)
        
        # Compute error
        pos_error = q_target - pos # Position error
        rot_error = compute_orientation_error(quat, goal_quat) # Orientation error
        error = np.hstack((pos_error, rot_error))
        # print(f"Iteration {i}: Position Error {np.linalg.norm(pos_error)}, Orientation Error {np.linalg.norm(rot_error)}")
        
        # if np.linalg.norm(pos_error) < 1e-3 and np.linalg.norm(rot_error) < 1e-3:
        if distance_finger_to_key(env, env.physics.model.site(site_name).id, geom_id) < 0.1:
            return data.qpos[qpos_inds].copy(), True # Success
        
        # Compute Jacobian
        J = compute_jacobian(model, data, site_name)

        dq = pinv(J) @ error
        data.qpos[:] += dq.flatten() * 1e-2 # small step update

        mujoco.mj_forward(model, data)
    
    return data.qpos[qpos_inds].copy(), False # Return final pose with success flag

def reconstruct_path( last_index):
    """Reconstructs the path from the goal to the start using qpos, ensuring each node is (23,) shaped."""
    path = []
    current_index = last_index

    while current_index is not None:
        _, qpos = tree[current_index]  # Extract qpos from the stored tuple (pos, qpos)
        
        # Ensure qpos is (23,) by appending a 0 at the end
        qpos_extended = np.append(qpos, 0)  # Converts (22,) to (23,)

        path.append(qpos_extended)
        current_index = parents.get(current_index)

    return path[::-1]  # Reverse to get path from start to goal

def _update_right_hand_pos(hand, physics: mjcf.Physics):
                
        stringpos0=[]
        stringpluck=[0 for i in range(6)]

        for i in range(4):
            if _fingerstate[i]!=-1:
                site_pos=_fingerPos[i][_finger_tick_id[i]]
                ikresult=qpos_from_site_pose(physics,'rh_shadow_hand/'+right_hand.fingertip_sites[i].name
                            ,site_pos,None,FINGER_JOINT[i])
                if(ikresult.success==True):
                    physics.data.qpos=ikresult.qpos
                _finger_tick_id[i]+=1
                if wait==0:
                    if _finger_tick_id[i]==_ANIMATION_NUM:
                        _string_play(_fingerstring[i]-1)
                        stringpluck[_fingerstring[i]-1]=1
                if wait==1:
                #在10个tick内 如果有一个tick达到了目标 就弹奏，否则在最后一个tick演奏
                    if _fingerstate[i]==_string_max_pos[_fingerstring[i]-1]:
                        if _fingerstate[i]==0:
                            #如果与该空弦时间相同的弦，也按住了正确的位置，此时空弦才发出声响
                            stringpos0.append(i)
                            pass
                        else:
                            _string_play(_fingerstring[i]-1)
                            _fingerstate[i]=-1
                if(_finger_tick_id[i]>=len(_fingerPos[i])):
                    _finger_tick_id[i]=0
                    _fingerstate[i]=-1
                    _fingerPos[i][0]=site_pos
                    _fingerstring[i]=0
                _last_finger_qpos[i]=physics.bind(_fingers_joint[i]).qpos.copy()
                
            else:
                physics.bind(_fingers_joint[i]).qpos=_last_finger_qpos[i]

        #如果与该空弦时间相同的弦，也按住了正确的位置，此时空弦才发出声响
        if wait:
            for i in stringpos0:
                if _fingerstate[i]==_string_max_pos[_fingerstring[i]-1]:
                    can = 1
                    for j in range(4):
                        if _fingerstate[j]!=0 and _fingerstate[j]!=-1:
                            can=0
                    if can:
                        _string_play(_fingerstring[i]-1)
                        _fingerstate[i]=-1
                    
        physics.bind(_fingers_joint[4]).qpos=_last_finger_qpos[4]
        _stringpluck=stringpluck