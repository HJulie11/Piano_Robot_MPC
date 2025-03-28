import mujoco
import mink # ik solver
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize

from .get_config import compute_ik, compute_fk
from action.action import HandController

import sys
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

import robopianist
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from . import ik_2 as ik


# Define the joint names and indices
def get_consts_rrt(env):
    
    # get joints with _J1 and _J2 inter-connected in <tendon> 
    # (.joints gets 26, shape, .actuators get 22, shape)
    # last two elements left out as they are the hand positions
    joints = env.task._hand.actuators[:-2] 
    hand_positions = env.task._hand.actuators[-2:]

    JOINT_NAMES = [joint.name for joint in joints]
    HAND_POSITIONS = [hand.name for hand in hand_positions]

    JOINT_INDS = [] # len = 20
    for joint in joints:
        if joint.joint:   
            JOINT_INDS.append(env.physics.model._model.joint("rh_shadow_hand/" + joint.joint.name).id)
        elif joint.tendon:
            JOINT_INDS.append(env.physics.model._model.tendon("rh_shadow_hand/" + joint.tendon.name).id)

        else:
            raise ValueError(f"Joint or tendon not found for {joint.name}")
        
    HAND_INDS = [] # len = 2
    for hand in hand_positions:
        if hand.joint:
            HAND_INDS.append(env.physics.model._model.joint("rh_shadow_hand/" + hand.joint.name).id)
        elif hand.tendon:
            HAND_INDS.append(env.physics.model._model.tendon("rh_shadow_hand/" + hand.tendon.name).id)
        else:
            raise ValueError(f"Joint or tendon not found for {hand.name}")
        
    action_spec = env.action_spec()
    JOINT_LIMITS = np.array([(low, high) for low, high in zip(action_spec.minimum[:22], action_spec.maximum[:22])])
    BOUNDS = [(low, high) for low, high in zip(action_spec.minimum[-3:-1], action_spec.maximum[-3:-1])]


    return JOINT_NAMES, JOINT_INDS, HAND_POSITIONS, HAND_INDS, BOUNDS, JOINT_LIMITS

def initial_config(env, joint_inds):
    config = np.zeros(23)
    config[:len(joint_inds)] = env.physics.data.qpos[joint_inds].copy()
    return config

def distance_finger_to_key(env, fingertip_site_id, geom_id):
    # fingertip_site = env.task._hand.fingertip_sites[fingertip_number - 1]
    # # print(f"Fingertip site: {fingertip_id}")
    # fingertip_site_idx = env.physics.model.site(fingertip_site).id
    fingertip_pos = env.physics.data.site_xpos[fingertip_site_id].copy()

    key_geom_pos = env.physics.data.geom_xpos[geom_id].copy()
    key_geom_pos[-1] += 0.5 * env.physics.model.geom_size[geom_id, 2].copy()
    key_geom_pos[0] += 0.35 * env.physics.model.geom_size[geom_id, 0].copy()
    
    diff = key_geom_pos - fingertip_pos
    distance = float(np.linalg.norm(diff))
    return distance

class RRTPlanner:
    # def __init__(self, env, goal_key_number, site_name, max_iter=50, step_size=0.2):
    def __init__(self, env, goal_pos, site_name, goal_key_number=None, max_iter=50, step_size=0.2):
        self.env = env
        self.fingertip = site_name
        self.fingertip_id = env.physics.model.site(site_name).id
        self.goal_key_number = goal_key_number
        if self.goal_key_number is None:
            self.goal = goal_pos
        else:
            self.key_name = env.task.piano._sites[goal_key_number].name
            self.key_geom_name = env.task.piano.keys[goal_key_number].geom[0].name
            self.key_geom_id = env.physics.model.geom("piano/" + self.key_geom_name).id
            self.key_id = env.physics.model.site("piano/" + self.key_name).id
            self.goal = env.physics.data.site_xpos[self.key_id].copy()
        self.max_iter = max_iter
        self.step_size = step_size
        self.pre_move_done = False

        fingertip_number = 0
        if "th" in self.fingertip:
            fingertip_number = 1
        elif "ff" in self.fingertip:
            fingertip_number = 2
        elif "mf" in self.fingertip:
            fingertip_number = 3
        elif "rf" in self.fingertip:
            fingertip_number = 4
        elif "lf" in self.fingertip:
            fingertip_number = 5
        else:
            raise ValueError(f"Fingertip name {self.fingertip} not found.")
        self.fingertip_number = fingertip_number

        self.JOINT_NAMES, self.JOINT_INDS, self.HAND_POSITIONS, self.HAND_INDS, self.BOUNDS, self.JOINT_LIMITS = get_consts_rrt(env)
        self.QPOS_NAMES = self.JOINT_NAMES + self.HAND_POSITIONS
        self.QPOS_INDS = self.JOINT_INDS + self.HAND_INDS

        self.hand_controller = HandController(env, self.fingertip, self.QPOS_INDS)
        self.FINGERTIP_INDS = [env.physics.model.site("rh_shadow_hand/" + site.name).id for site in env.task._hand.fingertip_sites]
        self.PIANO_KEY_INDS = [env.physics.model.site("piano/" + key.name).id for key in env.task.piano._sites]

        initial_qpos = np.array(env.physics.data.qpos[self.QPOS_INDS].copy())
        initial_fingertip_pos = np.array(env.physics.data.site_xpos[self.fingertip_id].copy())
        self.tree = [(initial_fingertip_pos, initial_qpos)]
        self.parents = {0: None}

        # Initialize IK solver
        self.ik_solver = ik.InverseKinematicsSolver(env, [], get_consts_rrt(env))  # Assuming no coupled joints for now

    def inverse_kinematics(self, target_position, site_name):
        """Adapt IK solver for RRT use."""
        # Create target dictionary for the designated finger
        finger = self.fingertip
        print(f"fingertip: {finger}")
        finger_map = {
            "rh_shadow_hand/thdistal_site": "thumb", 
            "rh_shadow_hand/ffdistal_site": "index",
            "rh_shadow_hand/mfdistal_site": "middle", 
            "rh_shadow_hand/rfdistal_site": "ring", 
            "rh_shadow_hand/lfdistal_site": "little"}
        
        finger_full = finger_map[finger]
        q_target_dict = {finger_full: target_position}
        site_names = {finger_full: finger}

        q_solution = self.ik_solver.solve_ik(
            self.env, self.env.physics.data, self.env.physics.data.qpos[self.QPOS_INDS].copy(),
            q_target_dict, site_names, position_only=True  # Start with position-only for movement
        )
        success = np.linalg.norm(target_position - self.ik_solver.compute_fk(self.env, self.env.physics.data._data, site_name)[0]) < 1e-2
        print(f"IK success: {success}, Config: {q_solution}")
        return q_solution, success

    def forward_kinematics(self, qpos):
        self.env.physics.data.qpos[self.QPOS_INDS] = qpos
        mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
        return self.env.physics.data.site_xpos[self.fingertip_id].copy()

    def collision_check(self, qpos):
        self.env.physics.data.qpos[self.QPOS_INDS] = qpos
        self.env.physics.forward()

        fingertip_pos = self.env.physics.data.site_xpos[self.fingertip_id].copy()
        key_site_pos = self.env.physics.data.site_xpos[self.key_id].copy()
        key_site_pos[-1] += 0.5 * self.env.physics.model.site_size[self.key_id, 2]
        key_site_pos[0] += 0.35 * self.env.physics.model.site_size[self.key_id, 0]

        finger_geom_pos = self.env.physics.data.geom_xpos[self.fingertip_id].copy()
        finger_geom_pos[-1] += 0.5 * self.env.physics.model.geom_size[self.fingertip_id, 2]
        finger_geom_pos[0] += 0.35 * self.env.physics.model.geom_size[self.fingertip_id, 0]

        # Check other keys
        for other_key_id in self.PIANO_KEY_INDS:
            if other_key_id == self.key_id:
                continue
            distance = distance_finger_to_key(self.env, self.fingertip_id, other_key_id)
            if distance < 0.1:
                print(f"Fingertip is close to another key: {distance}")
                return True

        # Check other fingers
        for site_id in self.FINGERTIP_INDS:
            if site_id == self.fingertip_id:
                continue
            for key_id in self.PIANO_KEY_INDS:
                distance = distance_finger_to_key(self.env, site_id, key_id)
                if distance < 0.1:
                    print(f"Other finger {site_id} is close to a key: {distance}")
                    return True

        return self.ik_solver.jointlimitsviolated(qpos)  # Use IK's joint limit check

    def nearest_node_idx(self, random_qpos):
        tree_pos = np.array([node[1] for node in self.tree])
        tree = KDTree(tree_pos)
        _, ind = tree.query(random_qpos)
        return ind

    def lerp(self, pos1, pos2, t):
        return (1 - t) * pos1 + t * pos2

    def steer(self, near_qpos, near_pos, random_target_pos, discretisationsteps, delta_q=None):
        end_pos = random_target_pos.copy()
        dist = np.linalg.norm(random_target_pos - near_pos)
        if delta_q is not None and dist > delta_q:
            end_pos = self.lerp(near_pos, random_target_pos, delta_q / dist)
            dist = delta_q

        dt = dist / discretisationsteps
        last_valid_pos = near_pos
        last_valid_qpos = near_qpos

        for i in range(1, discretisationsteps + 1):
            t = (dt * i) / dist
            lerp_pos = self.lerp(near_pos, end_pos, t)
            lerp_qpos, success = self.inverse_kinematics(lerp_pos, self.fingertip)
            if success:
                last_valid_pos = lerp_pos
                last_valid_qpos = lerp_qpos
            else:
                if i == 1:
                    return near_pos, near_qpos
                break
        return last_valid_pos, last_valid_qpos

    def random_configuration(self, goal_key_pos):
        if np.random.rand() < 0.3:
            random_pos = goal_key_pos + np.random.uniform(-0.05, 0.05, size=goal_key_pos.shape)
        else:
            random_pos = np.random.uniform(low=-1.5, high=1.5, size=goal_key_pos.shape)
        return random_pos

    def add_node(self, fingertip_pos, qpos, parent_idx):
        self.tree.append((np.array(fingertip_pos), np.array(qpos)))
        self.parents[len(self.tree) - 1] = parent_idx

    def move_hand_above_key(self):
        current_pos = self.env.physics.data.qpos[self.HAND_INDS].copy()
        target_pos = self.goal[:2]
        direction = (target_pos - current_pos) / np.linalg.norm(target_pos - current_pos)
        step = self.step_size * direction
        new_pos = np.clip(current_pos + step, [b[0] for b in self.BOUNDS], [b[1] for b in self.BOUNDS])
        return new_pos

    def plan(self):
        print("Planning...")
        goal_key_pos = self.goal
        

        for i in range(self.max_iter):
            print(f"Iteration {i}")
            random_target_pos = self.random_configuration(goal_key_pos)
            random_qpos, success = self.inverse_kinematics(random_target_pos, self.fingertip)
            if not success or self.collision_check(random_qpos):
                continue

            near_idx = self.nearest_node_idx(random_qpos)
            near_pos, near_qpos = self.tree[near_idx]

            new_pos, new_qpos = self.steer(near_qpos, near_pos, random_target_pos, discretisationsteps=20, delta_q=0.1)
            if self.collision_check(new_qpos):
                continue

            self.env.physics.data.qpos[self.QPOS_INDS] = new_qpos
            mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
            self.add_node(new_pos, new_qpos, near_idx)

            if distance_finger_to_key(self.env, self.fingertip_id, self.key_geom_id) < 0.1:
                print("Goal reached.")
                return self.reconstruct_path()

        print("Goal not reached.")
        return None

    def reconstruct_path(self):
        path = []
        current_index = len(self.tree) - 1
        while current_index is not None:
            _, qpos = self.tree[current_index]
            qpos_extended = np.zeros(23)
            qpos_extended[:22] = qpos
            path.append(qpos_extended)
            current_index = self.parents[current_index]
        return path[::-1]
