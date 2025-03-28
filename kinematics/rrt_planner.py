import mujoco
import mink # ik solver
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize

from .get_config import compute_ik, compute_fk
from sim_env.util import solve_ik
from action.action import HandController
from loop_rate_limiters import RateLimiter

import sys
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

import robopianist
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand


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
    def __init__(self, env, goal_key_number, site_name, max_iter=50, step_size=0.2):
        self.env = env

        self.fingertip = site_name # fingertip site names
        self.fingertip_id = self.env.physics.model.site(site_name).id
        
        self.key_name = env.task.piano._sites[goal_key_number].name
        self.key_site_name = "piano/" + self.env.task.piano.keys[goal_key_number].site[0].name
        self.key_geom_name = env.task.piano.keys[goal_key_number].geom[0].name

        self.key_geom_id = env.physics.model.geom("piano/" + self.key_geom_name).id
        self.key_id = self.env.physics.model.site("piano/" + self.key_name).id
        self.goal = self.env.physics.data.site_xpos[self.key_id].copy() # world frame
        self.goal_key_number = goal_key_number

        self.max_iter = max_iter
        self.step_size = step_size

        self.pre_move_done = False # Flag to track horizontal movement completion

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
        self.QPOS_NAMES = self.JOINT_NAMES + self.HAND_POSITIONS # len = 22 (joints(config) and hand pos(x, y coordinate)) names
        self.QPOS_INDS = self.JOINT_INDS + self.HAND_INDS # len = 22 (joints(config) and hand pos(x, y coordinate)) indices

        self.hand_controller = HandController(env, self.fingertip, self.QPOS_INDS)
        self.FINGERTIP_INDS = [env.physics.model.site("rh_shadow_hand/" + site.name).id for site in self.env.task._hand.fingertip_sites]
        self.PIANO_KEY_INDS = [env.physics.model.site("piano/" + key.name).id for key in self.env.task.piano._sites]
        
        initial_qpos = np.array(self.env.physics.data.qpos[self.QPOS_INDS].copy())
        # initial_fingertip_pos = np.array(self.env.physics.model.site(self.fingertips).pos.copy())
        initial_fingertip_pos = np.array(self.env.physics.data.site_xpos[self.fingertip_id].copy())

        self.tree = [(initial_fingertip_pos, initial_qpos)] # List to store nodes in the tree 
        self.parents = {0: None} # Dictionary to track parent nodes
    
    def inverse_kinematics(self, target_positions, site_name):
        """Solve for joint angles using IK. Given target positions (x, y, z), or key position."""
        target_positions = np.array(target_positions) # Ensure world-frame target positions
        # ik_joint_config, success = compute_ik(
        #     self.env, 
        #     self.env.physics.model._model, 
        #     self.env.physics.data._data, 
        #     self.env.physics.data.qpos[self.QPOS_INDS],
        #     self.key_geom_id,
        #     target_positions, 
        #     self.goal_key_number,
        #     site_name, 
        #     self.QPOS_INDS
        # )
        rate = RateLimiter(frequency=200.0)
        ik_joint_config = solve_ik(self.env, site_name, self.key_site_name, rate, press=True)
        success = self.is_valid_joint(ik_joint_config[self.QPOS_INDS])

        print(f"IK success: {success}, Config: {ik_joint_config}")  

        return ik_joint_config[self.QPOS_INDS], success # return joint angles and success flag
    
    def move_fingertip_to_key(self, target_key_id):
        """Moves the fingertip to the desired key while ensuring proper contact making and breaking."""
        target_key_pos = self.env.physics.data.site_xpos[target_key_id].copy()
        self.hand_controller.move_fingertip(target_key_pos)

    
    def forward_kinematics(self, qpos):
        self.env.physics.data.qpos[self.QPOS_INDS] = qpos
        mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
        return self.env.physics.data.site_xpos[self.fingertip_id].copy()

    def collision_check(self, qpos):
        """Check if the given qpos results in a collision based on the defined conditions."""

        # Apply qpos to the environment
        self.env.physics.data.qpos[self.QPOS_INDS] = qpos
        self.env.physics.forward()

        # Get fingertip position
        fingertip_pos = self.env.physics.data.site_xpos[self.fingertip_id].copy()

        # Get key position
        key_site_pos = self.env.physics.data.site_xpos[self.key_id].copy() # should i use key_geom_pos?
        key_site_pos[-1] += 0.5 * self.env.physics.model.site_size[self.key_id, 2]
        key_site_pos[0] += 0.35 * self.env.physics.model.site_size[self.key_id, 0]

        # Get finger geom position
        finger_geom_pos = self.env.physics.data.geom_xpos[self.fingertip_id].copy()
        finger_geom_pos[-1] += 0.5 * self.env.physics.model.geom_size[self.fingertip_id, 2]
        finger_geom_pos[0] += 0.35 * self.env.physics.model.geom_size[self.fingertip_id, 0]

        # Condition 1: Check if the fingertip is close to the key
        # distance_to_goal_key = np.linalg.norm(fingertip_pos - key_site_pos)
        # if distance_to_goal_key >= 0.1:
        #     print(f"Fingertip is not close ")
        #     return True # Collision detected
        
        # Condition 2: Check if the fingertip is close to other keys
        # Condition 2 (edited): check if the finger of the given fingertip is in collision with any other key
        for other_key_id in self.PIANO_KEY_INDS:
            if other_key_id == self.key_id:
                continue # Skip the goal key

            # other_key_pos = self.env.physics.data.geom_xpos[other_key_id].copy() # use site_xpos to get the distance between the positions but here im trying to check whether the finger and key collide
            # other_key_pos[-1] += 0.5 * self.env.physics.model.geom_size[other_key_id, 2]
            # other_key_pos[0] += 0.35 * self.env.physics.model.geom_size[other_key_id, 0]
            distance = distance_finger_to_key(self.env, self.fingertip_id, other_key_id)
            print(f"Distance between fingertip and other key: {distance}")
            # if np.linalg.norm(finger_geom_pos - other_key_pos) < 0.1:
            if distance < 0.1:
                print("Fingertip is close to another key.")
                return True # Collision detected
        
        # Condition 3: Check joint limits
        # if np.any(qpos < self.JOINT_LIMITS[:, 0]) or np.any(qpos > self.JOINT_LIMITS[:, 1]):
        if self.jointlimitsviolated(qpos):
            print("Joint limits violated.")
            return True # Collision detected
        
        # Condition 4: Ensure other fingers are not pressing any keys
        for site_id in self.FINGERTIP_INDS:
            if site_id == self.fingertip_id:
                continue # Skip the fingertip site

            # finger_pos = self.env.physics.data.site_xpos[site_id].copy()
            for key_id in self.PIANO_KEY_INDS:
                # key_pos = self.env.physics.data.site_xpos[key_id].copy()
                # key_pos[-1] += 0.5 * self.env.physics.model.site_size[key_id, 2]
                # key_pos[0] += 0.35 * self.env.physics.model.site_size[key_id, 0]
                distance = distance_finger_to_key(self.env, site_id, key_id)

                # if np.linalg.norm(finger_pos - key_pos) < 0.1:
                if distance < 0.1:
                    print(f"Finger {site_id} is close to a key.")
                    return True # Collision detected
        
        return False # No collision detected

    def nearest_node_idx(self, random_pos):
        """Find the nearest node in the tree using KDTree.""" 
        tree_pos = np.array([node[1] for node in self.tree])  # Ensure uniform shape
        tree = KDTree(tree_pos)    
        _, ind = tree.query(random_pos)
        return ind
    
    def lerp(self, pos1, pos2, t):
        """Linear interpolation between two positions."""
        return (1 - t) * pos1 + t * pos2
    
    def steer(self, near_qpos, near_pos, random_target_pos, discretisationsteps, delta_q=None):
        """Move from nearest node towards random_sample by a step size."""
        end_pos = random_target_pos.copy()
        dist = np.linalg.norm(random_target_pos - near_pos)

        if delta_q is not None and dist > delta_q:
            end_pos = self.lerp(near_pos, random_target_pos, delta_q / dist)
            dist = delta_q
        
        dt = dist / discretisationsteps
        last_valid_pos = near_pos
        last_valid_qpos = near_qpos

        for i in range(1, discretisationsteps):
            lerp_pos = self.lerp(near_pos, end_pos, (dt * i) / dist)
            lerp_qpos, success = self.inverse_kinematics(lerp_pos, self.fingertip)
            
            if success:
                last_valid_pos = lerp_pos
                last_valid_qpos = lerp_qpos
            else:
                while not success:
                    lerp_pos = self.lerp(near_pos, last_valid_pos, (dt * (i-1)) / dist)
                    lerp_qpos, success = self.inverse_kinematics(lerp_pos, self.fingertip)

                    if success:
                        last_valid_pos = lerp_pos
                        last_valid_qpos = lerp_qpos
                        return last_valid_pos, last_valid_qpos
                    if np.array_equal(last_valid_pos, near_pos):
                        return near_pos, near_qpos
                break
        return last_valid_pos, last_valid_qpos
    
        
    def is_valid(self, joint_config):
        """Check for validity (basic check: within joint limits)."""
        return all(low <= q <= high for q, (low, high) in zip(joint_config, self.BOUNDS))
    
    def is_valid_joint(self, joint_config):
        """Check for validity (basic check: within joint limits)."""
        return all(low <= q <= high for q, (low, high) in zip(joint_config, self.JOINT_LIMITS))
    
    def random_configuration(self, goal_key_pos):
        if np.random.rand() < 0.3:  # 30% chance to bias toward goal
            random_pos = goal_key_pos + np.random.uniform(-0.05, 0.05, size=goal_key_pos.shape)
        else:
            random_pos = np.random.uniform(low=-1.5, high=1.5, size=goal_key_pos.shape)
        
        return random_pos

        # action_spec = self.env.action_spec()
        # q_rand = np.random.uniform(action_spec.minimum[:-1], action_spec.maximum[:-1])
        # return q_rand

    def jointlimitscost(self, qpos):
        '''Return cost of joint limits'''
        return np.sum([max(0.0,low-q) + max(0.0,q-high) for q,(low,high) in zip(qpos,self.JOINT_LIMITS)])

    def jointlimitsviolated(self, qpos):
        '''Return true if config not in joint limits'''
        print(f"Joint limits cost: {self.jointlimitscost(qpos)}")
        return self.jointlimitscost(qpos) > 0.0
    
    def projecttojointlimits(self, qpos):
        '''Project config to joint limits'''
        # return np.clip(qpos, self.JOINT_LIMITS[:,0], self.JOINT_LIMITS[:,1])
        return np.minimum(np.maximum(qpos, self.JOINT_LIMITS[:,0]), self.JOINT_LIMITS[:,1])
    
    def add_node(self, fingertip_pos, qpos, parent_idx):
        """Add an new node to the tree."""
        self.tree.append((np.array(fingertip_pos), np.array(qpos)))
        self.parents[len(self.tree) - 1] = parent_idx # Track parent node

    def move_hand_above_key(self):
        """Moves the hand horizontally to align with the key's (x, y) coordinates."""
        current_pos = self.env.physics.data.qpos[self.HAND_INDS].copy()
        target_pos = self.goal[:2]
        
        direction = (target_pos - current_pos) / np.linalg.norm(target_pos - current_pos)
        step = self.step_size * direction
        new_pos = current_pos + step

        # qpos = self.env.physics.data.qpos.copy()
        # qpos[self.HAND_INDS[0]] = new_pos[0]
        # qpos[self.HAND_INDS[1]] = new_pos[1]

        # self.env.physics.data.qpos[:] = qpos
        # self.env.physics.forward()

        return new_pos
    
    def plan(self):
        """Perform RRT motion planning to move from start to goal."""
        # TODO: implement constraints 
        #  - distance between other fingertips and the adjacent keys (keep the neutral pos)
        print("Planning...")
        goal_key_pos = self.goal

        for i in range(self.max_iter):
            print(f"Iteration {i}")

            # while True:
            # if not self.pre_move_done:
            #     print("Moving hand above key.")
            #     random_target_pos = np.zeros(3)
            #     random_target_pos[:2] = self.move_hand_above_key()
            #     pos_tuple = tuple(random_target_pos)
            #     if pos_tuple not in sampled_pos:
            #         sampled_pos.add(pos_tuple)
            #     self.pre_move_done = True
            # else:
            #     print("Hand is already above the key")
            #     pos_tuple = tuple(random_target_pos)
            #     if pos_tuple not in sampled_pos:
            #         sampled_pos.add(pos_tuple)
            random_target_pos = self.random_configuration(goal_key_pos)
            random_qpos, _ = self.inverse_kinematics(random_target_pos, self.fingertip)

            # if self.jointlimitsviolated(random_qpos):
            #     self.projecttojointlimits(random_qpos)

            # 1. Sample a random target position in Cartesian space
            # now check if the random_pos can get the qpos that is within the joint limits

            near_pos_ind = self.nearest_node_idx(random_qpos) # random_target_pos
            near_pos, near_qpos = self.tree[near_pos_ind]
            near_qpos, success = self.inverse_kinematics(near_pos, self.fingertip)
            print("collision", success)

            new_pos, new_qpos = self.steer(near_qpos, near_pos, random_target_pos, discretisationsteps=20)
            if self.jointlimitsviolated(new_qpos):
                print("Joint limits violated for the new position. So projecting to joint limits.")
                new_qpos = self.projecttojointlimits(new_qpos)


            # 5. Apply the new configuration in the environment
            self.env.physics.data.qpos[self.QPOS_INDS] = new_qpos
            mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)

            # 6. store new node in the tree
            print(f"New position: {new_qpos}")
            # if not self.collision_check(new_qpos):
            self.add_node(new_pos, new_qpos, near_pos_ind)
        
            print(f"Distance between fingertip and key: {distance_finger_to_key(self.env, self.fingertip_id, self.key_geom_id)}")
            if distance_finger_to_key(self.env, self.fingertip_id, self.key_geom_id) < 0.1:
                print("Goal reached.")
                path = self.reconstruct_path()
                return path
            
        print("Goal not reached.")
        return None
    
    def reconstruct_path(self):
        """Reconstruct the path from goal to start."""
        path = []
        current_index = len(self.tree) - 1

        while current_index is not None:
            qpos_extended = np.zeros(23)
            _, qpos = self.tree[current_index]
            qpos_extended[:22] = qpos # Converts (22, ) to (23, )

            path.append(qpos_extended)
            current_index = self.parents[current_index]

        return path[::-1] # Reverse the path