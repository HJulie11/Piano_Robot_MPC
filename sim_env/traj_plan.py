import mink
import util
from loop_rate_limiters import RateLimiter
from typing import List, Tuple, Dict
from dataclasses import dataclass
import numpy as np
import mujoco
import pretty_midi
from dm_control import composer, mjcf
from util import construct_model
from scipy.interpolate import CubicSpline

@dataclass
class Note:
    pitch: int # MIDI pitch value
    velocity: int # MIDI velocity value
    start_time: float # Start time in seconds
    end_time: float # End time in seconds
    key_id: int # Index of the piano key
    sub_sequences: List[Tuple[str, float]] = None # (maneuver, duration)
    finger: str = None # Assigned finger

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @classmethod
    def from_pretty_midi(cls, pm_note: pretty_midi.Note, control_timestep: float) -> 'Note':
        key_id = pm_note.pitch - 21
        if not (0 <= key_id < 88):
            raise ValueError(f"Invalid key_id {key_id} for pitch {pm_note.pitch}")
        # Define sub-sequences with durations scaled by control_timestep
        base_duration = 0.1
        remaining_duration = max(0.1, (pm_note.end_time - pm_note.start_time) - 3 * base_duration)
        sub_sequences = [
            ("hover", base_duration),
            ("prepare", base_duration),
            ("press", base_duration),
            ("release", base_duration),
            ("transition", remaining_duration),
        ]
        return cls(
            pitch=pm_note.pitch,
            velocity=pm_note.velocity,
            start_time=pm_note.start_time,
            end_time=pm_note.end_time,
            key_id=key_id,
            sub_sequences=sub_sequences,
        )

class OfflineTrajectoryPlanner:
    def __init__(self, env, control_timstep): # _CONTROL_TIMESTEP.value

        
        self.env = env
        self.physics = self.env._physics.copy()
        self.physics = self.env.task.root_entity.physics = mjcf.Physics.from_mjcf_model(construct_model(self.env))
        self.model = self.physics.model._model
        self.data = self.physics.data._data
        self.configuration = mink.Configuration(self.model)
        self.fingertips = [
            "rh_shadow_hand/thdistal_site", 
            "rh_shadow_hand/ffdistal_site", 
            "rh_shadow_hand/mfdistal_site", 
            # "rh_shadow_hand/rfdistal_site", 
            # "rh_shadow_hand/lfdistal_site"
        ]
        self.fingertip_mocap_names = util.fingertip_mocap(self.fingertips)
        self.fingertip_mids = [self.model.body(fingertip_mocap).mocapid[0] for fingertip_mocap in self.fingertip_mocap_names]
        # self.base_mid = self.model.body("base_target").mocapid[0]
        self.dt = control_timstep
        self.piano = env.task.piano
        self.key_sites = {i: key.site[0].name for i, key in enumerate(self.piano.keys)}

        self.fingertip_tasks = [
            mink.FrameTask(
                frame_name = fingertip,
                frame_type = "site",
                position_cost = 1.0,
                orientation_cost = 0.0,
                gain = 1.0,
                lm_damping = 0.0,
            ) for fingertip in self.fingertips
        ]

        self.tasks=[*self.fingertip_tasks]

        self.solver = "quadprog"
        self.damping = 1e-3
        self.rate = RateLimiter(frequency=500.0, warn=False)

        joints = self.env.task._hand.actuators[:-2] 
        hand_positions = self.env.task._hand.actuators[-2:]

        JOINT_INDS = [] # len = 20
        for joint in joints:
            if joint.joint:   
                JOINT_INDS.append(self.model.joint("rh_shadow_hand/" + joint.joint.name).id)
            elif joint.tendon:
                JOINT_INDS.append(self.model.tendon("rh_shadow_hand/" + joint.tendon.name).id)

            else:
                raise ValueError(f"Joint or tendon not found for {joint.name}")
            
        HAND_INDS = [] # len = 2
        for hand in hand_positions:
            if hand.joint:
                HAND_INDS.append(self.model.joint("rh_shadow_hand/" + hand.joint.name).id)
            elif hand.tendon:
                HAND_INDS.append(self.model.tendon("rh_shadow_hand/" + hand.tendon.name).id)
            else:
                raise ValueError(f"Joint or tendon not found for {hand.name}")
            
        self.QPOS_INDS = JOINT_INDS + HAND_INDS

        # RRT params
        self.max_iterations = 5000
        self.step_size = 0.02
        self.goal_bias = 0.1
        self.joint_limits = np.array([self.model.jnt_range[i] for i in self.QPOS_INDS])
        self.q_neutral = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2
        self.n_joints = len(self.QPOS_INDS)
        self.full_n_joints = self.model.nq

        self.vel_limits = np.maximum(np.abs(self.joint_limits[:, 1] - self.joint_limits[:, 0]) / 2, 0.1)
        self.acc_limits = self.vel_limits * 2

    def interpret_music_sheet(self) -> List[Note]:
        midi = self.env.task.midi
        # if midi is None:
        #     # sample notes do re mi 
        notes = []
        print(midi.seq.notes)
        # help(midi)
        for pm_note in midi.seq.notes:
            try:
                note = Note.from_pretty_midi(pm_note, self.dt)
                notes.append(note)
            except ValueError as e:
                print(f"Skipping invalid note: {e}")
        return sorted(notes, key=lambda x: x.start_time)
    
    def compute_collision_risk(self, finger1: str, finger2: str, key1: int, key2: int) -> float:
        """Estimate collision risk between two fingers based on key positions."""
        key_site1 = f"piano/{self.key_sites[key1]}"
        key_site2 = f"piano/{self.key_sites[key2]}"
        pos1 = self.configuration.get_transform_frame_to_world(key_site1, "site").translation()
        pos2 = self.configuration.get_transform_frame_to_world(key_site2, "site").translation()
        dist = np.linalg.norm(pos1 - pos2)
        return 1.0 / (dist + 1e-3)
    
    def compute_joint_effort(self, q1: np.ndarray, q2: np.ndarray) -> float:
        return np.sum((q2 - q1) ** 2)
    
    def compute_dexterity_cost(self, finger: str, q: np.ndarray, key_id: int, prev_finger: str = None) -> float:
        """Compute dexterity cost based on joint comfort and fingercrossing preference."""
        joint_cost = np.sum(np.abs(q[self.QPOS_INDS] - self.q_neutral))    

        # Finger crossing preference
        crossing_cost = 0.0
        if finger == "rh_shadow_hand/thdistal_site":
            key_site = f"piano/{self.key_sites[key_id]}"
            key_pos = self.configuration.get_transform_frame_to_world(key_site, "site").translation()[0]
            key_name = f"piano/{self.env.task.piano.keys[39].site[0].name}"
            middle_c_pos = self.configuration.get_transform_frame_to_world(key_name, "site").translation()[0]
            if key_pos > middle_c_pos:
                crossing_cost = -0.05
            else:
                crossing_cost = 0.1
        
        repeat_cost = 0.5 if prev_finger == finger else 0.0
        return joint_cost + crossing_cost + repeat_cost
    
    def finger_key_correspondance(self, notes: List[Note]) -> Dict[int, str]:
        """Optimise finger assignments for dexterity, collision risk, and joint effort. """
        n = len(notes)
        fingertips = self.fingertips
        dp = np.full((n, len(fingertips)), float('inf'))
        prev = np.zeros((n, len(fingertips)), dtype=int)

        # initialise for the first note
        for f_idx, finger in enumerate(fingertips):
            q = self.solve_ik_to_pos({finger.split("/")[-1]: self.configuration.get_transform_frame_to_world(
                f"piano/{self.key_sites[notes[0].key_id]}", "site").translation()})
            dp[0][f_idx] = self.compute_dexterity_cost(finger, q, notes[0].key_id)

        for i in range(1, n):
            key_i = notes[i].key_id
            for f_idx, finger in enumerate(fingertips):
                q_i = self.solve_ik_to_pos({finger.split("/")[-1]: self.configuration.get_transform_frame_to_world(
                    f"piano/{self.key_sites[key_i]}", "site").translation()})
                prev_f_idx = int(np.argmin(dp[i-1]))
                prev_finger = fingertips[prev_f_idx]
                dexterity_cost = self.compute_dexterity_cost(finger, q_i, key_i, prev_finger)
                for f_prev_idx, prev_finger in enumerate(fingertips):
                    key_prev = notes[i-1].key_id
                    q_prev = self.solve_ik_to_pos({prev_finger.split("/")[-1]: self.configuration.get_transform_frame_to_world(
                        f"piano/{self.key_sites[key_prev]}", "site").translation()})
                    collision_cost = self.compute_collision_risk(finger, prev_finger, key_i, key_prev)
                    effort_cost = self.compute_joint_effort(q_prev[self.QPOS_INDS], q_i[self.QPOS_INDS])
                    total_cost = dp[i-1][f_prev_idx] + 0.4 * collision_cost + 0.3 * effort_cost + 0.3 * dexterity_cost
                    if total_cost < dp[i][f_idx]:
                        dp[i][f_idx] = total_cost
                        prev[i][f_idx] = f_prev_idx

        # Backtrack to find the optimal finger assignment
        finger_assignments = []
        f_idx = np.argmin(dp[-1])
        for i in range(n-1, -1, -1):
            finger_assignments.append(fingertips[f_idx])
            f_idx = int(prev[i][f_idx])
        finger_assignments.reverse()

        # Assign fingers to notes and create correspondence
        correspondence = {}
        for note,finger in zip(notes, finger_assignments):
            note.finger = finger
            correspondence[note.key_id] = finger
        print(f"Correspondance: {correspondence}")
        return correspondence
    
    def solve_ik_to_pos(self, target_positions: Dict[str, np.ndarray], max_steps: int = 200, forearm_pos: np.ndarray = None, fix_finger_joints: bool = False) -> np.ndarray:
        # Reset mocap positions to current fingertip positions
        for idx, fingertip in enumerate(self.fingertips):
            mink.move_mocap_to_frame(self.model, self.data, self.fingertip_mocap_names[idx], fingertip, "site")
        
        # Set fingertip targets
        for i, fingertip in enumerate(self.fingertips):
            fingertip_name = fingertip.split("/")[-1]
            if fingertip_name in target_positions:
                self.data.mocap_pos[self.fingertip_mids[i]] = target_positions[fingertip_name]
            self.fingertip_tasks[i].set_target(mink.SE3.from_mocap_id(self.data, self.fingertip_mids[i]))

        # Solve IK iteratively
        q_initial = self.configuration.q.copy()
        q_initial[self.QPOS_INDS[:-2]] = self.q_neutral[:-2]
        if forearm_pos is not None:
            q_initial[self.QPOS_INDS[-2:]] = forearm_pos

        for step in range(max_steps):
            vel = mink.solve_ik(self.configuration, self.tasks, self.rate.dt, self.solver, damping=self.damping)

            if forearm_pos is not None:
                vel[self.QPOS_INDS[-2:]] = 0.0
            if fix_finger_joints:
                vel[self.QPOS_INDS[:-2]] = 0.0
    
            self.configuration.integrate_inplace(vel, self.rate.dt)
            mujoco.mj_forward(self.model, self.data)

            fingertip_name = next(iter(target_positions))
            fingertip = [f for f in self.fingertips if f.endswith(fingertip_name)][0]
            key_id = None
            for k, v in self.key_sites.items():
                key_site_name = f"piano/{v}"
                key_pos = self.configuration.get_transform_frame_to_world(key_site_name, "site").translation()
                if np.linalg.norm(key_pos - target_positions[fingertip_name]) < 1e-3:
                    key_id = k
                    break
            
            if key_id is not None and not self.is_collision_free(self.configuration.q[self.QPOS_INDS], fingertip, key_id, strict=False): # strict=True
                vel += np.random.uniform(-0.05, 0.05, size=vel.shape)
                if forearm_pos is not None:
                    vel[self.QPOS_INDS[-2:]] = 0.0
                if fix_finger_joints:
                    vel[self.QPOS_INDS[:-2]] = 0.0
                self.configuration.integrate_inplace(vel, self.rate.dt)
                mujoco.mj_forward(self.model, self.data)

            # Check if converged
            errors = []
            for i, fingertip in enumerate(self.fingertips):
                fingertip_name = fingertip.split("/")[-1]
                if fingertip_name in target_positions:
                    current_pos = self.data.site_xpos[self.model.site(fingertip).id]
                    target_pos = target_positions[fingertip_name]
                    error = np.linalg.norm(current_pos - target_pos)
                    errors.append(error)

            if all(error < 1e-3 for error in errors):
                break
        
        if forearm_pos is not None:
            self.configuration.q[self.QPOS_INDS[-2:]] = forearm_pos
        return self.configuration.q.copy()
    
    def map_to_full_qpos(self, q: np.ndarray) -> np.ndarray:
        """Map a controlled joint configuration to the full joint configuration."""
        full_qpos = self.data.qpos.copy()
        full_qpos[self.QPOS_INDS] = q
        return full_qpos

    def is_collision_free(self, q: np.ndarray, finger: str = None, key_id: int = None, strict: bool = True, maneuver: str = None) -> bool:
        """check ifa joint configuration is collision-free."""
        full_qpos = self.map_to_full_qpos(q)
        self.data.qpos = full_qpos
        mujoco.mj_step(self.model, self.data)
        if self.data.ncon == 0:
            return True
        
        if finger is not None and key_id is not None:
            fingertip_site_id = self.model.site(finger).id
            print(finger, fingertip_site_id)
            key_site_name = f"piano/{self.key_sites[key_id]}"
            print(key_site_name)
            key_site_id = self.model.site(key_site_name).id

            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2

                # Get the body IDs of the geoms
                body1_id = self.model.geom_bodyid[geom1]
                body2_id = self.model.geom_bodyid[geom2]

                site1_ids = [site_id for site_id in range(self.model.nsite) if self.model.site_bodyid[site_id] == body1_id]
                site2_ids = [site_id for site_id in range(self.model.nsite) if self.model.site_bodyid[site_id] == body2_id]

                # Check if this contact involves the fingertip and the target key
                involves_fingertip = fingertip_site_id in site1_ids or fingertip_site_id in site2_ids
                involves_key = key_site_id in site1_ids or key_site_id in site2_ids

                if involves_fingertip and involves_key:
                    continue 

                geom1_name = self.model.geom(geom1).name
                geom2_name = self.model.geom(geom2).name

                if maneuver in ["hover", "transition"] and "rh_shadow_hand" in geom1_name and "rh_shadow_hand" in geom2_name:
                    continue

                if not strict and "rh_shadow_hand" in geom1_name and "rh_shadow_hand" in geom2_name:
                    continue

                # if key_site_name in geom1_name or key_site_name in geom2_name:
                #     continue
                
                print(f"Collision between {self.model.geom(geom1).name} and {self.model.geom(geom2).name}")
                return False
            
            return True

        print(f"Collision detected with {self.data.ncon} contacts")
        return False
    
    def sample_configuration(self, tree: list = None) -> np.ndarray:
        """Sample a random configuration within joint limits."""
        # q = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
        # return q
        if tree and np.random.rand() < 0.5:
            q_near, _ = self.nearest_node(tree, self.q_neutral)
            q = q_near + np.random.uniform(-0.1, 0.1, size=self.n_joints)
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        else:
            q = np.random.uniform(self.joint_limits[:, 0], self.joint_limits[:, 1])
        return q
    
    def nearest_node(self, tree: list, q_rand: np.ndarray) -> tuple:
        """Find the nearest node in the tree to q_rand."""
        distances = [np.linalg.norm(q - q_rand) for q, _ in tree]
        idx = np.argmin(distances)
        return tree[idx]

    def extend(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        """Extend from q_near toward q_rand by step_size."""
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_rand
        return q_near + (direction / dist) * self.step_size
    
    def dexterity_bias(self, q: np.ndarray, finger:str, key_id: int) -> np.ndarray:
        """Bias the configuration for dexterous movement (e.g., thumb under)."""
        if finger == "rh_shadow_hand/thdistal_site":
            key_site = f"piano/{self.key_sites[key_id]}"
            key_pos = self.configuration.get_transform_frame_to_world(key_site, "site").translation()[0]
            key_site_39 = f"piano/{self.key_sites[39]}"
            middle_c_pos = self.configuration.get_transform_frame_to_world(key_site_39, "site").translation()[0]
            if key_pos > middle_c_pos:
                self.data.qpos = self.map_to_full_qpos(q)
                mujoco.mj_forward(self.model, self.data)
                fingertip_pos = self.data.site_xpos[self.model.site(finger).id]
                if fingertip_pos[2] > self.configuration.get_transform_frame_to_world(key_site, "site").translation()[2] - 0.02:
                    taret_pos = fingertip_pos.copy()
                    taret_pos[2] -= 0.01
                    q_adjusted = self.solve_ik_to_pos({finger.split("/")[-1]: taret_pos})
                    if self.is_collision_free(q_adjusted[self.QPOS_INDS], finger, key_id, strict=False):
                        return q_adjusted[self.QPOS_INDS]
                    # return q_adjusted[self.QPOS_INDS]
        return q
    
    def rrt_connect_plan(self, q_start: np.ndarray, q_goal: np.ndarray, finger: str, key_id: int) -> list:
        q_start = q_start[self.QPOS_INDS]
        q_goal = q_goal[self.QPOS_INDS]
        
        # Add intermediate waypoints
        waypoints = [q_start]
        dist = np.linalg.norm(q_start - q_goal)
        if dist > 0.5: 
            num_waypoints = int(dist / 0.5)
            for i in range(1, num_waypoints):
                alpha = i / num_waypoints
                wp = (1 - alpha) * q_start + alpha * q_goal
                waypoints.append(wp)
        waypoints.append(q_goal)

        full_path = []
        for i in range(len(waypoints) - 1):
            wp_start = waypoints[i]
            wp_goal = waypoints[i+1]

            # RRT-Connect: Grow two trees
            tree_a = [(wp_start, None)]
            tree_b = [(wp_goal, None)]
            for iter in range(self.max_iterations):
                if np.random.rand() < self.goal_bias:
                    q_rand = wp_goal if len(tree_a) <= len(tree_b) else wp_start
                else:
                    q_rand = self.sample_configuration(tree_a if len(tree_a) <= len(tree_b) else tree_b)
                
                # Extend tree_a
                q_rand = self.dexterity_bias(q_rand, finger, key_id)
                q_near_a, parent_a = self.nearest_node(tree_a, q_rand)
                q_new_a = self.extend(q_near_a, q_rand)
                if not self.is_collision_free(q_new_a, finger, key_id, strict=False):
                    print(f"rrt check False (tree_a)")
                    continue
                tree_a.append((q_new_a, len(tree_a) - 1))

                # Try to connect to tree_b
                q_near_b, parent_b = self.nearest_node(tree_b, q_new_a)
                q_new_b = self.extend(q_near_b, q_new_a)
                if not self.is_collision_free(q_new_b, finger, key_id, strict=False):
                    print(f"rrt check False (tree_b)")
                    continue
                tree_b.append((q_new_b, len(tree_b) - 1))

                dist = np.linalg.norm(q_new_a - q_new_b)
                print(f"rrt check {dist}")
                if dist < self.step_size:
                    # Construct path from tree_a
                    path_a = []
                    current = len(tree_a) - 1
                    while current is not None:
                        path_a.append(tree_a[current][0])
                        current = tree_a[current][1]
                    path_a.reverse()

                    # Construct path from tree_b
                    path_b = []
                    current = len(tree_b) - 1
                    while current is not None:
                        path_b.append(tree_b[current][0])
                        current = tree_b[current][1]

                    # Combine paths
                    path = path_a + path_b
                    full_path.extend(path[:-1] if i < len(waypoints) - 2 else path)
                    break

                # swap trees to alternate growth
                tree_a, tree_b = tree_b, tree_a
            else:
                # raise ValueError("RRT-Connect failed to find a path")
                print("RRT-Connect failed to find a path")
        print("RRT-Connect succeeded")
        return full_path
    
    def smooth_path(self, path: list, finger: str, key_id: int) -> list:
        """Smooth the rrt path using shortcutting."""
        smoothed_path = path.copy()
        for _ in range(30):
            if len(smoothed_path) < 3:
                break
            i = np.random.randint(0, len(smoothed_path) - 2)
            j = np.random.randint(i + 2, len(smoothed_path))
            q_i, q_j = smoothed_path[i], smoothed_path[j]
            t = np.linspace(0, 1, 20)
            collision_free = True
            for alpha in t[1:-1]:
                q = (1-alpha) * q_i + alpha * q_j
                if not self.is_collision_free(q, finger, key_id, strict=True): # strict=True
                    collision_free = False
                    break
            if collision_free:
                smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
        return smoothed_path
    
    def parameterize_path(self, path: list, start_time: float, end_time: float, num_steps: int) -> np.ndarray:
        """Parameterize the path with time and ensure dynamic feasibility."""
        n_points = len(path)
        if n_points < 2:
            print(f"Warning: Path has only {n_points} point(s). Duplicating to create a minimal path.")
            path = [path[0], path[0]]
            n_points = 2
        t = np.linspace(start_time, end_time, n_points)
        path_array = np.array(path)

        t_new = np.linspace(start_time, end_time, num_steps)
        trajectory = np.zeros((num_steps, self.n_joints))
        for j in range(self.n_joints):
            spline = CubicSpline(t, path_array[:, j], bc_type = "clamped")
            q = spline(t_new)
            qdot = spline.derivative()(t_new)
            qddot = spline.derivative().derivative()(t_new)

            vel_limit = self.vel_limits[j]
            acc_limit = self.acc_limits[j]

            if np.any(np.abs(qdot) > vel_limit) or np.any(np.abs(qddot) > acc_limit):
                scale = 0.9 * min(
                    vel_limit / (np.max(np.abs(qdot)) + 1e-6),
                    acc_limit / (np.max(np.abs(qddot)) + 1e-6)
                )
                t_new = np.linspace(start_time, end_time / scale, num_steps)
                q = spline(t_new)
            
            q = np.clip(q, self.joint_limits[j, 0], self.joint_limits[j, 1])
            trajectory[:, j] = q
        
        return trajectory

    def generate_key_poses(self, notes: List[Note], correspondence: Dict[int, str]) -> Dict[int, List[np.ndarray]]:
        poses = {}
        for note in notes:
            print(note.key_id)
            print(correspondence)
            finger = correspondence[note.key_id]
            finger_name = finger.split('/')[-1]
            key_site_name = f"piano/{self.key_sites[note.key_id]}"
            key_geom_name = f"piano/{self.env.task.piano.keys[note.key_id].geom[0].name}"
            # key_pos = self.data.site_xpos[self.model.site(key_site_name).id]
            transform_fingertip_target_to_world = self.configuration.get_transform_frame_to_world(key_site_name, "site") 
            # transform_fingertip_target_to_world = self.configuration.get_transform_frame_to_world(key_geom_name, "geom") 
            translation = transform_fingertip_target_to_world.translation()
            # self.fingertip_tasks[self.fingertips.index(finger)].set_target(transform_fingertip_target_to_world)
            # mink.move_mocap_to_frame(self.model, self.data, self.fingertip_mocap_names[self.fingertips.index(finger)], finger, "site")
            key_pos = translation.copy()
            key_pos[0] += 0.05

            # forearm_pos = key_pos[:2]

            # if note.key_id == 39:
            #     forearm_pos[0] -= 0.01

            pose_sequence = []
            print(note.sub_sequences)
            for maneuver, duration in note.sub_sequences:

                target_pos = key_pos.copy()
                target_positions = {finger_name: target_pos}
                qpos = self.solve_ik_to_pos(target_positions)
                qpos_zero = np.zeros(114)

                if maneuver in ["hover", "transition"]:
                    # forearm_adjustment = qpos[self.QPOS_INDS[-2:]]
                    # qpos[self.QPOS_INDS[-2:]] += 0.5 * forearm_adjustment

                    # qpos[self.QPOS_INDS[-2:]] = np.clip(qpos[self.QPOS_INDS[-2:]], self.joint_limits[-2:, 0], self.joint_limits[-2:, 1])

                    qpos_zero[self.QPOS_INDS[-2:]] = qpos[self.QPOS_INDS[-2:]]
                    qpos = qpos_zero
                elif maneuver in ["prepare", "press", "release"]:
                    adjusted_target_pos = key_pos.copy()
                    adjusted_target_pos[0] += 0.05
                    if maneuver == "prepare":
                        adjusted_target_pos[2] = key_pos[2]
                    elif maneuver == "press":
                        adjusted_target_pos[2] = key_pos[2] - 0.05
                    elif maneuver == "release":
                        adjusted_target_pos[2] = key_pos[2] + 0.1
            
                    print(f"Target position for {maneuver}: {target_pos}")
                    forearm_pos = qpos[self.QPOS_INDS[-2:]]
                    target_positions = {finger_name: adjusted_target_pos}
                    qpos = self.solve_ik_to_pos(target_positions, forearm_pos=forearm_pos, fix_finger_joints=False)
                    
                print(f" Joint configuration (qpos): {qpos[self.QPOS_INDS]}")
                strict = False if maneuver in ["hover", "transition"] else True
                if not self.is_collision_free(qpos[self.QPOS_INDS], finger, note.key_id, strict=strict, maneuver=maneuver): # strict=True
                    # raise ValueError(f"Collision detected in {maneuver} pose for key {note.key_id}")
                    print(f"Collision detected in {maneuver} pose for key {note.key_id}")
                pose_sequence.append(qpos)
            
            poses[note.key_id] = pose_sequence
        
        return poses
    
    def interpolate_trajectory(self, poses: Dict[int, List[np.ndarray]], notes: List[Note]) -> List[np.ndarray]:
        trajectory = []
        current_time = 0.0

        for note_idx, note in enumerate(notes):
            finger = note.finger
            key_id = note.key_id
            pose_sequence = poses[note.key_id]
            times = np.cumsum([seq[1] for seq in note.sub_sequences])
            times = np.insert(times, 0, 0.0)
            times = times/self.dt  # Noremalize to timesteps

            # use RRT to plan paths between sub-sequences
            path = []

            # Hover to prepare
            q_start = pose_sequence[0]
            q_goal = pose_sequence[1]
            num_steps = int(times[1] - times[0])
            # sub_path = self.rrt_plan(q_start, q_goal, finger, key_id)
            sub_path = self.rrt_connect_plan(q_start, q_goal, finger, key_id)
            sub_path = self.smooth_path(sub_path, finger, key_id)
            sub_traj = self.parameterize_path(sub_path, times[0] * self.dt, times[1] * self.dt, num_steps)
            path.extend(sub_path[:-1])

            # prepare to press
            q_start = pose_sequence[1]
            q_goal = pose_sequence[2]
            num_steps = int(times[2] - times[1])
            # sub_path = self.rrt_plan(q_start, q_goal, finger, key_id)
            sub_path = self.rrt_connect_plan(q_start, q_goal, finger, key_id)
            sub_path = self.smooth_path(sub_path, finger, key_id)
            sub_traj = self.parameterize_path(sub_path, times[1] * self.dt, times[2] * self.dt, num_steps)
            path.extend(sub_path[:-1])

            # press to release
            q_start = pose_sequence[2]
            q_goal = pose_sequence[3]
            num_steps = int(times[3] - times[2])
            # sub_path = self.rrt_plan(q_start, q_goal, finger, key_id)
            sub_path = self.rrt_connect_plan(q_start, q_goal, finger, key_id)
            sub_path = self.smooth_path(sub_path, finger, key_id)
            sub_traj = self.parameterize_path(sub_path, times[2] * self.dt, times[3] * self.dt, num_steps)
            path.extend(sub_path[:-1])

            # release to transition
            q_start = pose_sequence[3]
            q_goal = pose_sequence[4]
            num_steps = int(times[4] - times[3])
            # sub_path = self.rrt_plan(q_start, q_goal, finger, key_id)
            sub_path = self.rrt_connect_plan(q_start, q_goal, finger, key_id)
            sub_path = self.smooth_path(sub_path, finger, key_id)
            sub_traj = self.parameterize_path(sub_path, times[3] * self.dt, times[4] * self.dt, num_steps)
            path.extend(sub_path)

            # Combine sub_trajectories
            # sub_trajectory = np.vstack([self.parameterize_path(sub_path, times[i] * self.dt, times[i+1] * self.dt, int(times[i+1] - times[i])) for i in range(len(pose_sequence) - 1)])
            sub_trajectory = np.vstack([self.parameterize_path(path[i:i+1], times[i] * self.dt, times[i+1] * self.dt, int(times[i+1] - times[i])) for i in range(len(pose_sequence) - 1)])
            trajectory.extend([self.map_to_full_qpos(q) for q in sub_trajectory])

            # Hold the press position
            num_hold_steps = int((note.duration - sum(seq[1] for seq in note.sub_sequences[:-1])) / self.dt)
            if num_hold_steps > 0:
                trajectory.extend([pose_sequence[2]] * num_hold_steps)
                current_time += num_hold_steps * self.dt

            next_note_time = notes[notes.index(note) + 1].start_time if notes.index(note) + 1 < len(notes) else note.end_time
            padding_steps = int((next_note_time - (note.start_time + note.duration)) / self.dt)
            if padding_steps > 0:
                # trajectory.extend([pose_sequence[-1]] * padding_steps)
                # current_time += padding_steps * self.dt
                if note_idx + 1 < len(notes):
                    next_note = notes[note_idx + 1]
                    next_pose_sequence = poses[next_note.key_id]
                    q_start = pose_sequence[-1]
                    q_goal = next_pose_sequence[0]
                    sub_path = self.rrt_connect_plan(q_start, q_goal, finger, key_id)
                    sub_path = self.smooth_path(sub_path, finger, next_note.key_id)
                    sub_traj = self.parameterize_path(sub_path, current_time, current_time + padding_steps * self.dt, padding_steps)
                    trajectory.extend([self.map_to_full_qpos(q) for q in sub_traj])
                else:
                    trajectory.extend([pose_sequence[-1]] * padding_steps)
                current_time += padding_steps * self.dt
            
        # Valiate trajectory for collisions
        for i, qpos in enumerate(trajectory):
            if not self.is_collision_free(qpos[self.QPOS_INDS], finger, key_id, strict=True): # strict=True
                print(f"Collision at trajectory step {i}")
                # raise ValueError("Trajectory is not collision-free")
                print("Trajectory is not collision-free")
        return trajectory
    
    def plan(self) -> List[np.ndarray]:
        # midi = self.interpret_music_sheet()
        notes = self.interpret_music_sheet()
        # notes = notes[:5]
        correspondence = self.finger_key_correspondance(notes)
        poses = self.generate_key_poses(notes, correspondence)
        trajectory = self.interpolate_trajectory(poses, notes)
        return trajectory

    def plan_qpos_hand(self, trajectory) -> List[np.ndarray]:
        trajectory_hand = []
        for qpos in trajectory:
            qpos_hand = np.zeros(23)
            qpos_hand[:-1] = qpos[self.QPOS_INDS]
            trajectory_hand.append(qpos_hand)
        return trajectory_hand