import numpy as np
import mujoco
from scipy.linalg import pinv
from scipy.spatial import KDTree
from . import rrt_planner_2 as rrt

class InverseKinematicsSolver:
    def __init__(self, env, coupled_joints, const_rrt):
        self.env = env
        self.model = env.physics.model._model
        self.damping = 1e-1

        self.finger_joints = {
            "thumb": ['rh_A_THJ5', 'rh_A_THJ4', 'rh_A_THJ3', 'rh_A_THJ2', 'rh_A_THJ1'],
            "index": ['rh_A_FFJ4', 'rh_A_FFJ3', 'rh_A_FFJ0'],
            "middle": ['rh_A_MFJ4', 'rh_A_MFJ3', 'rh_A_MFJ0'],
            "ring": ['rh_A_RFJ4', 'rh_A_RFJ3', 'rh_A_RFJ0'],
            "little": ['rh_A_LFJ5', 'rh_A_LFJ4', 'rh_A_LFJ3', 'rh_A_LFJ0']
        }

        self.joint_limits = {
            'rh_A_WRJ2': (-0.698, 0.489),
            'rh_A_WRJ1': (-0.489, 0.140),
            'rh_A_THJ5': (-1.047, 1.047),
            'rh_A_THJ4': (0.0, 1.222),
            'rh_A_THJ3': (-0.209, 0.209),
            'rh_A_THJ2': (-0.698, 0.698),
            'rh_A_THJ1': (-0.262, 1.571),
            'rh_A_FFJ4': (-0.349, 0.349),
            'rh_A_FFJ3': (-0.262, 1.57),
            'rh_A_FFJ0': (0.0, 1.57),
            'rh_A_MFJ4': (-0.349, 0.349),
            'rh_A_MFJ3': (-0.262, 1.57),
            'rh_A_MFJ0': (0.0, 1.57),
            'rh_A_RFJ4': (-0.349, 0.349),
            'rh_A_RFJ3': (-0.262, 1.57),
            'rh_A_RFJ0': (0.0, 1.57),
            'rh_A_LFJ5': (0.0, 0.785),
            'rh_A_LFJ4': (-0.349, 0.349),
            'rh_A_LFJ3': (-0.262, 1.57),
            'rh_A_LFJ0': (0.0, 1.57),
            'forearm_tx': (-0.7605, 0.4604999999999999),
            'forearm_ty': (0.0, 0.6)
        }
        self.coupled_joints = coupled_joints

        JOINT_NAMES, JOINT_INDS, HAND_POSITIONS, HAND_INDS, BOUNDS, JOINT_LIMITS = const_rrt
        QPOS_INDS = JOINT_INDS + HAND_INDS
        self.qpos_inds = QPOS_INDS
        self.QPOS_NAMES = JOINT_NAMES + HAND_POSITIONS
        self.resting_poses = {
            "index": [0.2, 0.1, 0.05],
            "middle": [0.2, 0.1, 0.05],
            "ring": [0.2, 0.1, 0.05],
            "little": [0.0, 0.1, 0.05, 0.05],
            "thumb": [0.1, 0.1, 0.0, 0.1, 0.0]
        }

    def compute_goal_orientation(self, env, site_name, key_number=None):
        desired_z_axis = np.array([0, 0, -1], dtype=np.float64)
        desired_x_axis = np.array([1, 0, 0], dtype=np.float64)

        if key_number is not None:
            geom_name = env.task.piano.keys[key_number].geom[0].name
            geom_id = self.model.geom("piano/" + geom_name).id
            if geom_id >= 0:
                key_rot_mat = np.array(env.physics.data.geom_xmat[geom_id].copy(), dtype=np.float64).reshape(3, 3)
                desired_z_axis = -key_rot_mat[:, 2]
                desired_x_axis = key_rot_mat[:, 0]

        desired_y_axis = np.cross(desired_z_axis, desired_x_axis)
        desired_y_axis /= np.linalg.norm(desired_y_axis)
        desired_x_axis = np.cross(desired_y_axis, desired_z_axis)
        desired_x_axis /= np.linalg.norm(desired_x_axis)

        rot_mat = np.column_stack((desired_x_axis, desired_y_axis, desired_z_axis))
        rot_mat_flat = rot_mat.flatten()

        goal_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(goal_quat, rot_mat_flat)
        return goal_quat

    def apply_coupling_constraints(self, dq, joint_names, alpha=0.5):
        for joint1, joint2 in self.coupled_joints:
            if joint1 in joint_names and joint2 in joint_names:
                idx1 = joint_names.index(joint1)
                idx2 = joint_names.index(joint2)
                dq[idx1] = alpha * dq[idx1] + (1 - alpha) * dq[idx2]
                dq[idx2] = alpha * dq[idx2] + (1 - alpha) * dq[idx1]
        return dq

    def clamp_joint_angles(self, q_values, joint_names):
        return np.array([
            np.clip(q_values[i], self.joint_limits[joint][0], self.joint_limits[joint][1])
            if joint in self.joint_limits else q_values[i]
            for i, joint in enumerate(joint_names)
        ])

    def compute_orientation_error(self, current_quat, goal_quat):
        current_quat = current_quat / np.linalg.norm(current_quat)
        goal_quat = goal_quat / np.linalg.norm(goal_quat)
        conjugate_goal = np.array([goal_quat[0], -goal_quat[1], -goal_quat[2], -goal_quat[3]], dtype=np.float64)

        q_error = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(q_error, conjugate_goal, current_quat)
        q_error = q_error / np.linalg.norm(q_error)

        angle = 2 * np.arccos(np.clip(q_error[0], -1.0, 1.0))
        axis = q_error[1:] / np.linalg.norm(q_error[1:]) if np.linalg.norm(q_error[1:]) > 1e-6 else np.zeros(3)
        rot_error = angle * axis
        if np.linalg.norm(rot_error) > np.pi:
            rot_error = rot_error - 2 * np.pi * (rot_error / np.linalg.norm(rot_error))

        print(f"Orientation error (angle, axis): {angle}, {axis}, norm: {np.linalg.norm(rot_error)}")
        return rot_error

    def solve_ik(self, env, data, q_current, q_target_dict, site_names, position_only=False, press_depth=0.1, max_iter=100):
        if len(q_current) != self.model.nv:
            print(f"Warning: q_current shape ({q_current.shape}) does not match model.nv ({self.model.nv}), using full zeros")
            q_current = np.zeros(self.model.nv)
        
        q_solution = q_current.copy()  # Full qpos (114)
        if not self.qpos_inds or len(self.qpos_inds) != 22:
            print(f"Warning: qpos_inds is invalid or length {len(self.qpos_inds)} != 22, returning unchanged q_current")
            return q_solution
        
        shadow_hand_qpos = np.zeros(22)
        shadow_hand_qpos[:] = q_current[self.qpos_inds]

        for finger, joints in self.finger_joints.items():
            if finger not in q_target_dict or finger not in site_names:
                continue
            site_name = site_names[finger]
            q_target = q_target_dict[finger]

            initial_pos, initial_quat = self.compute_fk(env, data, site_name)
            goal_quat = self.compute_goal_orientation(env, site_name)

            pos_error = q_target - initial_pos
            rot_error = self.compute_orientation_error(initial_quat, goal_quat) if not position_only else np.zeros(3)
            error_norm = np.linalg.norm(np.hstack([pos_error, rot_error if not position_only else np.zeros(3)]))

            for iter in range(max_iter):
                
                jac_pos, jac_rot = self.compute_jacobian(env, data, site_name)
                jacobian = np.vstack([jac_pos, jac_rot]) if not position_only else jac_pos
                delta_x = np.hstack([pos_error, rot_error]) if not position_only else pos_error
                
                # Apply press depth during the press phase
                if not position_only and press_depth > 0:
                    delta_x[:3] -= np.array([0, 0, press_depth])  # Move downward to press the key
                    print(f"Adjusted delta_x for press_depth={press_depth}: {delta_x}")

                print(f"Jacobian shape: {jacobian.shape}, delta_x shape: {delta_x.shape}")
                J_hand = jacobian
                jacobian_inv = np.linalg.pinv(J_hand, rcond=self.damping)
                dq = jacobian_inv @ delta_x

                dq = self.apply_coupling_constraints(dq, joints)
                # joint_indices = [self.qpos_inds[self.QPOS_NAMES.index(joint)] for joint in joints if joint in self.QPOS_NAMES]
                # q_solution[joint_indices] += dq[:len(joint_indices)]
                joint_indices = [i for i, name in enumerate(self.QPOS_NAMES) if name in joints]
                print(f"Joint indices for {finger}: {joint_indices}, length: {len(joint_indices)}")
                if not joint_indices:
                    print(f"Warning: No valid joint indices found for finger {finger}, skipping IK update.")
                    continue

                if len(joint_indices) > len(dq):
                    print(f"Warning: Too many joint indices ({len(joint_indices)}) for dq length ({len(dq)}), padding dq")
                    dq = np.pad(dq, (0, len(joint_indices) - len(dq)), mode='constant')
                shadow_hand_qpos[joint_indices] += dq[:len(joint_indices)]
                print(f"Updated hand_qpos shape: {shadow_hand_qpos.shape}, hand_qpos[joint_indices]: {shadow_hand_qpos[joint_indices]}")
                
                # Apply hand position for the current finger
                q_solution[self.qpos_inds] = shadow_hand_qpos
                env.physics.data.qpos[self.qpos_inds] = q_solution[self.qpos_inds]
                mujoco.mj_forward(self.model, data._data)
                new_pos, _ = self.compute_fk(env, data, site_name)
                pos_error = q_target - new_pos
                new_error_norm = np.linalg.norm(pos_error) if position_only else np.linalg.norm(np.hstack([pos_error, rot_error]))
                print(f"Iteration {iter}: Error norm: {new_error_norm}")
                if new_error_norm < 1e-2 or abs(new_error_norm - error_norm) < 1e-3:
                    break
                error_norm = new_error_norm

                target_x, target_y = q_target[0], q_target[1]
                hand_offset_x, hand_offset_y = 0.0, 0.0
                forearm_tx = np.clip(target_x + hand_offset_x, self.joint_limits["forearm_tx"][0], self.joint_limits["forearm_tx"][1])
                forearm_ty = np.clip(target_y + hand_offset_y, self.joint_limits["forearm_ty"][0], self.joint_limits["forearm_ty"][1])
                shadow_hand_qpos[20] = forearm_tx
                shadow_hand_qpos[21] = forearm_ty
                print(f"Updated hand positions: {forearm_tx}, {forearm_ty}")

            # Apply resting poses to other fingers
            for other_finger, other_joints in self.finger_joints.items():
                if other_finger != finger:
                    # other_indices = [self.qpos_inds[self.QPOS_NAMES.index(joint)] for joint in other_joints if joint in self.QPOS_NAMES]
                    # q_solution[other_indices] = self.resting_poses[other_finger][:len(other_indices)]
                    other_indices = [i for i, name in enumerate(self.QPOS_NAMES) if name in other_joints]
                    print(f"other_indices for {other_finger}: {other_indices}, length: {len(other_indices)}")
                    if other_indices and len(other_indices) <= len(self.resting_poses.get(other_finger, [])):
                        shadow_hand_qpos[other_indices] = self.resting_poses[other_finger][:len(other_indices)]

        if self.QPOS_NAMES:
            shadow_hand_qpos = self.clamp_joint_angles(shadow_hand_qpos, self.QPOS_NAMES)
        
        q_solution[self.qpos_inds] = shadow_hand_qpos
        print(f"Final q_solution shape: {q_solution.shape}, q_solution: {q_solution}")
        return q_solution

    def compute_fk(self, env, data, site_name):
        site_id = env.physics.model.site(site_name).id
        if site_id < 0:
            raise ValueError(f"Site {site_name} not found in the model.")
        
        pos = np.array(data.site_xpos[site_id].copy(), dtype=np.float64)
        rot_mat = np.array(data.site_xmat[site_id].copy(), dtype=np.float64).reshape(3, 3)
        rot_mat_flat = rot_mat.flatten()

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, rot_mat_flat)
        return pos, quat

    def compute_jacobian(self, env, data, site_name):
        site_id = env.physics.model.site(site_name).id
        if site_id < 0:
            raise ValueError(f"Site {site_name} not found in the model.")
        
        J_full = np.zeros((6, self.model.nv), dtype=np.float64, order='C')
        jacp = np.ascontiguousarray(J_full[:3], dtype=np.float64)
        jacr = np.ascontiguousarray(J_full[3:], dtype=np.float64)

        model_raw = env.physics.model._model
        data_raw = env.physics.data._data
        mujoco.mj_jacSite(model_raw, data_raw, jacp, jacr, site_id)

        J_full[:3] = jacp
        J_full[3:] = jacr
        J = J_full[:, self.qpos_inds]
        return J[:3, :], J[3:, :]

class PianoMotionPlanner:
    def __init__(self, env, key_sequence, press_durations, fingertip_mapping):
        self.env = env
        self.key_sequence = key_sequence  # List of key indices to press
        self.press_durations = press_durations  # List of durations (seconds) for each key press
        self.fingertip_mapping = fingertip_mapping  # Dict mapping key index to fingertip site (e.g., {0: "thdistal_site"})
        self.ik_solver = InverseKinematicsSolver(env, [], rrt.get_consts_rrt(self.env))  # Using JOINT_INDS as qpos_inds
        self.dt = env.physics.model.opt.timestep  # Simulation timestep

    def plan_motion(self):
        path = []
        current_qpos = self.env.physics.data.qpos.copy()

        for key_idx, duration in zip(self.key_sequence, self.press_durations):
            fingertip_site = self.fingertip_mapping.get(key_idx, "thdistal_site")  # Default to thumb
            key_site_name = self.env.task.piano._sites[key_idx].name
            key_pos = self.env.physics.data.site_xpos[self.env.physics.model.site("piano/" + key_site_name).id].copy()

            # Phase 1: Move toward the key (position-only)
            q_target_dict = {self._get_finger(fingertip_site): key_pos}
            site_names = {self._get_finger(fingertip_site): f"rh_shadow_hand/{fingertip_site}"}
            
            move_qpos = self.ik_solver.solve_ik(
                self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=True
            )
            print(f"move_qpos shape: {move_qpos.shape if move_qpos.size else 'empty'}")
            if self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22 and len(move_qpos):
                current_qpos[self.ik_solver.qpos_inds] = move_qpos[self.ik_solver.qpos_inds]
            else:
                print(f"Warning: Invalid move_qpos shape {move_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
            path.extend(self._interpolate_path(current_qpos, move_qpos, 10))  # 10 steps to move
            current_qpos = move_qpos.copy() if self.ik_solver.qpos_inds else current_qpos

            # Phase 2: Press the key (add press depth)
            press_qpos = self.ik_solver.solve_ik(
                self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=False, press_depth=0.1
            )

            if self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22 and len(press_qpos) == self.env.physics.model.nv:
                current_qpos[self.ik_solver.qpos_inds] = press_qpos[self.ik_solver.qpos_inds]
            else:
                print(f"Warning: Invalid press_qpos shape {press_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
            path.extend(self._interpolate_path(current_qpos, press_qpos, 5))  # 5 steps to press
            current_qpos = press_qpos.copy() if self.ik_solver.qpos_inds else current_qpos

            # Phase 3: Hold the key (repeat current state)
            hold_steps = int(duration / self.dt)
            path.extend([current_qpos.copy() for _ in range(hold_steps)])

            # Phase 4: Release the key (return to move position)
            release_qpos = self.ik_solver.solve_ik(
                self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=True
            )
            if self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22 and len(release_qpos) == self.env.physics.model.nv:
                current_qpos[self.ik_solver.qpos_inds] = release_qpos[self.ik_solver.qpos_inds]
            else:
                print(f"Warning: Invalid release_qpos shape {release_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
            path.extend(self._interpolate_path(current_qpos, release_qpos, 5))  # 5 steps to release
            current_qpos = release_qpos.copy() if self.ik_solver.qpos_inds else current_qpos

        return path
    
    def hybrid_plan_motion(self):
        path = []
        current_qpos = self.env.physics.data.qpos.copy()

        for key_idx, duration in zip(self.key_sequence, self.press_durations):
            fingertip_site = self.fingertip_mapping[key_idx]
            key_site_name = self.env.task.piano._sites[key_idx].name
            key_pos = self.env.physics.data.site_xpos[self.env.physics.model.site("piano/" + key_site_name).id].copy()

            q_target_dict = {self._get_finger(fingertip_site): key_pos}
            site_names = {self._get_finger(fingertip_site): f"rh_shadow_hand/{fingertip_site}"}

            # Phase 1: Move toward the key
            move_qpos = self.ik_solver.solve_ik(self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=True)
            if move_qpos is not None and self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22:
                # Extract the fingertip position for the move phase
                move_fingertip_pos, _ = self.ik_solver.compute_fk(self.env, self.env.physics.data, site_names[self._get_finger(fingertip_site)])
                rrt_planner = rrt.RRTPlanner(self.env, move_fingertip_pos, "rh_shadow_hand/" + fingertip_site, goal_key_number=None, max_iter=100) # goal_tolerance = 0.05, EPSILON=0.1)
                rrt_path = rrt_planner.plan()
                if rrt_path is None:
                    print(f"<move> Failed to find RRT for key {key_idx}, skipping")
                    return path
                path.extend([self._full_qpos(current_qpos, rrt_q) for rrt_q in rrt_path])
                current_qpos = self._full_qpos(current_qpos, rrt_path[-1])
            else:
                print(f"<move> Warning: Invalid move_qpos shape {move_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
                return path

            # Phase 2: Press the key
            press_qpos = self.ik_solver.solve_ik(self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=False, press_depth=0.1)
            if press_qpos is not None and self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22:
                # Extract the fingertip position for the press phase
                self.env.physics.data.qpos = press_qpos
                mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
                press_fingertip_pos, _ = self.ik_solver.compute_fk(self.env, self.env.physics.data, site_names[self._get_finger(fingertip_site)])
                rrt_planner = rrt.RRTPlanner(self.env, press_fingertip_pos, "rh_shadow_hand/" + fingertip_site, goal_key_number=None, max_iter=100) # goal_tolerance = 0.05, EPSILON=0.1)
                rrt_path = rrt_planner.plan()
                if rrt_path is None:
                    print(f"<press> Failed to find RRT for key {key_idx}, skipping")
                    return path
                path.extend([self._full_qpos(current_qpos, rrt_q) for rrt_q in rrt_path])
                current_qpos = self._full_qpos(current_qpos, rrt_path[-1])
            else:
                print(f"<press> Warning: Invalid press_qpos shape {press_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
                return path

            # Phase 3: Hold the key
            hold_steps = int(duration / self.dt)
            path.extend([current_qpos.copy() for _ in range(hold_steps)])

            # Phase 4: Release the key
            release_qpos = self.ik_solver.solve_ik(self.env, self.env.physics.data, current_qpos, q_target_dict, site_names, position_only=True)
            if release_qpos is not None and self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22:
                self.env.physics.data.qpos = release_qpos
                mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
                release_fingertip_pos, _ = self.ik_solver.compute_fk(self.env, self.env.physics.data, site_names[self._get_finger(fingertip_site)])
                rrt_planner = rrt.RRTPlanner(self.env, release_fingertip_pos, "rh_shadow_hand/" + fingertip_site, goal_key_number=None, max_iter=100) # goal_tolerance = 0.05, EPSILON=0.1)
                rrt_path = rrt_planner.plan()
                if rrt_path is None:
                    print(f"<press> Failed to find RRT for key {key_idx}, skipping")
                    return path
                path.extend([self._full_qpos(current_qpos, rrt_q) for rrt_q in rrt_path])
                # current_qpos[self.ik_solver.qpos_inds] = rrt_path[-1]
                current_qpos = self._full_qpos(current_qpos, rrt_path[-1])
            else:
                print(f"<press> Warning: Invalid release_qpos shape {release_qpos.shape} or qpos_inds length {len(self.ik_solver.qpos_inds)}, using current_qpos")
                return path
        
        return path
    
    def _full_qpos(self, base_qpos, hand_qpos):
        full_qpos = base_qpos.copy()
        full_qpos[self.ik_solver.qpos_inds] = hand_qpos
        return full_qpos

    def _get_finger(self, site_name):
        finger_map = {"thdistal": "thumb", "ffdistal": "index", "mfdistal": "middle", "rfdistal": "ring", "lfdistal": "little"}
        return finger_map[site_name.split("_")[0]]

    def _interpolate_path(self, start_qpos, end_qpos, steps):
        path = []
        for t in np.linspace(0, 1, steps):
            print(f"shapes: {start_qpos.shape}, {end_qpos.shape}")
            interp_qpos = start_qpos + t * (end_qpos - start_qpos)
            path.append(interp_qpos)
        return path

    def execute(self, path):
        for qpos in path:
            if self.ik_solver.qpos_inds and len(self.ik_solver.qpos_inds) == 22:
                self.env.physics.data.qpos[self.ik_solver.qpos_inds] = qpos[self.ik_solver.qpos_inds]
            else:
                print(f"Warning: Invalid qpos_inds length {len(self.ik_solver.qpos_inds)}, skipping update")
            mujoco.mj_forward(self.env.physics.model._model, self.env.physics.data._data)
            self.env.render()  # Assuming render is available
            self.env.step()    # Advance simulation