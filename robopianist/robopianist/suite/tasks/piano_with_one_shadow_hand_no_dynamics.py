# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One-handed version of `piano_with_shadow_hands.py`."""

from typing import List, Optional, Sequence, Tuple

import numpy as np
from dm_control.composer import variation as base_variation
from dm_control.composer.observation import observable
from dm_control.utils.rewards import tolerance
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_env import specs
from mujoco_utils import spec_utils
import mujoco

import robopianist.models.hands.shadow_hand_constants as hand_consts
from robopianist.models.arenas import stage
from robopianist.models.hands import HandSide
from robopianist.music import midi_file
from robopianist.suite import composite_reward
from robopianist.suite.tasks import base

# Distance thresholds for the shaping reward.
_FINGER_CLOSE_ENOUGH_TO_KEY = 0.01
_KEY_CLOSE_ENOUGH_TO_PRESSED = 0.05

# Energy penalty coefficient.
_ENERGY_PENALTY_COEF = 5e-3

_NUM_STEPS_PER_SEGMENT = 10

_FINGER_JOINTS = [
    ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1'],
    ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ0'],
    ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ0'],
    ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ0'],
    ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ0'],
]

_WRIST_JOINTS = ['rh_WRJ2', 'rh_WRJ1']

_FOREARM_JOINTS = ['forearm_tx', 'forearm_ty']

_FULL_JOINTS = [
    _WRIST_JOINTS + ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1'] + _FOREARM_JOINTS,
]

_FULL_JOINTS_ONE_ARRAY = _WRIST_JOINTS + [
    'rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1',
    'rh_FFJ4', 'rh_FFJ3', 'rh_FFJ2', 'rh_FFJ1',
    'rh_MFJ4', 'rh_MFJ3', 'rh_MFJ2', 'rh_MFJ1',
    'rh_RFJ4', 'rh_RFJ3', 'rh_RFJ2', 'rh_RFJ1',
    'rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ2', 'rh_LFJ1',
] + _FOREARM_JOINTS



class PianoWithOneShadowHand(base.PianoTask):
    def __init__(
        self,
        midi: midi_file.MidiFile,
        hand_side: HandSide,
        n_steps_lookahead: int = 1,
        n_seconds_lookahead: Optional[float] = None,
        trim_silence: bool = False,
        wrong_press_termination: bool = False,
        initial_buffer_time: float = 0.0,
        disable_fingering_reward: bool = False,
        disable_colorization: bool = False,
        augmentations: Optional[Sequence[base_variation.Variation]] = None,
        **kwargs,
    ) -> None:
        # super().__init__(arena=None, **kwargs)
        super().__init__(arena=stage.Stage(), **kwargs)

        self._midi = midi
        self._hand_side = hand_side
        self._hand = self._right_hand if hand_side == HandSide.RIGHT else self._left_hand
        self._t_idx = 0
        self._notes = [[] for _ in range(int(midi.seq.notes[-1].end_time / 0.05) + 1)]
        for note in midi.seq.notes:
            t = int(note.start_time / 0.05)
            self._notes[t].append(note)
        self._keys_current = []
        self._trajectories = [[] for _ in range(5)]
        self._traj_steps = [0 for _ in range(5)]
        self._last_planned_keys = [None] * 5
        self._metrics = {"collision_rate": 0, "success_rate": 0, "planning_time": 0}
        self._set_rewards()

    def _set_rewards(self):
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            energy_reward=lambda physics: 0.0,  # No dynamics, no energy
        )

    def _plan_with_rrt(self, key: int, finger: int, physics) -> List[np.ndarray]:
        import time
        start_time = time.time()
        start_qpos = np.zeros(len(_FINGER_JOINTS[finger]))
        fingertip_site = self._hand.fingertip_sites[finger]
        key_site = self.piano.keys[key].site[0]
        key_pos = physics.bind(key_site).xpos.copy()
        key_pos[2] += 0.005
        full_site_name = f"rh_shadow_hand/{fingertip_site.name}" if self._hand_side == HandSide.RIGHT else f"lh_shadow_hand/{fingertip_site.name}"
        # Set the resting position
        original_qpos = physics.data.qpos.copy()
        physics.data.qpos[:] = self._resting_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)

        # Get the current finger position (start)
        # start_qpos = np.zeros(len(self._finger_joints[finger]))
        for i, joint_name in enumerate(_FINGER_JOINTS[finger]):
            if "J0" in joint_name:
                # joint_name[-1] = "2"
                # joint_idx = physics.model.name2id(joint_name, "joint")
                joint_idx = physics.model.name2id(joint_name, "tendon")
            else:
                joint_idx = physics.model.name2id(joint_name, "joint")
            start_qpos[i] = physics.data.qpos[joint_idx]
        print(f"Finger {finger} start qpos: {start_qpos}")

        fingertip_site = self._hand.fingertip_sites[finger]
        fingertip_pos = physics.bind(fingertip_site).xpos.copy()
        fingertip_xmat = physics.bind(fingertip_site).xmat.copy()
        fingertip_xmat = fingertip_xmat.reshape(3, 3)
        print(f"Finger {finger} fingertip pos: {fingertip_pos}")

        # Get target key position in world frame
        key_site = self.piano.keys[key].site[0]
        key_pos = physics.bind(key_site).xpos.copy()
        print(f"Key {key} position: {key_pos}")

        # Adjust key position for presing
        press_pos = key_pos.copy()
        press_pos[2] += 0.005
        print(f"Adjusted press position (just above key): {press_pos}")

        # Convert press_pos to the fingertip's local frame
        relative_pos = press_pos - fingertip_pos
        fingertip_xmat_inv = np.linalg.inv(fingertip_xmat)
        local_press_pos = fingertip_xmat_inv @ relative_pos
        print(f"Finger {finger} local press position: {local_press_pos}")

        # if local_press_pos[2] > 0:
        #     local_press_pos[2] = -local_press_pos[2]
        print(f"Finger {finger}: Target position in local frame (after fix) {local_press_pos}")

        site_name = self._hand.fingertip_sites[finger].name
        if self._hand_side == HandSide.LEFT:
            full_site_name = f"lh_shadow_hand/{site_name}"
        else:
            full_site_name = f"rh_shadow_hand/{site_name}"

        ik_result = qpos_from_site_pose(
            physics,
            full_site_name,
            local_press_pos,
            None,
            self._full_finger_joints[finger], # excluding wrist for now: + self._wrist_joints,
            tol=1e-2,
            max_steps=200,
            regularization_threshold = 0.01,
            regularization_strength = 0.1,
            max_update_norm = 1.0,
            progress_thresh = 50.0,
        )

        if ik_result.err_norm > 0.2:
            print(f"Finger {finger}: IK failed to converge, err_norm={ik_result.err_norm}")
            print(f"Finger {finger}: Falling back to projection toward goal position")
            goal_qpos = self._project_toward_goal(
                physics,
                finger,
                start_qpos,
                fingertip_pos,
                press_pos
            )

            if goal_qpos is None:
                physics.data.qpos[:] = original_qpos
                mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
                return []
        else:

            goal_qpos = np.zeros(len(self._finger_joints[finger]))
            for i, joint_name in enumerate(self._finger_joints[finger]):
                if "J0" in joint_name:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                else:
                    joint_idx = physics.model.name2id(joint_name, "joint")
                goal_qpos[i] = physics.data.qpos[joint_idx]
        
        print(f"Finger {finger} goal qpos: {goal_qpos}")

        # Joint limits
        joint_limits = np.zeros((len(self._finger_joints[finger]), 2))
        for i, joint_name in enumerate(self._finger_joints[finger]):
            if "J0" in joint_name:
                joint_idx = physics.model.name2id(joint_name, "tendon")
            else:
                joint_idx = physics.model.name2id(joint_name, "joint")
            joint_limits[i] = physics.model.jnt_range[joint_idx]
        print(f"Finger {finger} joint limits: {joint_limits}")

        # RRT parameters
        step_size = 0.5
        max_iterations = 1000
        goal_bias = 0.2

        # Initialize RRT
        tree = [(start_qpos, None)]
        path_found = False
        
        for iteration in range(max_iterations):
            if np.random.random() < goal_bias:
                rand_qpos = goal_qpos
            else:
                rand_qpos = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
            
            distances = [np.linalg.norm(qpos - rand_qpos) for qpos, _ in tree]
            nearest_idx = np.argmin(distances)
            q_near, _ = tree[nearest_idx]

            direction = rand_qpos - q_near
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                continue
            direction = direction / distance
            q_new = q_near + min(step_size, distance) * direction

            q_new = np.clip(q_new, joint_limits[:, 0], joint_limits[:, 1])

            if not self._check_collision(physics, finger, q_new):
                tree.append((q_new, nearest_idx))

                if np.linalg.norm(q_new - goal_qpos) < 2 * step_size:
                    print(f"Finger {finger}: Found path to goal in {iteration} iterations")
                    path_found = True
                    break
        
        if not path_found:
            print(f"Finger {finger}: Path not found after {max_iterations} iterations")
            physics.data.qpos[:] = original_qpos
            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
            return []
    
        path = []
        current_idx = len(tree) - 1
        while current_idx is not None:
            qpos, parent_idx = tree[current_idx]
            path.append(qpos)
            current_idx = parent_idx
        path.reverse()

        smooth_path = []
        num_steps = _NUM_STEPS_PER_SEGMENT
        for i in range(len(path) - 1):
            q0 = path[i]
            q1 = path[i + 1]
            for j in range(num_steps):
                alpha = j / num_steps
                q = (1 - alpha) * q0 + alpha * q1
                smooth_path.append(q)

        print(f"RRT planning took {time.time() - start_time:.2f} seconds")
        physics.data.qpos[:] = original_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        return smooth_path

    def _check_collision_kinematic(self, physics, finger, qpos):
        # Simplified collision check without dynamics
        return False  # Assume collision-free for now; implement if needed

    def _update_hand_position(self, physics):
        self._keys_current = [(note.pitch - 21, i % 5) for i, note in enumerate(self._notes[self._t_idx])]
        action = np.zeros(23)
        for key, finger in self._keys_current:
            if self._last_planned_keys[finger] != key or not self._trajectories[finger]:
                self._trajectories[finger] = self._plan_with_rrt(key, finger, physics)
                self._traj_steps[finger] = 0
                self._last_planned_keys[finger] = key
            if self._traj_steps[finger] < len(self._trajectories[finger]):
                action[finger * 5: finger * 5 + len(_FINGER_JOINTS[finger])] = self._trajectories[finger][self._traj_steps[finger]]
                self._traj_steps[finger] += 1
        return action

    def before_step(self, physics, action, random_state):
        pass  # No dynamics applied

    def after_step(self, physics, random_state):
        self._t_idx += 1

    def get_action(self, physics):
        return self._update_hand_position(physics)

    def get_custom_metrics(self):
        return self._metrics.copy()

    def _compute_key_press_reward(self, physics):
        return 1.0 if self._keys_current else 0.0  # Simplified for no dynamics