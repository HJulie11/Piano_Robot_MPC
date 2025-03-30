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
        self._metrics = {"oscillation_amplitude": 0, "timing_error": 0, "press_depth_var": 0}
        self._last_qpos = None
        self._step_count = 0
        self._set_rewards()

    def _set_rewards(self):
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            energy_reward=self._compute_energy_reward,
        )

    def _plan_with_rrt(self, key: int, finger: int, physics) -> list[np.ndarray]:
        import time
        start_time = time.time()
        print(f"Finger {finger} planning for key {key}...")

        original_qpos = physics.data.qpos.copy()

        # Get current finger position (start)
        start_qpos = np.zeros(len(_FULL_JOINTS[finger]))
        for i, joint_name in enumerate(_FULL_JOINTS[finger]):
            joint_idx = physics.model.name2id(joint_name, "tendon" if "J0" in joint_name else "joint")
            start_qpos[i] = physics.data.qpos[joint_idx]
        print(f"Finger {finger} start qpos: {start_qpos}")

        # Get fingertip and key positions
        fingertip_site = self._hand.fingertip_sites[finger]
        fingertip_pos = physics.bind(fingertip_site).xpos.copy()
        fingertip_xmat = physics.bind(fingertip_site).xmat.copy().reshape(3, 3)
        key_site = self.piano.keys[key].site[0]
        key_id = physics.model.name2id(f"piano/{key_site.name}", "site")
        key_pos = physics.data.site_xpos[key_id].copy()
        press_pos = key_pos.copy()
        press_pos[2] -= 0.005

        # Convert to local frame and solve IK
        relative_pos = press_pos - fingertip_pos
        local_press_pos = np.linalg.inv(fingertip_xmat) @ relative_pos
        site_name = f"rh_shadow_hand/{fingertip_site.name}" if self._hand_side == HandSide.RIGHT else f"lh_shadow_hand/{fingertip_site.name}"
        ik_result = qpos_from_site_pose(
            physics,
            site_name,
            local_press_pos,
            None,
            _FULL_JOINTS[finger],
            tol=1e-2,
            max_steps=200,
            regularization_threshold=0.01,
            regularization_strength=0.1,
            max_update_norm=1.0,
            progress_thresh=50.0,
        )

        if ik_result.err_norm > 0.2:
            print(f"Finger {finger}: IK failed, err_norm={ik_result.err_norm}")
            physics.data.qpos[:] = original_qpos
            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
            return []
        else:
            print(f"Finger {finger}: IK succeeded, err_norm={ik_result.err_norm}")
            goal_qpos = np.array([physics.data.qpos[physics.model.name2id(j, "tendon" if "J0" in j else "joint")] for j in _FULL_JOINTS[finger]])

        # Joint limits
        joint_limits = np.array([physics.model.jnt_range[physics.model.name2id(j, "tendon" if "J0" in j else "joint")] for j in _FULL_JOINTS[finger]])

        # RRT
        step_size = 0.5
        max_iterations = 1000
        goal_bias = 0.2
        tree = [(start_qpos, None)]
        path_found = False

        for iteration in range(max_iterations):
            rand_qpos = goal_qpos if np.random.random() < goal_bias else np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
            nearest_idx = np.argmin([np.linalg.norm(qpos - rand_qpos) for qpos, _ in tree])
            q_near, _ = tree[nearest_idx]
            direction = rand_qpos - q_near
            distance = np.linalg.norm(direction)
            if distance < 1e-6:
                continue
            q_new = q_near + min(step_size, distance) * direction / distance
            q_new = np.clip(q_new, joint_limits[:, 0], joint_limits[:, 1])
            if not self._check_collision(physics, finger, q_new):
                tree.append((q_new, nearest_idx))
                if np.linalg.norm(q_new - goal_qpos) < 2 * step_size:
                    path_found = True
                    break

        if not path_found:
            print(f"Finger {finger}: Path not found")
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

        smooth_path = [path[i] + (path[i + 1] - path[i]) * j / _NUM_STEPS_PER_SEGMENT for i in range(len(path) - 1) for j in range(_NUM_STEPS_PER_SEGMENT)]
        print(f"RRT planning took {time.time() - start_time:.2f} seconds")
        physics.data.qpos[:] = original_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        return smooth_path

    def _check_collision(self, physics, finger, qpos):
        original_qpos = physics.data.qpos.copy()
        for i, joint_name in enumerate(_FULL_JOINTS[finger]):
            joint_idx = physics.model.name2id(joint_name, "tendon" if "J0" in joint_name else "joint")
            physics.data.qpos[joint_idx] = qpos[i]
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        collision = bool(mujoco.mj_contactCount(physics.model.ptr, physics.data.ptr))
        physics.data.qpos[:] = original_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        return collision

    def _update_hand_position(self, physics):
        self._keys_current = [(note.pitch - 21, i % 5) for i, note in enumerate(self._notes[self._t_idx])]
        action = np.zeros(23)
        for key, finger in self._keys_current:
            if self._last_planned_keys[finger] != key or not self._trajectories[finger]:
                self._trajectories[finger] = self._plan_with_rrt(key, finger, physics)
                self._traj_steps[finger] = 0
                self._last_planned_keys[finger] = key
            if self._traj_steps[finger] < len(self._trajectories[finger]):
                qpos_finger = self._trajectories[finger][self._traj_steps[finger]]
                for i, joint_name in enumerate(_FULL_JOINTS[finger]):
                    joint_idx = physics.model.name2id(joint_name, "tendon" if "J0" in joint_name else "joint")
                    error = qpos_finger[i] - physics.data.qpos[joint_idx]
                    action_idx = next(i for i, act in enumerate(self._hand.actuators) if act.joint.name == joint_name or (hasattr(act, "tendon") and act.tendon.name == joint_name))
                    action[action_idx] = 50.0 * error
                self._traj_steps[finger] += 1
        return action

    def before_step(self, physics, action, random_state):
        self._hand.apply_action(physics, action[:-1], random_state)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)
        if self._last_qpos is not None:
            self._metrics["oscillation_amplitude"] += np.mean(np.abs(physics.data.qpos - self._last_qpos))
        self._last_qpos = physics.data.qpos.copy()

    def after_step(self, physics, random_state):
        self._t_idx += 1
        self._step_count += 1
        for key, _ in self._keys_current:
            actual_time = self._t_idx * 0.05
            desired_time = self._notes[self._t_idx - 1][0].start_time
            self._metrics["timing_error"] += abs(actual_time - desired_time)
            self._metrics["press_depth_var"] += np.var([self.piano.state[key]])

    def get_action(self, physics):
        return self._update_hand_position(physics)

    def get_custom_metrics(self):
        metrics = self._metrics.copy()
        if self._step_count > 0:
            metrics["oscillation_amplitude"] /= self._step_count
            metrics["timing_error"] /= self._step_count
            metrics["press_depth_var"] /= self._step_count
        return metrics

    def _compute_key_press_reward(self, physics):
        return sum(tolerance(self.piano.state[key], bounds=(0, 0.05)) for key, _ in self._keys_current) / max(1, len(self._keys_current))

    def _compute_energy_reward(self, physics):
        return -_ENERGY_PENALTY_COEF * np.sum(self._hand.observables.actuators_power(physics))