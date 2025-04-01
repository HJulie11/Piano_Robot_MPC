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

_WRIST_JOINTS = ['rh_WRJ2', 'rh_WRJ1']

_FOREARM_JOINTS = ['forearm_tx', 'forearm_ty']

_FINGER_JOINTS = [
    _WRIST_JOINTS + ['rh_THJ5', 'rh_THJ4', 'rh_THJ3', 'rh_THJ2', 'rh_THJ1'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_FFJ4', 'rh_FFJ3', 'rh_FFJ0'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_MFJ4', 'rh_MFJ3', 'rh_MFJ0'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_RFJ4', 'rh_RFJ3', 'rh_RFJ0'] + _FOREARM_JOINTS,
    _WRIST_JOINTS + ['rh_LFJ5', 'rh_LFJ4', 'rh_LFJ3', 'rh_LFJ0'] + _FOREARM_JOINTS,
]

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
        self._trajectories = self._generate_fixed_trajectory()
        self._traj_steps = [0 for _ in range(5)]
        self._metrics = {"timing_error": 0, "tracking_error": 0, "return_speed": 0}
        self._last_qpos = None
        self._set_rewards()

    def _set_rewards(self):
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            energy_reward=self._compute_energy_reward,
        )

    def _generate_fixed_trajectory(self):
        # Fixed trajectory for consistency
        traj = []
        for note in self._midi.seq.notes:
            key = note.pitch - 21
            finger = (key - 48) % 5  # Simple assignment
            qpos = np.zeros(23)
            qpos[finger * 5: finger * 5 + 5] = [0.1] * 5  # Dummy press
            traj.extend([qpos] * _NUM_STEPS_PER_SEGMENT)
        return [traj] * 5  # Same for all fingers

    def _update_hand_position(self, physics):
        action = np.zeros(23)
        for finger in range(5):
            if self._traj_steps[finger] < len(self._trajectories[finger]):
                target_qpos = self._trajectories[finger][self._traj_steps[finger]]
                current_qpos = physics.data.qpos[finger * 5: finger * 5 + 5]
                error = target_qpos[finger * 5: finger * 5 + 5] - current_qpos
                action[finger * 5: finger * 5 + 5] = 50.0 * error
                self._traj_steps[finger] += 1
                self._metrics["tracking_error"] += np.mean(np.abs(error))
        return action

    def before_step(self, physics, action, random_state):
        self._hand.apply_action(physics, action[:-1], random_state)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)

    def after_step(self, physics, random_state):
        self._t_idx += 1
        for key, _ in self._keys_current:
            actual_time = self._t_idx * 0.05
            desired_time = self._notes[self._t_idx - 1][0].start_time
            self._metrics["timing_error"] += abs(actual_time - desired_time)
        if self._last_qpos is not None and not self._keys_current:
            self._metrics["return_speed"] += np.mean(np.abs(physics.data.qpos - self._last_qpos)) / 0.05
        self._last_qpos = physics.data.qpos.copy()

    def get_action(self, physics):
        self._keys_current = [(note.pitch - 21, i % 5) for i, note in enumerate(self._notes[self._t_idx])]
        return self._update_hand_position(physics)

    def get_custom_metrics(self):
        return self._metrics.copy()

    def _compute_key_press_reward(self, physics):
        return tolerance(self.piano.state[self._keys_current[0][0]], bounds=(0, 0.05))

    def _compute_energy_reward(self, physics):
        return -_ENERGY_PENALTY_COEF * np.sum(self._hand.observables.actuators_power(physics))