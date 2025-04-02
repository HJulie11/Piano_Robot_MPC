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

"""One-handed version of `piano_with_shadow_hands.py` with dynamics and note velocities."""

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
        """Task constructor.

        Args:
            midi: A `MidiFile` object.
            n_steps_lookahead: Number of timesteps to look ahead when computing the
                goal state.
            n_seconds_lookahead: Number of seconds to look ahead when computing the
                goal state. If specified, this will override `n_steps_lookahead`.
            trim_silence: If True, shifts the MIDI file so that the first note starts
                at time 0.
            wrong_press_termination: If True, terminates the episode if the hands press
                the wrong keys at any timestep.
            initial_buffer_time: Specifies the duration of silence in seconds to add to
                the beginning of the MIDI file. A non-zero value can be useful for
                giving the agent time to place its hands near the first notes.
            disable_fingering_reward: If True, disables the shaping reward for
                fingering. This will also disable the colorization of the fingertips
                and corresponding keys.
            disable_colorization: If True, disables the colorization of the fingertips
                and corresponding keys.
            augmentations: A list of `Variation` objects that will be applied to the
                MIDI file at the beginning of each episode. If None, no augmentations
                will be applied.
        """
        super().__init__(arena=stage.Stage(), **kwargs)

        self._resting_qpos = None

        if trim_silence:
            midi = midi.trim_silence()
        self._midi = midi
        self._n_steps_lookahead = n_steps_lookahead
        if n_seconds_lookahead is not None:
            self._n_steps_lookahead = int(
                np.ceil(n_seconds_lookahead / self.control_timestep)
            )
        self._initial_buffer_time = initial_buffer_time
        self._disable_fingering_reward = disable_fingering_reward
        self._wrong_press_termination = wrong_press_termination
        self._disable_colorization = disable_colorization
        self._augmentations = augmentations

        self._hand_side = hand_side
        if self._hand_side == HandSide.LEFT:
            self._hand = self._left_hand
            self._right_hand.detach()
            self._finger_joints = [[f"lh_shadow_hand/{j}" for j in joints] for joints in _FINGER_JOINTS]
            self._full_finger_joints = [[f"lh_shadow_hand/{j}" for j in joints] for joints in _FULL_JOINTS]
            self._full_joints_one_array = [f"lh_shadow_hand/{j}" for j in _FULL_JOINTS_ONE_ARRAY]
            self._wrist_joints = [f"lh_shadow_hand/{j}" for j in _WRIST_JOINTS]
            self._forearm_joints = [f"lh_shadow_hand/{j}" for j in _FOREARM_JOINTS]
        else:
            self._hand = self._right_hand
            self._left_hand.detach()
            self._finger_joints = [[f"rh_shadow_hand/{j}" for j in joints] for joints in _FINGER_JOINTS]
            self._full_finger_joints = [[f"rh_shadow_hand/{j}" for j in joints] for joints in _FULL_JOINTS]
            self._full_joints_one_array = [f"rh_shadow_hand/{j}" for j in _FULL_JOINTS_ONE_ARRAY]
            self._wrist_joints = [f"rh_shadow_hand/{j}" for j in _WRIST_JOINTS]
            self._forearm_joints = [f"rh_shadow_hand/{j}" for j in _FOREARM_JOINTS]
        
        if not disable_fingering_reward and not disable_colorization:
            self._colorize_fingertips()
        self._reset_quantities_at_episode_init()
        self._reset_trajectory(self._midi)  # Important: call before adding observables.
        self._thumb_under = False
        self._last_finger = None
        self._last_key = None
        self._metrics = {"collition_rate": 0, "success_rate": 0, "planning_time": 0}
        self._add_observables()
        self._set_rewards()

        # Motion planning states
        self._trajectories = [[] for _ in range(5)]
        self._traj_steps = [0 for _ in range(5)]
        self._last_planned_keys = [None] * 5

        self._last_action = np.zeros(23)

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            key_press_reward=self._compute_key_press_reward,
            sustain_reward=self._compute_sustain_reward,
            energy_reward=self._compute_energy_reward,
        )
        if not self._disable_fingering_reward:
            self._reward_fn.add("finger_movement_reward", self._compute_fingering_reward)

    def _reset_quantities_at_episode_init(self) -> None:
        self._t_idx: int = 0
        self._should_terminate: bool = False
        self._discount: float = 1.0

        self._notes = []
        self._keys = []
        self._keys_current = []
        self._fingering_state = np.zeros((5,), dtype=np.float64)

        self._trajectories = [[] for _ in range(5)]
        self._traj_steps = [0 for _ in range(5)]
        self._last_planned_keys = [None] * 5
        self._last_action = None
        self._sustain_state = 0.0

        self._last_finger = None
        self._last_key = None

    def _maybe_change_midi(self, random_state) -> None:
        if self._augmentations is not None:
            midi = self._midi
            for var in self._augmentations:
                midi = var(initial_value=midi, random_state=random_state)
            self._reset_trajectory(midi)

    def _reset_trajectory(self, midi: midi_file.MidiFile) -> None:
        print(f"MIDI file notes: {midi.seq.notes}")
        midi_duration = midi.seq.notes[-1].end_time if midi.seq.notes else 0
        print(f"MIDI duration: {midi_duration} seconds")
        total_duration = midi_duration + self._initial_buffer_time
        num_steps = int(total_duration / self.control_timestep) + 1
        print(f"Control timestep: {self.control_timestep}, Initial buffer time: {self._initial_buffer_time}, Number of timesteps: {num_steps}")
        self._notes = [[] for _ in range(num_steps)]
        for note in midi.seq.notes:
            adjusted_start_time = note.start_time + self._initial_buffer_time
            t = int(adjusted_start_time / self.control_timestep)
            if t < num_steps:
                print(f"Note: {note}, Original start time: {note.start_time}, Adjusted start time: {adjusted_start_time}, Timestep: {t}, Velocity: {note.velocity}")
                # Store the note and its velocity as a tuple
                self._notes[t].append((note, note.velocity))
            else:
                print(f"Warning: Note at adjusted time {adjusted_start_time} exceeds total duration, skipping")
        self._sustain_events = []
        if hasattr(midi.seq, "events"):
            for event in midi.seq.events:
                if hasattr(event, "type") and event.type == "control_change" and event.control == 64:
                    adjusted_time = event.time + self._initial_buffer_time
                    t = int(adjusted_time / self.control_timestep)
                    if t < num_steps:
                        self._sustain_events.append((t, event.value / 127.0))
        print(f"Notes per timestep: {self._notes}")

    def _set_resting_position(self, physics) -> np.ndarray:
        resting_qpos = np.zeros(physics.data.qpos.shape)
        for finger in range(5):
            for joint_name in self._full_finger_joints[finger]:
                joint_idx = physics.model.name2id(joint_name, "joint")
                if joint_idx == -1:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                    if joint_idx == -1:
                        raise ValueError(f"Joint {joint_name} not found")
                joint_range = physics.model.jnt_range[joint_idx]
                resting_angle = joint_range[0] + 0.1 * (joint_range[1] - joint_range[0])
                resting_qpos[joint_idx] = resting_angle
        physics.data.qpos[:] = resting_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
        return resting_qpos

    def _plan_with_rrt(self, key: int, finger: int, physics) -> List[np.ndarray]:
        import time
        start_time = time.time()
        print(f"Finger {finger} planning for key {key}...")

        # Set the resting position
        original_qpos = physics.data.qpos.copy()
        physics.data.qpos[:] = self._resting_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics

        # Get the current finger position (start)
        start_qpos = np.zeros(len(self._finger_joints[finger]))
        for i, joint_name in enumerate(self._finger_joints[finger]):
            if "J0" in joint_name:
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

        # Adjust key position for pressing
        press_pos = key_pos.copy()
        press_pos[2] -= 0.01
        print(f"Adjusted press position (just above key): {press_pos}")

        # Convert press_pos to the fingertip's local frame
        relative_pos = press_pos - fingertip_pos
        fingertip_xmat_inv = np.linalg.inv(fingertip_xmat)
        local_press_pos = fingertip_xmat_inv @ relative_pos
        print(f"Finger {finger} local press position: {local_press_pos}")

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
            self._full_finger_joints[finger],
            tol=1e-2,
            max_steps=200,
            regularization_threshold=0.01,
            regularization_strength=0.1,
            max_update_norm=1.0,
            progress_thresh=50.0,
        )

        if ik_result.err_norm > 0.1:
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
                print(f"Finger {finger}: Projection failed, returning empty trajectory")
                physics.data.qpos[:] = original_qpos
                mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
                return []
        else:
            print(f"Finger {finger}: IK succeeded, err_norm={ik_result.err_norm}")
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
            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
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
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
        return smooth_path
    
    def _project_toward_goal(self, physics, finger, start_qpos, fingertip_pos, target_pos) -> Optional[np.ndarray]:
        current_qpos = start_qpos.copy()
        current_fingertip_pos = fingertip_pos.copy()
        max_projection_steps = 10
        step_size = 0.005

        for step in range(max_projection_steps):
            direction = target_pos - current_fingertip_pos
            distance = np.linalg.norm(direction)
            if distance < 1e-3:
                print(f"Finger {finger}: Projection converged in {step} steps")
                return current_qpos
        
            direction = direction / distance
            intermediate_pos = current_fingertip_pos + step_size * direction
            print(f"Finger {finger}: Intermediate position: {intermediate_pos}")

            fingertip_site = self._hand.fingertip_sites[finger]
            fingertip_xmat = physics.bind(fingertip_site).xmat.copy()
            fingertip_xmat = fingertip_xmat.reshape(3, 3)
            fingertip_xmat_inv = np.linalg.inv(fingertip_xmat)
            local_intermediate_pos = fingertip_xmat_inv @ (intermediate_pos - current_fingertip_pos)

            if local_intermediate_pos[2] > 0:
                local_intermediate_pos[2] = -local_intermediate_pos[2]
            print(f"Finger {finger}: Local intermediate position: {local_intermediate_pos}")

            site_name = self._hand.fingertip_sites[finger].name
            if self._hand_side == HandSide.LEFT:
                full_site_name = f"lh_shadow_hand/{site_name}"
            else:
                full_site_name = f"rh_shadow_hand/{site_name}"

            ik_result = qpos_from_site_pose(
                physics,
                full_site_name,
                local_intermediate_pos,
                None,
                self._full_finger_joints[finger],
                tol=1e-1,
                max_steps=200,
                regularization_threshold=0.01,
                regularization_strength=0.1,
                max_update_norm=1.0,
                progress_thresh=50.0,
            )

            if ik_result.err_norm > 0.2:
                print(f"Finger {finger}: IK failed for intermediate position, err_norm={ik_result.err_norm}")
                break

            current_qpos = np.zeros(len(self._finger_joints[finger]))
            for i, joint_name in enumerate(self._finger_joints[finger]):
                if "J0" in joint_name:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                else:
                    joint_idx = physics.model.name2id(joint_name, "joint")
                current_qpos[i] = physics.data.qpos[joint_idx]

            original_qpos = physics.data.qpos.copy()
            for i, joint_name in enumerate(self._finger_joints[finger]):
                if "J0" in joint_name:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                else:
                    joint_idx = physics.model.name2id(joint_name, "joint")
                joint_range = physics.model.jnt_range[joint_idx]
                physics.data.qpos[joint_idx] = joint_range[0] + 0.5 * (joint_range[1] - joint_range[0])

            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
            current_fingertip_pos = physics.bind(fingertip_site).xpos.copy()
            physics.data.qpos[:] = original_qpos
            mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics

        print(f"Finger {finger}: Projection failed to converge")
        return current_qpos
    
    def _check_collision(self, physics, finger, qpos) -> bool:
        original_qpos = physics.data.qpos.copy()
        for i, joint_name in enumerate(self._finger_joints[finger]):
            if "J0" in joint_name:
                joint_idx = physics.model.name2id(joint_name, "tendon")
            else:
                joint_idx = physics.model.name2id(joint_name, "joint")
            physics.data.qpos[joint_idx] = qpos[i]
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics

        fingertip_site = self._hand.fingertip_sites[finger]
        fingertip_body = fingertip_site.parent
        fingertip_body_id = physics.model.name2id(f"rh_shadow_hand/{fingertip_body.name}", "body")

        collision_detected = False
        for i in range(physics.data.ncon):
            contact = physics.data.contact[i]
            body1 = physics.model.geom_bodyid[contact.geom1]
            body2 = physics.model.geom_bodyid[contact.geom2]
            if body1 == fingertip_body_id or body2 == fingertip_body_id:
                if "key" not in physics.model.id2name(contact.geom1, "geom") and \
                    "key" not in physics.model.id2name(contact.geom2, "geom"):
                    collision_detected = True
                    break

        physics.data.qpos[:] = original_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
        return collision_detected
    
    def _update_hand_position(self, physics):
        import time
        start_time = time.time()
        print(f"Updating hand position at timestep {self._t_idx}...")
        print(f"Current keys to play: {self._keys_current}")

        # Step 1: Compute target hand position based on current keys
        if self._keys_current:
            key_positions = []
            for key, _ in self._keys_current:
                key_site = self.piano.keys[key].site[0]
                key_pos = physics.bind(key_site).xpos.copy()
                key_positions.append(key_pos)
            if key_positions:
                # Average key positions to center the hand
                target_hand_pos = np.mean(key_positions, axis=0)
                target_hand_pos[2] += 0.05  # Keep hand above keys
                print(f"Target hand position: {target_hand_pos}")

                # Step 2: Adjust forearm/wrist position using IK
                forearm_site = self._hand.mjcf_model.find("site", "forearm_ty_site")
                full_forearm_site = f"rh_shadow_hand/{forearm_site.name}"
                # current_forearm_pos = physics.named.data.site_xpos[full_forearm_site]
                current_forearm_pos = physics.bind(forearm_site).xpos.copy()
                print(f"Current forearm position: {current_forearm_pos}")

                ik_result = qpos_from_site_pose(
                    physics,
                    full_forearm_site,
                    target_hand_pos,
                    None,
                    self._full_joints_one_array,
                    tol=1e-2,
                    max_steps=200,
                    regularization_threshold=0.01,
                    regularization_strength=0.1,
                )

                if ik_result.success:
                    print(f"Hand IK successful, new forearm pos: {target_hand_pos}")
                    for joint_name in self._wrist_joints + self._forearm_joints:
                        joint_idx = physics.model.name2id(joint_name, "joint")
                        physics.data.qpos[joint_idx] = ik_result.qpos[joint_idx]
                    mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
                else:
                    print("Hand IK failed, using current position")

        # Step 3: Handle thumb-under adjustment (optional refinement)
        thumb_assigned = any(finger == 0 for _, finger in self._keys_current)
        if self._thumb_under and thumb_assigned:
            print("Thumb-under detected: Fine-tuning wrist position...")
            forearm_site = self._hand.mjcf_model.find("site", "forearm_tx_site")
            full_forearm_site = f"rh_shadow_hand/{forearm_site.name}"
            # forearm_pos = physics.named.data.site_xpos[full_forearm_site]
            forearm_pos = physics.bind(forearm_site).xpos.copy()

            target_forearm_pos = forearm_pos.copy()
            target_forearm_pos[0] -= 0.05  # Small lateral shift for thumb-under
            target_forearm_pos[2] += 0.01

            ik_result = qpos_from_site_pose(
                physics,
                full_forearm_site,
                target_forearm_pos,
                None,
                self._full_joints_one_array[finger][:-2],
                tol=1e-2,
                max_steps=200,
                regularization_threshold=0.01,
                regularization_strength=0.1,
            )

            if ik_result.success:
                print(f"Thumb-under IK successful, new forearm pos: {target_forearm_pos}")
                for joint_name in self._wrist_joints + self._forearm_joints:
                    joint_idx = physics.model.name2id(joint_name, "joint")
                    physics.data.qpos[joint_idx] = ik_result.qpos[joint_idx]
                mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
            else:
                print("Thumb-under IK failed, skipping adjustment")

        # Step 4: Generate finger trajectories
        for key, mjcf_fingering in self._keys_current:
            if self._last_planned_keys[mjcf_fingering] != key or not self._trajectories[mjcf_fingering]:
                print(f"Planning for finger {mjcf_fingering}...")
                traj = self._plan_with_rrt(key, mjcf_fingering, physics)
                self._trajectories[mjcf_fingering] = traj
                self._traj_steps[mjcf_fingering] = 0
                self._last_planned_keys[mjcf_fingering] = key
                print(f"Trajectory length for finger {mjcf_fingering}: {len(traj)}")

        # Step 5: Compute desired velocities for each key press
        # Map key to its note velocity
        key_to_velocity = {}
        if self._t_idx < len(self._notes):
            for note, velocity in self._notes[self._t_idx]:
                key = note.pitch - 21
                key_to_velocity[key] = velocity
        print(f"Key to velocity mapping: {key_to_velocity}")

        # Step 6: Execute trajectories with velocity-based control
        action = np.zeros(len(self._hand.actuators) + 1)
        print(f"Total actuators: {len(self._hand.actuators)}")

        for finger in range(5):
            if self._trajectories[finger]:
                step = self._traj_steps[finger]
                if step < len(self._trajectories[finger]):
                    qpos_finger = self._trajectories[finger][step]
                    print(f"Finger {finger}, Step {step}: qpos_finger: {qpos_finger}")
                    if len(qpos_finger) != len(self._finger_joints[finger]):
                        raise ValueError(f"Invalid qpos length for finger {finger}: {len(qpos_finger)}")

                    # Find the key this finger is pressing
                    key = next((k for k, f in self._keys_current if f == finger), None)
                    if key is not None and key in key_to_velocity:
                        # Map MIDI velocity (0-127) to desired joint velocity (e.g., 0 to 1 rad/s)
                        midi_velocity = key_to_velocity[key]
                        desired_velocity = (midi_velocity / 127.0) * 1.0  # Scale to 0-1 rad/s
                        print(f"Finger {finger}: MIDI velocity={midi_velocity}, Desired joint velocity={desired_velocity}")
                    else:
                        desired_velocity = 0.5  # Default velocity if not specified
                        print(f"Finger {finger}: No velocity specified, using default={desired_velocity}")

                    for i, joint_name in enumerate(self._finger_joints[finger]):
                        if "J0" in joint_name:
                            joint_idx = physics.model.name2id(joint_name, "tendon")
                        else:
                            joint_idx = physics.model.name2id(joint_name, "joint")
                        current_qpos = physics.data.qpos[joint_idx]
                        current_qvel = physics.data.qvel[joint_idx]

                        # Position error for reaching the target
                        pos_error = qpos_finger[i] - current_qpos

                        # Velocity-based control: Adjust action to achieve desired velocity
                        # Use a PD controller with velocity term
                        kp = 100.0  # Position gain
                        kv = 10.0  # Velocity gain
                        vel_error = desired_velocity - current_qvel
                        control_effort = kp * pos_error + kv * vel_error

                        try:
                            if "J0" in joint_name:
                                action_idx = next(
                                    i for i, act in enumerate(self._hand.actuators)
                                    if hasattr(act, "tendon") and act.tendon and act.tendon.name in joint_name
                                )
                            else:
                                action_idx = next(
                                    i for i, act in enumerate(self._hand.actuators)
                                    if hasattr(act, "joint") and act.joint and act.joint.name in joint_name
                                )
                            action[action_idx] = control_effort
                            print(f"Finger {finger}, Joint {joint_name}, Pos error: {pos_error:.4f}, Vel error: {vel_error:.4f}, Action: {action[action_idx]:.4f}")
                        except StopIteration:
                            print(f"No actuator found for joint {joint_name}")
                            raise
                    self._traj_steps[finger] += 1
                else:
                    self._trajectories[finger] = []
                    self._traj_steps[finger] = 0
                    print(f"Finger {finger} trajectory completed")

        # Debug: Track fingertip positions and velocities
        for finger in range(5):
            site_name = self._hand.fingertip_sites[finger].name
            full_site_name = f"rh_shadow_hand/{site_name}" if self._hand_side == HandSide.RIGHT else f"lh_shadow_hand/{site_name}"
            # fingertip_pos = physics.named.data.site_xpos[full_site_name]
            fingertip_pos = physics.bind(self._hand.fingertip_sites[finger]).xpos.copy()
            # Compute the site's linear velocity using the Jacobian
            site_idx = physics.model.name2id(full_site_name, "site")
            nq = physics.model.nq
            jacp = np.zeros((3, nq))  # Translational Jacobian
            jacr = np.zeros((3, nq))  # Rotational Jacobian (not needed)
            mujoco.mj_jacSite(physics.model.ptr, physics.data.ptr, jacp, jacr, site_idx)
            qvel = physics.data.qvel  # Joint velocities
            fingertip_vel = jacp @ qvel  # Linear velocity (vx, vy, vz)
            assigned_key = next((key for key, f in self._keys_current if f == finger), None)
            if assigned_key is not None:
                key_site = self.piano.keys[assigned_key].site[0]
                key_pos = physics.bind(key_site).xpos.copy()
                distance = np.linalg.norm(fingertip_pos - key_pos)
                print(f"Finger {finger}: Fingertip pos: {fingertip_pos}, Vel: {fingertip_vel}, Key {assigned_key} pos: {key_pos}, Distance: {distance:.4f}")
            else:
                print(f"Finger {finger}: Not assigned, pos: {fingertip_pos}, Vel: {fingertip_vel}")

        action[-1] = self._sustain_state
        self._hand.apply_action(physics, action[:-1], random_state=None)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics after applying action
        self._last_action = action
        print(f"Hand position update took {time.time() - start_time:.2f} seconds")
        return action
    
    def _set_resting_position(self, physics) -> np.ndarray:
        """Set the hand to a resting position with fingers slightly bent and hand near key 38."""
        resting_qpos = np.zeros(physics.data.qpos.shape)

        for finger in range(5):
            for joint_name in self._full_finger_joints[finger]:
                joint_idx = physics.model.name2id(joint_name, "joint")
                if joint_idx == -1:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                    if joint_idx == -1:
                        raise ValueError(f"Joint {joint_name} not found in model")
                joint_range = physics.model.jnt_range[joint_idx]
                resting_angle = joint_range[0] + 0.1 * (joint_range[1] - joint_range[0])
                resting_qpos[joint_idx] = resting_angle
        
        physics.data.qpos[:] = resting_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
        return resting_qpos

    # Composer methods.

    def initialize_episode(self, physics, random_state) -> None:
        self._reset_quantities_at_episode_init()
        self._maybe_change_midi(random_state)
        self._resting_qpos = self._set_resting_position(physics)

    def _assign_fingers(self, notes) -> Tuple[List[Tuple[int, int]], bool]:
        if not notes or not isinstance(notes, list):
            print(f"Timestep {self._t_idx}: No valid notes provided, returning empty assignment")
            return ([], False)

        # Notes are now tuples of (note, velocity)
        try:
            keys = [note.pitch - 21 for note, _ in notes]
        except AttributeError:
            print(f"Timestep {self._t_idx}: Invalid note format in {notes}, returning empty assignment")
            return ([], False)

        keys.sort()
        print(f"Timestep {self._t_idx}: Keys to assign: {keys}")

        finger_assignments = []
        thumb_under = False

        if len(keys) > 1:
            for i, key in enumerate(keys):
                finger = i % 3
                finger_assignments.append((key, finger))
            self._last_finger = finger_assignments[-1][1]
            self._last_key = finger_assignments[-1][0]
        else:
            key = keys[0]
            if self._last_key is None:
                finger = 0
            else:
                key_distance = key - self._last_key
                if self._last_finger == 2 and key_distance == 1:
                    finger = 0
                    thumb_under = True
                elif abs(key_distance) <= 3:
                    finger = (self._last_finger + 1) % 3
                else:
                    finger = 2
            
            finger_assignments.append((key, finger))
            self._last_finger = finger
            self._last_key = key
        
        adjusted_assignments = []
        for key, finger in finger_assignments:
            if self._hand_side == HandSide.RIGHT:
                adjusted_assignments.append((key, finger))
            else:
                adjusted_assignments.append((88 - key, 4 - finger))
        print(adjusted_assignments)
        print(thumb_under)
        return (adjusted_assignments, thumb_under)

    def after_step(self, physics, random_state) -> None:
        import time
        start_time = time.time()
        print(f"After step {self._t_idx}")
        self._t_idx += 1
        self._should_terminate = (self._t_idx - 1) == len(self._notes) - 1

        if not self._disable_fingering_reward:
            self._update_fingering_state()
            self._keys_current = self._keys
            print(f"Keys to press at timestep {self._t_idx}: {self._keys_current}")
            if not self._disable_colorization:
                self._colorize_keys(physics)

        self._update_goal_state()
        self._goal_current = self._goal_state[0]
        print(f"Goal state: {self._goal_current}")
        print(f"Non-zero indices in goal state: {np.flatnonzero(self._goal_current)}")

        should_not_be_pressed = np.flatnonzero(1 - self._goal_current[:-1])
        self._failure_termination = self.piano.activation[should_not_be_pressed].any()

        action = self._update_hand_position(physics)
        print(f"action shape: {action.shape}")

        self._hand.apply_action(physics, action[:-1], random_state)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics
        self._last_action = action

        sustain = self._goal_current[-1]
        self.piano.apply_sustain(physics, sustain, random_state)

        print(f"After step took {time.time() - start_time:.2f} seconds")

    def get_reward(self, physics) -> float:
        return self._reward_fn.compute(physics)

    def get_discount(self, physics) -> float:
        del physics  # Unused.
        return self._discount

    def should_terminate_episode(self, physics) -> bool:
        del physics  # Unused.
        if self._should_terminate:
            return True
        if self._wrong_press_termination and self._failure_termination:
            self._discount = 0.0
            return True
        return False

    # Other.

    @property
    def midi(self) -> midi_file.MidiFile:
        return self._midi

    @property
    def reward_fn(self) -> composite_reward.CompositeReward:
        return self._reward_fn

    @property
    def task_observables(self):
        return self._task_observables

    def action_spec(self, physics):
        hand_spec = self._hand.action_spec(physics)
        sustain_spec = specs.BoundedArray(
            shape=(1,),
            dtype=hand_spec.dtype,
            minimum=[0.0],
            maximum=[1.0],
            name="sustain",
        )
        return specs.BoundedArray(
            shape=(hand_spec.shape[0] + 1,),
            dtype=np.float32,
            minimum=np.concatenate([hand_spec.minimum, sustain_spec.minimum]),
            maximum=np.concatenate([hand_spec.maximum, sustain_spec.maximum]),
            name="action"
        )
    
    def get_action_trajectory(self, physics):
        print("Starting get_action_trajectory...")
        self._reset_quantities_at_episode_init()
        self._reset_trajectory(self._midi)
        print(f"Length of self._notes: {len(self._notes)}")
        physics.data.qpos[:] = self._resting_qpos
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics

        trajectory = []
        for t in range(len(self._notes)):
            print(f"Processing timestep {t}...")
            self._t_idx = t
            self._update_fingering_state()
            self._keys_current = self._keys
            action = self._update_hand_position(physics)
            trajectory.append(action)
        print(f"Trajectory length: {len(trajectory)}")
        return trajectory

    def before_step(self, physics, action, random_state) -> None:
        sustain = action[-1]
        self.piano.apply_sustain(physics, sustain, random_state)
        self._hand.apply_action(physics, action[:-1], random_state)
        mujoco.mj_forward(physics.model.ptr, physics.data.ptr)  # Compute dynamics

    # Helper methods.

    def _compute_sustain_reward(self, physics) -> float:
        del physics  # Unused.
        return tolerance(
            self._goal_current[-1] - self.piano.sustain_activation[0],
            bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
            margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
            sigmoid="gaussian",
        )

    def _compute_energy_reward(self, physics) -> float:
        power = self._hand.observables.actuators_power(physics).copy()
        return -_ENERGY_PENALTY_COEF * np.sum(power)

    def _compute_key_press_reward(self, physics) -> float:
        del physics  # Unused.
        on = np.flatnonzero(self._goal_current[:-1])
        rew = 0.0
        if on.size > 0:
            actual = np.array(self.piano.state / self.piano._qpos_range[:, 1])
            rews = tolerance(
                self._goal_current[:-1][on] - actual[on],
                bounds=(0, _KEY_CLOSE_ENOUGH_TO_PRESSED),
                margin=(_KEY_CLOSE_ENOUGH_TO_PRESSED * 10),
                sigmoid="gaussian",
            )
            rew += 0.5 * rews.mean()
        off = np.flatnonzero(1 - self._goal_current[:-1])
        rew += 0.5 * (1 - float(self.piano.activation[off].any()))
        return rew

    def _compute_fingering_reward(self, physics) -> float:
        def _distance_finger_to_key(
            hand_keys: List[Tuple[int, int]], hand
        ) -> List[float]:
            distances = []
            for key, mjcf_fingering in hand_keys:
                fingertip_site = hand.fingertip_sites[mjcf_fingering]
                fingertip_pos = physics.bind(fingertip_site).xpos.copy()
                key_geom = self.piano.keys[key].geom[0]
                key_geom_pos = physics.bind(key_geom).xpos.copy()
                key_geom_pos[-1] += 0.5 * physics.bind(key_geom).size[2]
                key_geom_pos[0] += 0.35 * physics.bind(key_geom).size[0]
                diff = key_geom_pos - fingertip_pos
                distances.append(float(np.linalg.norm(diff)))
            return distances

        distances = _distance_finger_to_key(self._keys_current, self._hand)

        if not distances:
            return 0.0

        rews = tolerance(
            np.hstack(distances),
            bounds=(0, _FINGER_CLOSE_ENOUGH_TO_KEY),
            margin=(_FINGER_CLOSE_ENOUGH_TO_KEY * 10),
            sigmoid="gaussian",
        )
        return float(np.mean(rews))

    def _update_goal_state(self) -> None:
        if self._t_idx == len(self._notes):
            return

        self._goal_state = np.zeros(
            (self._n_steps_lookahead + 1, self.piano.n_keys + 1),
            dtype=np.float64,
        )
        t_start = self._t_idx
        t_end = min(t_start + self._n_steps_lookahead + 1, len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.pitch - 21 for note, _ in self._notes[t]]
            self._goal_state[i, keys] = 1.0

    def _update_fingering_state(self) -> None:
        if not hasattr(self, '_notes') or not self._notes or self._t_idx >= len(self._notes):
            print(f"Timestep {self._t_idx}: No notes or uninitialized, resetting state")
            self._keys = []
            self._fingering_state = np.zeros((5,), dtype=np.float64)
            self._thumb_under = False
            return
        
        notes = self._notes[self._t_idx]
        print(f"Timestep {self._t_idx}: Notes to play: {[(note.pitch, velocity) for note, velocity in notes]}")
        self._keys, self._thumb_under = self._assign_fingers(notes)

        self._fingering_state = np.zeros((5,), dtype=np.float64)
        for _, mjcf_fingering in self._keys:
            self._fingering_state[mjcf_fingering] = 1.0

    def _add_observables(self) -> None:
        enabled_observables = [
            "joints_pos",
            "position",
        ]
        for obs in enabled_observables:
            getattr(self._hand.observables, obs).enabled = True

        self.piano.observables.state.enabled = True
        self.piano.observables.sustain_state.enabled = True
        self.piano.observables.activation.enabled = False
        self.piano.observables.sustain_activation.enabled = False

        def _get_goal_state(physics) -> np.ndarray:
            del physics
            self._update_goal_state()
            return self._goal_state.ravel()

        goal_observable = observable.Generic(_get_goal_state)
        goal_observable.enabled = True
        self._task_observables = {"goal": goal_observable}

        def _get_fingering_state(physics) -> np.ndarray:
            del physics
            self._update_fingering_state()
            return self._fingering_state.ravel()

        fingering_observable = observable.Generic(_get_fingering_state)
        fingering_observable.enabled = not self._disable_fingering_reward
        self._task_observables["fingering"] = fingering_observable

        def _get_steps_left(physics) -> float:
            del physics
            return (len(self._notes) - self._t_idx) / len(self._notes)

        steps_left_observable = observable.Generic(_get_steps_left)
        steps_left_observable.enabled = False
        self._task_observables["steps_left"] = steps_left_observable

    def _colorize_fingertips(self) -> None:
        for i, name in enumerate(hand_consts.FINGERTIP_BODIES):
            color = hand_consts.FINGERTIP_COLORS[i] + (0.5,)
            body = self._hand.mjcf_model.find("body", name)
            for geom in body.find_all("geom"):
                if geom.dclass.dclass == "plastic_visual":
                    geom.rgba = color
            self._hand.fingertip_sites[i].rgba = color

    def _colorize_keys(self, physics) -> None:
        for key, mjcf_fingering in self._keys_current:
            key_geom = self.piano.keys[key].geom[0]
            fingertip_site = self._hand.fingertip_sites[mjcf_fingering]
            if not self.piano.activation[key]:
                physics.bind(key_geom).rgba = tuple(fingertip_site.rgba[:3]) + (1.0,)

    # def get_action(self, physics):
    #     return self._update_hand_position(physics)