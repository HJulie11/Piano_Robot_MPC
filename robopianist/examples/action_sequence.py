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

"""Piano with shadow hands environment."""
from pathlib import Path
import dm_env
import numpy as np
from typing import Any, Mapping, Optional, Union, Dict
from absl import app, flags
import time

from dm_control import composer
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils

from robopianist import suite, viewer
import robopianist.music as music
import robopianist.models.hands.shadow_hand as shadow_hand
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from robopianist.music import midi_file
from robopianist.models.hands.base import HandSide

_ENV_NAME = flags.DEFINE_string(
    "env_name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0", ""
)
_MIDI_FILE = flags.DEFINE_string("midi_file", None, "")
_CONTROL_TIMESTEP = flags.DEFINE_float("control_timestep", 0.05, "")
_STRETCH = flags.DEFINE_float("stretch", 1.0, "")
_SHIFT = flags.DEFINE_integer("shift", 0, "")
_RECORD = flags.DEFINE_bool("record", False, "")
_EXPORT = flags.DEFINE_bool("export", False, "")
_GRAVITY_COMPENSATION = flags.DEFINE_bool("gravity_compensation", False, "")
_HEADLESS = flags.DEFINE_bool("headless", False, "")
_TRIM_SILENCE = flags.DEFINE_bool("trim_silence", False, "")
_PRIMITIVE_FINGERTIP_COLLISIONS = flags.DEFINE_bool(
    "primitive_fingertip_collisions", False, ""
)
_REDUCED_ACTION_SPACE = flags.DEFINE_bool("reduced_action_space", False, "")
_DISABLE_FINGERING_REWARD = flags.DEFINE_bool("disable_fingering_reward", False, "")
_DISABLE_COLORIZATION = flags.DEFINE_bool("disable_colorization", True, "")
_CANONICALIZE = flags.DEFINE_bool("canonicalize", False, "")
_N_STEPS_LOOKAHEAD = flags.DEFINE_integer("n_steps_lookahead", 1, "")
_ATTACHMENT_YAW = flags.DEFINE_float("attachment_yaw", 0.0, "")
_ACTION_SEQUENCE = flags.DEFINE_string(
    "action_sequence",
    None,
    "Path to an npy file containing a sequence of actions to replay.",
)
_N_SECONDS_LOOKAHEAD = flags.DEFINE_integer("n_seconds_lookahead", None, "")
_WRONG_PRESS_TERMINATION = flags.DEFINE_bool("wrong_press_termination", False, "")

# For load function:
_BASE_REPERTOIRE_NAME = "RoboPianist-repertoire-150-{}-v0"
REPERTOIRE_150 = [_BASE_REPERTOIRE_NAME.format(name) for name in music.PIG_MIDIS]
_REPERTOIRE_150_DICT = dict(zip(REPERTOIRE_150, music.PIG_MIDIS))

_BASE_ETUDE_NAME = "RoboPianist-etude-12-{}-v0"
ETUDE_12 = [_BASE_ETUDE_NAME.format(name) for name in music.ETUDE_MIDIS]
_ETUDE_12_DICT = dict(zip(ETUDE_12, music.ETUDE_MIDIS))

_DEBUG_BASE_NAME = "RoboPianist-debug-{}-v0"
DEBUG = [_DEBUG_BASE_NAME.format(name) for name in music.DEBUG_MIDIS]
_DEBUG_DICT = dict(zip(DEBUG, music.DEBUG_MIDIS))

ALL = REPERTOIRE_150 + ETUDE_12 + DEBUG
_ALL_DICT: Dict[str, Union[Path, str]] = {
    **_REPERTOIRE_150_DICT,
    **_ETUDE_12_DICT,
    **_DEBUG_DICT,
}

def load(
    environment_name: str,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
) -> composer.Environment:
    import mido

    # Create a simple MIDI file with "Do-Re-Mi" (C-D-E)
    midi = mido.MidiFile()
    track = mido.MidiTrack()
    midi.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=500000))  # 120 BPM

    tick = 0
    for note in [48, 50, 52]:
        track.append(mido.Message('note_on', note=note, velocity=64, time=tick))
        tick = 500
        track.append(mido.Message('note_off', note=note, velocity=64, time=tick))
        tick = 0

    midi.save("do-re-mi-test.mid")
    midi = midi_file.MidiFile.from_file("do-re-mi-test.mid")

    task_kwargs = task_kwargs or {}
    task_kwargs["midi"] = midi  # Explicitly pass the MIDI file to the task

    # Create the task and environment
    task = PianoWithOneShadowHand(**task_kwargs)
    env = composer_utils.Environment(
        task=task,
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
    )

    # Debug: Print the MIDI note trajectory
    print(f"Control timestep: {task.control_timestep} seconds")
    print(f"Initial buffer time: {task._initial_buffer_time} seconds")
    print(f"Total timesteps in MIDI: {len(task._notes)}")
    for t, notes in enumerate(task._notes):
        if notes:
            print(f"Timestep {t}: {notes}")

    return env

def main(_) -> None:
    env = load(
        environment_name=_ENV_NAME.value,
        shift=_SHIFT.value,
        task_kwargs=dict(
            change_color_on_activation=True,
            trim_silence=_TRIM_SILENCE.value,
            control_timestep=_CONTROL_TIMESTEP.value,
            gravity_compensation=_GRAVITY_COMPENSATION.value,
            primitive_fingertip_collisions=_PRIMITIVE_FINGERTIP_COLLISIONS.value,
            reduced_action_space=_REDUCED_ACTION_SPACE.value,
            n_steps_lookahead=_N_STEPS_LOOKAHEAD.value,
            n_seconds_lookahead=_N_SECONDS_LOOKAHEAD.value,
            wrong_press_termination=_WRONG_PRESS_TERMINATION.value,
            disable_fingering_reward=_DISABLE_FINGERING_REWARD.value,
            disable_colorization=_DISABLE_COLORIZATION.value,
            attachment_yaw=_ATTACHMENT_YAW.value,
            hand_side=HandSide.RIGHT,
            initial_buffer_time=0.5,  # Add buffer time to allow hand to move
        ),
    )

    # action_spec = env.action_spec()
    # zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    # zeros[-1] = -1.0  # Disable sustain pedal.
    # print(f"Action dimension: {action_spec.shape}")

    # # Sanity check observables.
    # timestep = env.reset()
    # dim = 0
    # for k, v in timestep.observation.items():
    #     print(f"\t{k}: {v.shape} {v.dtype}")
    #     dim += int(np.prod(v.shape))
    # print(f"Observation dimension: {dim}")

    # # Manual stepping loop for debugging
    # print("Starting manual simulation loop...")
    # task = env.task
    # actions_list = []
    # for step in range(500):  # Run for 500 steps to cover the MIDI sequence
    #     action = np.zeros(env.action_spec().shape)  # Dummy action
    #     timestep = env.step(action)

    #     # Extract the action from the task
    #     action = task._last_action
    #     print(f"Action: {action}")
    #     actions_list.append(action.copy())

    #     # Print timestep information
    #     print(f"\nStep {step}:")
    #     print(f"  Step type: {timestep.step_type}")
    #     print(f"  Reward: {timestep.reward}")
    #     print(f"  Discount: {timestep.discount}")
    #     print(f"  Should terminate: {task.should_terminate_episode(env.physics)}")
    #     print(f"  Action: {action}")

    #     # Print piano key activations
    #     key_activations = task.piano.activation
    #     active_keys = np.flatnonzero(key_activations)
    #     print(f"  Active keys: {active_keys}")

    #     # Print fingertip distances for assigned fingers
    #     for finger in range(5):
    #         site_name = task._hand.fingertip_sites[finger].name
    #         if task._hand_side == HandSide.LEFT:
    #             full_site_name = f"lh_shadow_hand/{site_name}"
    #         else:
    #             full_site_name = f"rh_shadow_hand/{site_name}"
    #         fingertip_pos = env.physics.named.data.site_xpos[full_site_name]

    #         assigned_key = None
    #         for key, mjcf_fingering in task._keys_current:
    #             if mjcf_fingering == finger:
    #                 assigned_key = key
    #                 break

    #         if assigned_key is not None:
    #             key_site = task.piano.keys[assigned_key].site[0]
    #             key_pos = env.physics.bind(key_site).xpos.copy()
    #             distance = np.linalg.norm(fingertip_pos - key_pos)
    #             print(f"  Finger {finger} distance to key {assigned_key}: {distance:.4f} m")

    #     # Add a small delay to make logs readable
    #     time.sleep(0.1)

    # # Optionally launch the viewer after manual stepping

    # actions_array = np.array(actions_list)
    # np.save("action_sequence.npy", actions_array)
    print("Launching viewer...")
    viewer.launch(env)

if __name__ == "__main__":
    app.run(main)