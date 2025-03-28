# # Copyright 2023 The RoboPianist Authors.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

"""Piano with shadow hands environment."""
from pathlib import Path
import dm_env
import numpy as np
from typing import Any, Mapping, Optional, Union, Dict
from absl import app, flags

from dm_control import composer
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils

from robopianist import suite, viewer
from robopianist.wrappers import PianoSoundVideoWrapper
import robopianist.music as music
from robopianist.suite.tasks import piano_with_one_shadow_hand
import robopianist.models.hands as shadow_hand
from robopianist.music import midi_file
from robopianist.models.hands import HandSide

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

# for load function:
# RoboPianist-repertoire-150.
_BASE_REPERTOIRE_NAME = "RoboPianist-repertoire-150-{}-v0"
REPERTOIRE_150 = [_BASE_REPERTOIRE_NAME.format(name) for name in music.PIG_MIDIS]
_REPERTOIRE_150_DICT = dict(zip(REPERTOIRE_150, music.PIG_MIDIS))

# RoboPianist-etude-12.
_BASE_ETUDE_NAME = "RoboPianist-etude-12-{}-v0"
ETUDE_12 = [_BASE_ETUDE_NAME.format(name) for name in music.ETUDE_MIDIS]
_ETUDE_12_DICT = dict(zip(ETUDE_12, music.ETUDE_MIDIS))

# RoboPianist-debug.
_DEBUG_BASE_NAME = "RoboPianist-debug-{}-v0"
DEBUG = [_DEBUG_BASE_NAME.format(name) for name in music.DEBUG_MIDIS]
_DEBUG_DICT = dict(zip(DEBUG, music.DEBUG_MIDIS))

# All valid environment names.
ALL = REPERTOIRE_150 + ETUDE_12 + DEBUG
_ALL_DICT: Dict[str, Union[Path, str]] = {
    **_REPERTOIRE_150_DICT,
    **_ETUDE_12_DICT,
    **_DEBUG_DICT,
}

midi = midi_file.MidiFile.from_file("do-re-mi.mid")
task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
    midi=midi,
    hand_side=HandSide.RIGHT,
    n_steps_lookahead=1,
    trim_silence=True,
    wrong_press_termination=False,
    initial_buffer_time=0.5,
    disable_fingering_reward=False,
    disable_colorization=True,
)

env = composer_utils.Environment(
    task = task,
)

# if _RECORD.value:
#     env = PianoSoundVideoWrapper(env, record_every=1)
# if _CANONICALIZE.value:
#     env = CanonicalSpecWrapper(env)

action_spec = env.action_spec()
zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
zeros[-1] = -1.0  # Disable sustain pedal.
print(f"Action dimension: {action_spec.shape}")

# Sanity check observables.
timestep = env.reset()
dim = 0
for k, v in timestep.observation.items():
    print(f"\t{k}: {v.shape} {v.dtype}")
    dim += int(np.prod(v.shape))
print(f"Observation dimension: {dim}")

# class Policy:
#     def __init__(self) -> None:
#         self.reset()

#     def reset(self) -> None:
#         self._idx = 0
#         hand_spec = task._hand.action_spec(env.physics)
#         sustain_spec = np.array([0.0])
#         dummy_action = np.zeros(hand_spec.shape, dtype=hand_spec.dtype)
#         # self._actions = np.load(_ACTION_SEQUENCE.value)
#         self._actions = np.concatenate([dummy_action, sustain_spec])

#     def __call__(self, timestep: dm_env.TimeStep) -> np.ndarray:
#         del timestep  # Unused.
#         actions = self._actions[self._idx]
#         self._idx += 1
#         return actions

# policy = Policy()

viewer.launch(env)
