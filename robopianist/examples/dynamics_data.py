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

"""Example script for running the piano with one shadow hand environment."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from absl import app
from absl import flags
from dm_control import composer
from PIL import Image

import mido

from robopianist import music
# from robopianist.models.hands import HandSide
import robopianist.suite.tasks.piano_with_one_shadow_hand as piano_with_one_shadow_hand
import robopianist.models.hands.shadow_hand as shadow_hand
from robopianist.models.hands.base import HandSide
import pickle

from mujoco_utils import composer_utils


# MIDI file to play.
_MIDI = flags.DEFINE_string(
    "midi",
    str(Path(__file__).parent / "midi" / "twinkle_twinkle.mid"),
    "Path to the MIDI file to play.",
)

# Whether to disable visualization.
_NO_VIZ = flags.DEFINE_boolean(
    "no_viz",
    False,
    "If True, disables visualization.",
)

# Number of timesteps to run the simulation for.
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps",
    1000,
    "Number of timesteps to run the simulation for.",
)

# Control timestep in seconds.
# _CONTROL_TIMESTEP = flags.DEFINE_float(
#     "control_timestep",
#     0.05,
#     "Control timestep in seconds.",
# )

# Whether to trim silence from the MIDI file.
# _TRIM_SILENCE = flags.DEFINE_boolean(
#     "trim_silence",
#     True,
#     "If True, trims silence from the MIDI file.",
# )

# Initial buffer time in seconds.
_INITIAL_BUFFER_TIME = flags.DEFINE_float(
    "initial_buffer_time",
    0.5,
    "Initial buffer time in seconds.",
)

# Whether to disable the fingering reward.
# _DISABLE_FINGERING_REWARD = flags.DEFINE_boolean(
#     "disable_fingering_reward",
#     False,
#     "If True, disables the fingering reward.",
# )

# Whether to disable colorization.
# _DISABLE_COLORIZATION = flags.DEFINE_boolean(
#     "disable_colorization",
#     False,
#     "If True, disables colorization.",
# )

# Whether to terminate the episode on wrong key presses.
# _WRONG_PRESS_TERMINATION = flags.DEFINE_boolean(
#     "wrong_press_termination",
#     False,
#     "If True, terminates the episode on wrong key presses.",
# )

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
    "action_sequence.npy",  # Path to the saved action sequence
    "Path to an npy file containing a sequence of actions to replay.",
)
_N_SECONDS_LOOKAHEAD = flags.DEFINE_integer("n_seconds_lookahead", None, "")
_WRONG_PRESS_TERMINATION = flags.DEFINE_bool("wrong_press_termination", False, "")
_TIMESTEP_REWARDS_PATH = flags.DEFINE_string(
    "timestep_rewards_path", "results/timestep_rewards.csv", "Path to save the timestep rewards."
)
_TIMESTEP_KEY_PRESS_REWARDS_PATH = flags.DEFINE_string(
    "timestep_key_press_rewards_path", "results/timestep_key_press_rewards.csv", "Path to save the timestep key press rewards."
)

_TIMESTEP_ENERGY_REWARDS_PATH = flags.DEFINE_string(
    "timestep_energy_rewards_path", "results/timestep_energy_rewards.csv", "Path to save the timestep energy rewards."
)

_TIMESTEP_FINGER_MOVEMENT_REWARDS_PATH = flags.DEFINE_string(
    "timestep_finger_movement_rewards_path", "results/timestep_finger_movement_rewards.csv", "Path to save the timestep finger movement rewards."
)

# for load function:
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
    midi_file: Optional[Path] = None,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
) -> composer.Environment:
    if midi_file is not None:
        midi = music.load(midi_file, stretch=stretch, shift=shift)
    else:
        if environment_name not in ALL:
            raise ValueError(
                f"Unknown environment {environment_name}. "
                f"Available environments: {ALL}"
            )
        midi = music.load(_ALL_DICT[environment_name], stretch=stretch, shift=shift)

    task_kwargs = task_kwargs or {}

    return composer_utils.Environment(
        task=piano_with_one_shadow_hand.PianoWithOneShadowHand(midi=midi, **task_kwargs),
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
    )

def create_simple_midi_file(path: Path) -> None:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Add notes for "Twinkle, Twinkle, Little Star" (C4, C4, G4, G4, A4, A4, G4)
    # notes = [60, 60, 67, 67, 69, 69, 67]  # MIDI note numbers
    notes = [60, 62, 64, 65]  # MIDI note numbers
    # notes = [48, 50, 52, 53]
    time = 0
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=64, time=time))
        track.append(mido.Message("note_off", note=note, velocity=64, time=500))  # 500 ticks = 0.5 seconds
        time = 0  # Reset time for the next note_on

    mid.save(str(path))


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load the MIDI file.
    # midi = music.load(_MIDI.value)
    test_midi_path = Path("test_twinkle.mid")
    create_simple_midi_file(test_midi_path)

    # Create the task.
    # task = piano_with_one_shadow_hand.PianoWithOneShadowHand(
    #     midi=test_midi_path,
    #     hand_side=HandSide.RIGHT,  # Use the right hand.
    #     control_timestep=_CONTROL_TIMESTEP.value,
    #     trim_silence=_TRIM_SILENCE.value,
    #     initial_buffer_time=_INITIAL_BUFFER_TIME.value,
    #     disable_fingering_reward=_DISABLE_FINGERING_REWARD.value,
    #     disable_colorization=_DISABLE_COLORIZATION.value,
    #     wrong_press_termination=_WRONG_PRESS_TERMINATION.value,
    # )

    # # Create the environment.
    # env = composer.Environment(
    #     task=task,
    #     time_limit=task.midi_duration + _INITIAL_BUFFER_TIME.value,
    #     random_state=np.random.RandomState(1234),
    # )

    env = load(
        environment_name=_ENV_NAME.value,
        midi_file=test_midi_path,
        shift=_SHIFT.value,
        # time_limit=task.midi_duration + _INITIAL_BUFFER_TIME.value,
        # random_state=np.random.RandomState(1234),
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
            initial_buffer_time=0.5,
        ),
    )

    # Reset the environment.
    env.reset()
    task = env.task


    # Compute the number of timesteps to run the simulation for.
    num_timesteps = min(
        _NUM_TIMESTEPS.value,
        int((task.midi.seq.notes[-1].end_time + _INITIAL_BUFFER_TIME.value) / _CONTROL_TIMESTEP.value),
    )
    print(f"Running for {num_timesteps} timesteps")

    # Create a temporary directory to store the rendered frames.
    temp_dir = tempfile.mkdtemp()
    print(f"Saving frames to {temp_dir}")

    # Collect dynamics data over the episode.
    all_dynamics_data = {
        "fingertip_velocities": [],
        "joint_accelerations": [],
        "contact_forces": [],
        "energy": [],
    }

    # Run the simulation.
    if not _NO_VIZ.value:
        for t in range(num_timesteps):
            # Get the action and dynamics data from the task.
            action, dynamics_data = task.get_action_trajectory(env.physics)
            env.step(action)

            # Collect dynamics data.
            all_dynamics_data["fingertip_velocities"].append(dynamics_data["fingertip_velocities"])
            all_dynamics_data["joint_accelerations"].append(dynamics_data["joint_accelerations"])
            all_dynamics_data["contact_forces"].append(dynamics_data["contact_forces"])
            all_dynamics_data["energy"].append(dynamics_data["energy"])

            # Render and save the frame.
            if t % 10 == 0:
                print(f"Timestep {t}/{num_timesteps}")
            pixels = env.physics.render(height=480, width=640, camera_id=0)
            image = Image.fromarray(pixels)
            image.save(os.path.join(temp_dir, f"frame_{t:04d}.png"))
    else:
        for t in range(num_timesteps):
            # Get the action and dynamics data from the task.
            action, dynamics_data = task.get_action_trajectory(env.physics)
            env.step(action)

            # Collect dynamics data.
            all_dynamics_data["fingertip_velocities"].append(dynamics_data["fingertip_velocities"])
            all_dynamics_data["joint_accelerations"].append(dynamics_data["joint_accelerations"])
            all_dynamics_data["contact_forces"].append(dynamics_data["contact_forces"])
            all_dynamics_data["energy"].append(dynamics_data["energy"])

            if t % 10 == 0:
                print(f"Timestep {t}/{num_timesteps}")

    # Save the dynamics data to a file.
    with open("dynamics_data.pkl", "wb") as f:
        pickle.dump(all_dynamics_data, f)
    print(f"Saved dynamics data to dynamics_data.pkl")

    # Optionally, create a video from the rendered frames.
    if not _NO_VIZ.value:
        import subprocess

        video_path = os.path.join(temp_dir, "video.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-r",
                "30",
                "-i",
                os.path.join(temp_dir, "frame_%04d.png"),
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                video_path,
            ],
            check=True,
        )
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    app.run(main)