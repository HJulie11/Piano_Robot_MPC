from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

# mujoco_menagerie rendering imports
from dataclasses import dataclass
from tqdm import tqdm
import enum

import numpy as np
from absl import app, flags
import mujoco
import dm_env

from dm_control import composer, mjcf
from dm_control.mjcf import export_with_assets
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils
import xml.etree.ElementTree as ET

import mido

from robopianist.music.midi_file import MidiFile
import robopianist
import robopianist.models.hands as shadow_hand
import robopianist.music as music
from robopianist.suite.tasks import piano_with_one_shadow_hand
# from robopianist.suite.tasks.piano_with_one_shadow_hand_with_dynamics import PianoWithOneShadowHand
import robopianist.suite.tasks.piano_with_one_shadow_hand_with_dynamics as piano_with_one_shadow_hand
from robopianist.models.piano import piano_constants as piano_consts
from robopianist import viewer, suite
from robopianist.wrappers.evaluation import MidiEvaluationWrapper

sys.path.append("/Users/shjulie/Desktop/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

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
    # notes = [60, 62, 64, 65]  # MIDI note numbers
    notes = [48, 50, 52, 53]
    time = 0
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=64, time=time))
        track.append(mido.Message("note_off", note=note, velocity=64, time=500))  # 500 ticks = 0.5 seconds
        time = 0  # Reset time for the next note_on

    mid.save(str(path))

def main(_) -> None:
    # Create a test MIDI file
    test_midi_path = Path("test_twinkle.mid")
    # create_simple_midi_file(test_midi_path)

    env = load(
        environment_name=_ENV_NAME.value,
        midi_file=test_midi_path,
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
            hand_side=shadow_hand.HandSide.RIGHT,
            initial_buffer_time=0.5,
        ),
    )
    wrapped_env = MidiEvaluationWrapper(env)

    task = env.task
    env.reset()
    print(env.task.midi)
    trajectory, all_dynamics_data = task.get_action_trajectory(env.physics)

    actions = trajectory
    print(f"Action sequence shape: {len(actions)}")
    print(f"Trajectory: {actions}")

    action_spec = wrapped_env.action_spec()
    zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    zeros[-1] = -1.0  # Disable sustain pedal.

    if _EXPORT.value:
        export_with_assets(
            env.task.root_entity.mjcf_model,
            out_dir="/tmp/robopianist/piano_with_one_shadow_hand",
            out_file_name="scene.xml",
        )
        mujoco_viewer.launch_from_path(
            "/tmp/robopianist/piano_with_one_shadow_hand/scene.xml"
        )
        return

    if _RECORD.value:
        wrapped_env = robopianist.wrappers.PianoSoundVideoWrapper(wrapped_env, record_every=1)
    if _CANONICALIZE.value:
        wrapped_env = CanonicalSpecWrapper(wrapped_env)

    class ActionSequencePlayer:
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            if _ACTION_SEQUENCE.value is not None:
                self._idx = 0
                self._actions = np.array(actions)
                print(f"Loaded action sequence with shape {self._actions.shape}")
            else:
                self._idx = 0
                self._actions = np.zeros(22)

        def __call__(self, timestep):
            if self._idx < len(self._actions):
                action = self._actions[self._idx]
                self._idx += 1
                print(f"Step {self._idx}: Action: {action}")
                print(f"action shape: {action.shape}")
                return action
            else:
                print(f"Step {self._idx}: Reached end of action sequence, repeating last action")
                return self._actions[-1]

    policy = ActionSequencePlayer()

    print("Running policy ...")
    timestep = wrapped_env.reset()
    step_count = 0

    # List to store the fingertip trajectories
    thumb_trajectory = []
    index_trajectory = []
    middle_trajectory = []

    if _HEADLESS.value:
        print("Running headless ...")
        viewer.launch(wrapped_env, policy=policy)
    else:
        print("Running with viewer ...")
        with mujoco_viewer.launch_passive(
            wrapped_env.physics.model.ptr, wrapped_env.physics.data.ptr
        ) as mujoco_viewer_handle:
            while mujoco_viewer_handle.is_running() and not timestep.last():
                action = policy(timestep)
                try:
                    timestep = wrapped_env.step(action)
                    step_count += 1
                    mujoco_viewer_handle.sync()

                    # extract fingertip positions
                    thumb_pos = env.physics.bind(env.task._hand.fingertip_sites[0]).xpos.copy()
                    index_pos = env.physics.bind(env.task._hand.fingertip_sites[1]).xpos.copy()
                    middle_pos = env.physics.bind(env.task._hand.fingertip_sites[2]).xpos.copy()
                    # thumb_pos = env.physics.named.data.site_xpos["rh_shadow_hand/thdistal_site"].copy()
                    # index_pos = env.physics.named.data.site_xpos["rh_shadow_hand/ffdistal_site"].copy()
                    # middle_pos = env.physics.named.data.site_xpos["rh_shadow_hand/ffdistal_site"].copy()

                    # Append to trajectory lists
                    thumb_trajectory.append(thumb_pos)
                    index_trajectory.append(index_pos)
                    middle_trajectory.append(middle_pos)

                    print(f"Step {step_count}: Reward: {timestep.reward}")
                    time.sleep(_CONTROL_TIMESTEP.value)
                except Exception as e:
                    print(f"Error during step: {e}")
                    break
            print(f"Episode completed in {step_count} steps or viewer closed")

    # Convert to numpy arrays for easier plotting
    thumb_trajectory = np.array(thumb_trajectory)
    index_trajectory = np.array(index_trajectory)
    middle_trajectory = np.array(middle_trajectory)

    np.save("results/trajectories/thumb_trajectory.npy", thumb_trajectory)
    np.save("results/trajectories/index_trajectory.npy", index_trajectory)
    np.save("results/trajectories/middle_trajectory.npy", middle_trajectory)

    wrapped_env.save_timestep_rewards(_TIMESTEP_REWARDS_PATH.value)
    wrapped_env.save_timestep_energy_rewards(_TIMESTEP_ENERGY_REWARDS_PATH.value)
    wrapped_env.save_timestep_finger_movement_rewards(_TIMESTEP_FINGER_MOVEMENT_REWARDS_PATH.value)
    wrapped_env.save_timestep_key_press_rewards(_TIMESTEP_KEY_PRESS_REWARDS_PATH.value)

    # import matplotlib.pyplot as plt
    # plt.plot(all_dynamics_data["energy"])
    # plt.xlabel("Timestep")
    # plt.ylabel("Energy")
    # plt.title("Energy Over Time")
    # plt.grid()
    # plt.show()

if __name__ == "__main__":
    app.run(main)