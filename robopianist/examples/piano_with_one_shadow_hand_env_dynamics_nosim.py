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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
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
# from robopianist.suite.tasks import piano_with_one_shadow_hand
# from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
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
    notes = [60, 62, 64, 65]  # MIDI note numbers
    # notes = [48, 50, 52, 53]
    time = 0
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=64, time=time))
        track.append(mido.Message("note_off", note=note, velocity=64, time=500))  # 500 ticks = 0.5 seconds
        time = 0  # Reset time for the next note_on

    mid.save(str(path))

def main(_) -> None:
    # Create a test MIDI file
    test_midi_path = Path("test_twinkle.mid")
    create_simple_midi_file(test_midi_path)

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
    # Run 1 trial for trajectory plotting (can extend to multiple trials)
    num_trials = 1  # Set to 1 for simplicity; adjust as needed
    collision_rates = []
    execution_times = []
    success_rates = []

    # Lists to store fingertip trajectories for thumb (0), index (1), and middle (2)
    fingers_to_plot = [0, 1, 2]  # Thumb, index, middle
    fingertip_trajectories = {finger: {'x': [], 'y': [], 'z': []} for finger in fingers_to_plot}

    for trial in range(num_trials):
        print(f"Starting trial {trial + 1}/{num_trials}")
        env.reset()
        trajectory, _ = env.task.get_action_trajectory(env.physics)  # MP Only (no dynamics or PD)
        total_steps = len(trajectory)
        collisions = 0
        successes = 0
        start_time = env.physics.data.time

        for t in range(total_steps):
            action = trajectory[t]
            print(f"Timestep {t}: Action = {action}")
            env.physics.set_control(action[:-1])
            env.physics.step()
            mujoco.mj_forward(env.physics.model.ptr, env.physics.data.ptr)

            # Debug: Log the state of _keys_current and _active_notes
            # print(f"Timestep {t}: _keys_current = {env.task._keys_current}, _active_notes = {env.task._active_notes}")

            # Record fingertip positions for thumb, index, and middle fingers
            for finger in fingers_to_plot:
                fingertip_site = env.task._hand.fingertip_sites[finger]
                full_site_name = f"rh_shadow_hand/{fingertip_site.name}"
                fingertip_pos = env.physics.named.data.site_xpos[full_site_name].copy()
                fingertip_trajectories[finger]['x'].append(fingertip_pos[0])  # x-coordinate
                fingertip_trajectories[finger]['y'].append(fingertip_pos[1])  # y-coordinate
                fingertip_trajectories[finger]['z'].append(fingertip_pos[2])  # z-coordinate

            # Check collisions
            for finger1 in range(5):
                for finger2 in range(finger1 + 1, 5):
                    if env.task._check_collision(env.physics, finger1, env.physics.data.qpos) or \
                    env.task._check_collision(env.physics, finger2, env.physics.data.qpos):
                        collisions += 1

            # Check success for assigned keys
            if env.task._keys_current:  # Only proceed if there are active keys
                for key, finger in env.task._keys_current:
                    fingertip_site = env.task._hand.fingertip_sites[finger]
                    full_site_name = f"rh_shadow_hand/{fingertip_site.name}"
                    site_idx = env.physics.model.name2id(full_site_name, "site")
                    contact_force = env.physics.data.cfrc_ext[site_idx][2]  # z-component
                    if contact_force > 0.5:
                        successes += 1
            else:
                print(f"Timestep {t}: No active keys to evaluate")

        # Compute metrics
        collision_rates.append(collisions / total_steps * 100 if total_steps > 0 else 0.0)
        execution_times.append(env.physics.data.time - start_time)

        # Compute success rate, handle empty _keys_current
        if env.task._keys_current:
            success_rate = successes / len(env.task._keys_current) * 100
        else:
            success_rate = 0.0  # Default to 0% if no keys were active
            print(f"Trial {trial + 1}: No active keys at the end of the trial, setting success rate to 0%")
        success_rates.append(success_rate)

    # Plot fingertip trajectories in 3D for thumb, index, and middle fingers
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle("3D Fingertip Trajectories (Thumb, Index, Middle Fingers)")

    # Labels for fingers
    finger_labels = {0: "Thumb", 1: "Index", 2: "Middle"}

    # Use Matplotlib's tab10 palette for colorblind-friendly colors
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Get the first 10 colors from tab10
    finger_colors = {0: colors[0], 1: colors[1], 2: colors[2]}  # Assign to thumb, index, middle

    # Define different line styles for additional distinction
    line_styles = {0: '-', 1: '-', 2: '-'}  # Solid, dashed, dotted

    # Plot trajectories
    for finger in fingers_to_plot:
        x = fingertip_trajectories[finger]['x']
        y = fingertip_trajectories[finger]['y']
        z = fingertip_trajectories[finger]['z']
        ax.plot(x, y, z, label=finger_labels[finger], color=finger_colors[finger],
                linestyle=line_styles[finger], linewidth=2)
        # Add scatter points for start and end
        ax.scatter(x[0], y[0], z[0], color=finger_colors[finger], marker='o', s=50)  # Start point
        ax.scatter(x[-1], y[-1], z[-1], color=finger_colors[finger], marker='x', s=50)  # End point

    # Set labels and title
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.legend()
    ax.grid(True)

    # Set axis limits to focus on the relevant region
    ax.set_xlim([min([min(fingertip_trajectories[finger]['x']) for finger in fingers_to_plot]) - 0.01,
                max([max(fingertip_trajectories[finger]['x']) for finger in fingers_to_plot]) + 0.01])
    ax.set_ylim([min([min(fingertip_trajectories[finger]['y']) for finger in fingers_to_plot]) - 0.01,
                max([max(fingertip_trajectories[finger]['y']) for finger in fingers_to_plot]) + 0.01])
    ax.set_zlim([min([min(fingertip_trajectories[finger]['z']) for finger in fingers_to_plot]) - 0.01,
                max([max(fingertip_trajectories[finger]['z']) for finger in fingers_to_plot]) + 0.01])

    plt.savefig("results/trajectories/fingertip_trajectories_dynamics_predefined.png")
    plt.show()

    # Plot remaining metrics (collision rate, execution time, success rate)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("MP Only Experiment Results (Other Metrics)")

    # Collision Rate
    axes[0].bar(range(num_trials), collision_rates)
    axes[0].set_title("Collision Rate (%)")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Collision Rate (%)")

    # Execution Time
    axes[1].bar(range(num_trials), execution_times)
    axes[1].set_title("Execution Time (s)")
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Time (s)")

    # Success Rate
    axes[2].bar(range(num_trials), success_rates)
    axes[2].set_title("Success Rate (%)")
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Success Rate (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("mp_only_other_metrics.png")
    plt.show()

if __name__ == "__main__":
    app.run(main)