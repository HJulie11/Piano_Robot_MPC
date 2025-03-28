from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

# mujoco_menagerie rendering imports
from dataclasses import dataclass
# import mediapy as media
from tqdm import tqdm
import enum
# mujoco, numpy, pathlib already imported.

from robopianist.wrappers.sound import PianoSoundVideoWrapper
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

# from models.base import HandSide
import numpy as np
from absl import app, flags
import mujoco
import dm_env

from dm_control import composer, mjcf
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer
from mujoco_utils import composer_utils
import xml.etree.ElementTree as ET

import robopianist
import robopianist.models.hands as shadow_hand
import robopianist.music as music
from robopianist.suite.tasks import piano_with_one_shadow_hand
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from robopianist.models.piano import piano_constants as piano_consts
from robopianist import viewer as robopianist_viewer
import mink

import kinematics.rrt_planner_2 as rrt
import kinematics.get_config as get_config
import parse_pose
import pinocchio as pin
from plot_traj import plot_traj
import kinematics.ik_2 as ik_2
from util import create_joint_limits_dict, add_mocap, construct_model
from loop_rate_limiters import RateLimiter
import traj_plan

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
    """Loads a RoboPianist environment.

    Args:
        environment_name: Name of the environment to load. Must be of the form
            "RoboPianist-repertoire-150-<name>-v0", where <name> is the name of a
            PIG dataset MIDI file in camel case notation.
        midi_file: Path to a MIDI file to load. If provided, this will override
            `environment_name`.
        seed: Optional random seed.
        stretch: Stretch factor for the MIDI file.
        shift: Shift factor for the MIDI file.
        recompile_physics: Whether to recompile the physics.
        legacy_step: Whether to use the legacy step function.
        task_kwargs: Additional keyword arguments to pass to the task.
    """
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

def main(_) -> None:
    env = load(
        environment_name=_ENV_NAME.value,
        midi_file=_MIDI_FILE.value,
        shift=_SHIFT.value,
        task_kwargs=dict(
            # midi=music.load("TwinkleTwinkleRousseau"),
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
            hand_side = shadow_hand.HandSide.RIGHT,
            initial_buffer_time = 0.0,
        ),
    )
    
    planner = traj_plan.OfflineTrajectoryPlanner(env, _CONTROL_TIMESTEP.value)
    
    # precompute offline trajectory
    print("Computing offline trajectory...")
    trajectory = planner.plan()
    # plot_traj(trajectory)
    print(f"Trajectory length: {len(trajectory)} steps")
    action_sequence = planner.plan_qpos_hand(trajectory)

    np.save("trajectory_action.npy", action_sequence)

    

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
    
    actions_sim = action_sequence
        
    class ActionSequencePlayer:
        """Applies a given sequence of actions at each timestep."""

        def __init__(self) -> None:
            self.reset()
        
        def reset(self) -> None:
            """
            Args:
                env: The simulation environment.
                action_sequence: A sequence of actions to apply.
            """

            if _ACTION_SEQUENCE.value is not None:
                self._idx = 0
                self._actions = np.load(_ACTION_SEQUENCE.value)
            elif actions_sim is not None:
                self._idx = 0
                self._actions = actions_sim
            else:
                self._idx = 0
                self._actions = np.zeros(23)
        
        def __call__(self, timestep):
            del timestep
            if _ACTION_SEQUENCE.value is not None:
                actions = self._actions[self._idx][22:]
                self._idx += 1
                # return actions
            elif actions_sim is not None:
                # actions = self._actions[self._idx]
                if self._idx < len(self._actions):
                    actions = self._actions[self._idx]
                    # print("actions: ", actions)
                    self._idx += 1
                else:
                    actions = np.zeros(23)
                    # return actions
                # else:
                #     print("actions: ", actions)
                #     return actions
            else:
                # return np.zeros(23)
                actions = np.zeros(23)
            
            if len(actions) != 23:
                raise ValueError(f"Expected 23 actions, got {len(actions)}")
            actions[-1] = -1.0
            return actions
            
    policy = ActionSequencePlayer()

    robopianist_viewer.launch(env, policy=policy)

if __name__ == "__main__":
    app.run(main)