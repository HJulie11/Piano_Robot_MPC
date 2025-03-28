from pathlib import Path
import sys
import time
from typing import Any, Dict, Mapping, Optional, Union

# mujoco_menagerie rendering imports
from dataclasses import dataclass
import mediapy as media
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
from robopianist import viewer, suite

import mink

import kinematics.rrt_planner_2 as rrt
import kinematics.get_config as get_config
import parse_pose
import pinocchio as pin
from plot_traj import plot_traj
import kinematics.ik_2 as ik_2
from util import create_joint_limits_dict

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

    # if self._hand_side == base.HandSide.RIGHT:
    #         mjcf_utils.safe_find(self._mjcf_root, "site", "grasp_site").remove()

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
        
    # site_names:  ['thdistal_site', 'ffdistal_site', 'mfdistal_site', 'rfdistal_site', 'lfdistal_site']

    key_numbers = [63, 65, 67, 68, 70] # -> geom ids: [65, 67, 69, 70, 72]
    key_site_id = [env.physics.model.site("piano/" + env.task.piano.keys[key_number].site[0].name).id for key_number in key_numbers]
    key_pos = [env.physics.data.geom_xpos[key_id] for key_id in key_site_id]

    action_joints = env.task._hand.actuators
    joint_names = [joint.name for joint in action_joints]
    
    q_target_dict = {
        "thumb": np.array(key_pos[0]),
        "index": np.array(key_pos[1]),
        "middle": np.array(key_pos[2]),
        "ring": np.array(key_pos[3]),
        "little": np.array(key_pos[4])
    }

    site_names = {
        "thumb": "rh_shadow_hand/thdistal_site",
        "index": "rh_shadow_hand/ffdistal_site",
        "middle": "rh_shadow_hand/mfdistal_site",
        "ring": "rh_shadow_hand/rfdistal_site",
        "little": "rh_shadow_hand/lfdistal_site"
    }

    coupled_joints = []

    key_sequence = [63] #, 65, 67]
    press_durations = [2.0] #, 2.0, 2.0]  # 0.5 seconds each
    fingertip_mapping = {
        63: "thdistal_site", 
        # 65: "ffdistal_site", 
        # 67: "mfdistal_site"
    }
    planner = ik_2.PianoMotionPlanner(env, key_sequence, press_durations, fingertip_mapping)
    path = planner.hybrid_plan_motion()
    
    action_sequence = []
    for q in path:
        node = np.zeros(23)
        node[:-1] = q[np.array(planner.ik_solver.qpos_inds)]
        action_sequence.append(node)

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
    
    if _RECORD.value or True:
        env = PianoSoundVideoWrapper(env, record_every=1)
    if _CANONICALIZE.value:
        env = CanonicalSpecWrapper(env)
    
    action_spec = env.action_spec()
    zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    zeros[-1] = -1.0  # Disable sustain pedal.
    
    timestep = env.reset()
    dim = 0
    print("observation: ", timestep.observation.items())
    for k, v in timestep.observation.items():
        print(f"\t{k}: {v.shape} {v.dtype}")
        dim += int(np.prod(v.shape))
    print(f"Observation dimension: {dim}")

    print(f"Control frequency: {1 / _CONTROL_TIMESTEP.value} Hz")
    print(f"npy file: {_ACTION_SEQUENCE.value}")

    # actions_sim = []
    # if action_sequence is not None:
    #     actions_sim = action_sequence
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
                    print("actions: ", actions)
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

    if not _RECORD.value:
        print("Running policy ...")
        if _HEADLESS.value:
            print("Running headless ...")
            timestep = env.reset()
            while not timestep.last():
                action = policy(timestep)
                timestep = env.step(action)
                for finger, site_name in site_names.items():
                    site_id = env.physics.model.site(site_name).id
                    fingertip_pos = env.physics.data.site_xpos[site_id]
                    print(f"{finger}: {fingertip_pos}")
                for key_idx in key_sequence:
                    key_site_name = env.task.piano._sites[key_idx].name
                    key_site_id = env.physics.model.site("piano/" + key_site_name).id
                    key_pos = env.physics.data.site_xpos[key_site_id]
                    print(f"key {key_idx}: {key_pos}")

        else:
            print("Running viewer ...")
            # export figure (reward graph) here
            viewer.launch(env, policy=policy)
    else:
        timestep = env.reset()
        while not timestep.last():
            action = policy(timestep)
            timestep = env.step(action)

if __name__ == "__main__":
    app.run(main)