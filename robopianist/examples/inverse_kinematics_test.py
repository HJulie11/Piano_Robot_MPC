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

sys.path.append("/Users/shjulie/Desktop/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

# from models.base import HandSide
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

import robopianist
import robopianist.models.hands as shadow_hand
import robopianist.music as music
from robopianist.suite.tasks import piano_with_one_shadow_hand
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from robopianist.models.piano import piano_constants as piano_consts
from robopianist import viewer, suite

import mink

# import kinematics.rrt_planner as rrt_planner
# import kinematics.get_config as get_config
# import parse_pose
# import pinocchio as pin
# from plot_traj import plot_traj
from util import solve_ik, get_key_pos, construct_model, jointlimitsviolated, projecttojointlimits
# from loop_rate_limiters import RateLimiter

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

    joints = env.task._hand.actuators[:-2] 
    hand_positions = env.task._hand.actuators[-2:]

    JOINT_INDS = [] # len = 20
    joint_names = []
    for joint in joints:
        if joint.joint:   
            JOINT_INDS.append(env.physics.model._model.joint("rh_shadow_hand/" + joint.joint.name).id)
            joint_names.append(joint.joint.name)
        elif joint.tendon:
            JOINT_INDS.append(env.physics.model._model.tendon("rh_shadow_hand/" + joint.tendon.name).id)
            joint_names.append(joint.tendon.name)

        else:
            raise ValueError(f"Joint or tendon not found for {joint.name}")
        
    HAND_INDS = [] # len = 2
    for hand in hand_positions:
        if hand.joint:
            HAND_INDS.append(env.physics.model._model.joint("rh_shadow_hand/" + hand.joint.name).id)
            joint_names.append(hand.joint.name)
        elif hand.tendon:
            HAND_INDS.append(env.physics.model._model.tendon("rh_shadow_hand/" + hand.tendon.name).id)
            joint_names.append(hand.tendon.name)
        else:
            raise ValueError(f"Joint or tendon not found for {hand.name}")
        
    QPOS_INDS = JOINT_INDS + HAND_INDS

    WRIST = ['rh_shadow_hand/rh_WRJ2','rh_shadow_hand/rh_WRJ2']

    forearm = ['rh_shadow_hand/forearm_tx','rh_shadow_hand/forearm_ty']

    FINGER_JOINT=[
        ['rh_shadow_hand/rh_THJ6','rh_shadow_hand/rh_THJ4','rh_shadow_hand/rh_THJ3','rh_shadow_hand/rh_THJ2'],
        ['rh_shadow_hand/rh_FFJ4','rh_shadow_hand/rh_FFJ3','rh_shadow_hand/rh_FFJ2','rh_shadow_hand/rh_FFJ1'],
        ['rh_shadow_hand/rh_MFJ4','rh_shadow_hand/rh_MFJ3','rh_shadow_hand/rh_MFJ2','rh_shadow_hand/rh_MFJ1'],
        ['rh_shadow_hand/rh_RFJ4','rh_shadow_hand/rh_RFJ3','rh_shadow_hand/rh_RFJ2','rh_shadow_hand/rh_RFJ1'],
        ['rh_shadow_hand/rh_LFJ4','rh_shadow_hand/rh_LFJ3','rh_shadow_hand/rh_LFJ2','rh_shadow_hand/rh_LFJ1'],
    ]
    
    key_pos = get_key_pos(env, 63, env.task._hand.fingertip_sites[3].name)
    ikresult=qpos_from_site_pose(env.physics,'rh_shadow_hand/'+ env.task._hand.fingertip_sites[3].name, key_pos, None, FINGER_JOINT[3] + WRIST + forearm)
    qpos = ikresult.qpos[QPOS_INDS]
    print("qpos: ", qpos)

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
        env = PianoSoundVideoWrapper(env, record_every=1)
    if _CANONICALIZE.value:
        env = CanonicalSpecWrapper(env)
    
    actions_sim = []
    node = np.zeros(23)
    node[:-1] = qpos
    only_hand = np.zeros(23)
    only_hand[-3:-1] = qpos[-2:]
    for _ in range(20):
        actions_sim.append(only_hand)
    actions_sim.append(node)
    print("actions_sim: ", actions_sim)

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
                return actions
            elif actions_sim is not None:
                actions = self._actions[self._idx]
                if self._idx < len(self._actions) - 1:
                    self._idx += 1
                    print("actions: ", actions)
                    return actions
                else:
                    print("actions: ", actions)
                    return actions
            else:
                return np.zeros(23)
            
    policy = ActionSequencePlayer()

    if not _RECORD.value:
        print("Running policy ...")
        if _HEADLESS.value:
            print("Running headless ...")
            timestep = env.reset()
            while not timestep.last():
                action = policy(timestep)
                timestep = env.step(action)
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