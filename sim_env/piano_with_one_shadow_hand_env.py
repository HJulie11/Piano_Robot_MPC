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

import kinematics.rrt_planner as rrt_planner
import kinematics.get_config as get_config
import parse_pose
import pinocchio as pin
from plot_traj import plot_traj
from util import solve_ik, get_key_pos, construct_model, jointlimitsviolated, projecttojointlimits
from loop_rate_limiters import RateLimiter

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
        
    # site_names:  ['thdistal_site', 'ffdistal_site', 'mfdistal_site', 'rfdistal_site', 'lfdistal_site']
    key_pos = get_key_pos(env, 63)
    ikresult=qpos_from_site_pose(env.physics,'rh_shadow_hand/'+ env.task._hand.fingertip_sites[0].name, key_pos, None, None)
    qpos = ikresult.qpos[QPOS_INDS]
    print("qpos: ", qpos)

    action_sequence = []
    fingertip_list = ["rh_shadow_hand/thdistal_site", "rh_shadow_hand/ffdistal_site", "rh_shadow_hand/mfdistal_site", "rh_shadow_hand/thdistal_site", "rh_shadow_hand/ffdistal_site"]
    # goal_site_list = ["piano/white_key_site_38", "piano/white_key_site_39", "piano/white_key_site_41"]  
    goal_number_list = [38, 39, 41, 43, 44]
    goal_site_list = []
    goal_pos_list = []
    goal_geom_list = []
    # key_body_list = []

    # TODO: uncomment this after successfully getting neutral pose
    for goal_number in goal_number_list:
        goal_name = f"piano/white_key_geom_{goal_number}"
        goal_id = env.physics.model.geom(goal_name).id
        goal_geom_list.append(goal_id)
        goal_pos = env.physics.data.geom_xpos[goal_id].copy() # this works the best
        goal_pos_list.append(np.array(goal_pos))

    # print("contact", type(env.physics.data.contact))
    # print(mujoco.mj_collision(env.physics.model._model, env.physics.data._data))
    # print("geom in contact", env.physics.model.geom("piano/black_key_geom_52").id in env.physics.data.contact.geom)
    # print("contact geom", env.physics.data.contact)
    # return

    print("Getting neutral pose")
    neutral_pose = np.zeros(23)
    # site_names =  ['rh_shadow_hand/thdistal_site', 'rh_shadow_hand/ffdistal_site', 'rh_shadow_hand/mfdistal_site', 'rh_shadow_hand/rfdistal_site', 'rh_shadow_hand/lfdistal_site']
    site_names =  ['rh_shadow_hand/thdistal_site', 'rh_shadow_hand/ffdistal_site', 'rh_shadow_hand/mfdistal_site', 'rh_shadow_hand/rfdistal_site', 'rh_shadow_hand/lfdistal_site']
    # neutral_key_numbers = [38, 39, 41, 43, 44]
    neutral_key_numbers = [63, 65, 67, 68, 70] # -> geom ids: [65, 67, 69, 70, 72]
    neutral_goal_pos = []
    for i, number in enumerate(neutral_key_numbers):
        site = env.task.piano.keys[number].site #.geom[0]
        goal_name = site[0].name
        goal_id = env.physics.model.site("piano/" + goal_name).id
        goal_site_list.append(goal_id)
        goal_pos = env.physics.data.site_xpos[goal_id].copy()
        goal_pos[2] += 0.05
        goal_pos[1] -= 0.05
        neutral_goal_pos.append(np.array(goal_pos))
    
    print("ids: ", goal_site_list)
    
    neutral_qpos, success = get_config.compute_ik_multiple_keys(
        env, 
        env.physics.model._model, 
        env.physics.data._data,
        env.physics.data.qpos[QPOS_INDS].copy(),
        site_names,
        neutral_key_numbers,
        neutral_goal_pos,
        QPOS_INDS
    )

    # rate = RateLimiter(frequency=500.0)

    # action_spec = env.action_spec()
    # joint_limits = np.array([(low, high) for low, high in zip(action_spec.minimum[:22], action_spec.maximum[:22])])

    # viewer = mujoco.viewer.launch_passive(model=model, data=data, show_left_ui=False, show_right_ui=False)

    # ik_result = solve_ik(env, "rh_shadow_hand/thdistal_site", "piano/white_key_site_63", rate, press=True)
    # qpos = ik_result[QPOS_INDS]
    # print("qpos: ", qpos)
    # print("joint limits violated: ", jointlimitsviolated(joint_limits, qpos))
    # qpos = projecttojointlimits(joint_limits, qpos)

    # node = np.zeros(23)
    # node[:-1] = qpos
    # action_sequence.append(node)

    
    print("success: ", success)

    print("neutral goal pos: ", neutral_goal_pos)
    print("Neutral pose: ", neutral_qpos)
    move_torwards_neutral = np.zeros(23)
    move_torwards_neutral[-3:-1] = neutral_qpos[-2:]
    action_sequence.append(np.array(move_torwards_neutral))
    neutral_pose[:-3] = neutral_qpos[:-2]
    action_sequence.append(np.array(neutral_pose))
    print("action_sequence: ", action_sequence)




    # TODO: press and release keys determination should be done between the path for each finger
    print("Computing path...")
    # for idx, goal in enumerate(goal_number_list):
    #     print("current key: ", goal)
    #     planner = rrt_planner.RRTPlanner(env, goal, fingertip_list[idx], max_iter=100, step_size=0.05)
    #     path = planner.plan()
    #     if path is not None:
    #         print("path from rrt:", path)
    #         action_sequence.extend(path)
    #         # print("lifting action")
    #         # lift_action = np.zeros(23)
    #         # lift_goal_pos = goal_pos_list[idx].copy()
    #         # lift_goal_pos[2] += 0.1
    #         # lift_action_qpos, success = get_config.compute_ik(
    #         #     env,
    #         #     env.physics.model._model,
    #         #     env.physics.data._data,
    #         #     env.physics.data.qpos[QPOS_INDS].copy(),
    #         #     goal,
    #         #     lift_goal_pos,
    #         #     goal,
    #         #     fingertip_list[idx],
    #         #     QPOS_INDS
    #         # )
    #         # lift_action[:-1] = lift_action_qpos
    #         # action_sequence.append(lift_action)
    #     else:
    #         print("No valid path found.")
        
    # plot_traj(action_sequence, title="Trajectory")

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
    
    # action_spec = env.action_spec()
    # zeros = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    # zeros[-1] = -1.0  # Disable sustain pedal.
    
    # timestep = env.reset()
    # dim = 0
    # print("observation: ", timestep.observation.items())
    # for k, v in timestep.observation.items():
    #     print(f"\t{k}: {v.shape} {v.dtype}")
    #     dim += int(np.prod(v.shape))
    # print(f"Observation dimension: {dim}")

    # print(f"Control frequency: {1 / _CONTROL_TIMESTEP.value} Hz")
    # print(f"npy file: {_ACTION_SEQUENCE.value}")

    # actions_sim = []
    # if action_sequence is not None:
    #     actions_sim = action_sequence
    
    actions_sim = []
    node = np.zeros(23)
    node[:-1] = qpos
    only_hand = np.zeros(23)
    only_hand[-3:-1] = qpos[-2:]
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
    actions_sim.append(only_hand)
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