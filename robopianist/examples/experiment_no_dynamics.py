from pathlib import Path
import sys
import numpy as np
from absl import app, flags
import mujoco
import pandas as pd
import mido
from typing import Optional, Mapping, Any

sys.path.append("/Users/shjulie/Desktop/Piano_Robot_MPC/robopianist/robopianist")

from dm_control import composer
from dm_control.mjcf import export_with_assets
from mujoco import viewer as mujoco_viewer
from robopianist import music
from robopianist.wrappers.evaluation import MidiEvaluationWrapper
import robopianist.suite.tasks.piano_with_one_shadow_hand as TaskNoDynamics
from robopianist.models.hands.base import HandSide
from mujoco_utils import composer_utils

FLAGS = flags.FLAGS
flags.DEFINE_string("midi_file", "test_twinkle.mid", "Path to MIDI file")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes per condition")
flags.DEFINE_string("output_dir", "results", "Directory to save results")

ALL = ["RoboPianist-debug-TwinkleTwinkleLittleStar-v0"]
_ALL_DICT = {"RoboPianist-debug-TwinkleTwinkleLittleStar-v0": "TwinkleTwinkle"}

def create_simple_midi_file(path: Path) -> None:
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    notes = [48, 50, 52, 53]  # C3, D3, E3, F3
    time = 0
    for note in notes:
        track.append(mido.Message("note_on", note=note, velocity=64, time=time))
        track.append(mido.Message("note_off", note=note, velocity=64, time=500))
        time = 0
    mid.save(str(path))

def load(
    environment_name: str,
    task_class,
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
        task=task_class(midi=midi, **task_kwargs),
        random_state=seed,
        strip_singleton_obs_buffer_dim=True,
        recompile_physics=recompile_physics,
        legacy_step=legacy_step,
    )

def run_experiment(task_class, name, output_dir):
    midi_path = Path(FLAGS.midi_file)
    if not midi_path.exists():
        create_simple_midi_file(midi_path)

    task_kwargs = dict(
        change_color_on_activation=True,
        trim_silence=False,
        control_timestep=0.05,
        gravity_compensation=False,
        primitive_fingertip_collisions=False,
        reduced_action_space=False,
        n_steps_lookahead=1,
        n_seconds_lookahead=None,
        wrong_press_termination=False,
        disable_fingering_reward=False,
        disable_colorization=True,
        attachment_yaw=0.0,
        hand_side=HandSide.RIGHT,
        initial_buffer_time=0.5,
    )

    env = load(
        environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        task_class=task_class,
        midi_file=midi_path,
        shift=0,
        task_kwargs=task_kwargs,
    )
    wrapped_env = MidiEvaluationWrapper(env)

    results = {"episode": [], "key_press_reward": [], "finger_movement_reward": [], "energy_reward": []}
    custom_metrics = []

    for episode in range(FLAGS.num_episodes):
        timestep = wrapped_env.reset()
        step_count = 0
        episode_metrics = []

        while not timestep.last():
            # Use get_action to ensure proper task lifecycle
            action = wrapped_env.task.get_action(wrapped_env.physics)
            timestep = wrapped_env.step(action)
            step_count += 1

            # Log standard rewards
            results["episode"].append(episode)
            results["key_press_reward"].append(
                wrapped_env.task._reward_fn.reward_fns["key_press_reward"](wrapped_env.physics)
            )
            results["finger_movement_reward"].append(
                wrapped_env.task._reward_fn.reward_fns["finger_movement_reward"](wrapped_env.physics)
            )
            results["energy_reward"].append(
                wrapped_env.task._reward_fn.reward_fns["energy_reward"](wrapped_env.physics)
            )

            # Log custom metrics from task
            episode_metrics.append(wrapped_env.task.get_custom_metrics())

        print(f"{name} Episode {episode + 1}: {step_count} steps")
        custom_metrics.append(episode_metrics)

    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(f"{output_dir}/{name}_rewards.csv", index=False)
    
    flat_metrics = []
    for ep, ep_metrics in enumerate(custom_metrics):
        for step, metric in enumerate(ep_metrics):
            metric["episode"] = ep
            metric["step"] = step
            flat_metrics.append(metric)
    pd.DataFrame(flat_metrics).to_csv(f"{output_dir}/{name}_custom_metrics.csv", index=False)

def main(_):
    run_experiment(TaskNoDynamics.PianoWithOneShadowHand, "with_dynamics", FLAGS.output_dir)
    # Uncomment these when you add the other task classes
    # run_experiment(TaskWithDynamics.PianoWithOneShadowHand, "with_dynamics", FLAGS.output_dir)
    # run_experiment(TaskController.PianoWithOneShadowHand, "controller", FLAGS.output_dir)

if __name__ == "__main__":
    app.run(main)