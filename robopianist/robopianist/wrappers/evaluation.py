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

"""A wrapper for tracking episode statistics pertaining to music performance.

TODO(kevin):
- Look into `mir_eval` for metrics.
- Should sustain be a separate metric or should it just be applied to the note sequence
    as a whole?
"""

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Sequence

import dm_env
import numpy as np
import pandas as pd
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(self, environment: dm_env.Environment, deque_size: int = 1) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []
        self._sustain_presses: List[np.ndarray] = []

        # Keep track of episode number for plotting
        self._episode_count = 0
        self._episode_numbers: Deque[int] = deque(maxlen=deque_size)

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

        # Sustain metrics.
        self._sustain_precisions: Deque[float] = deque(maxlen=deque_size)
        self._sustain_recalls: Deque[float] = deque(maxlen=deque_size)
        self._sustain_f1s: Deque[float] = deque(maxlen=deque_size)

        # Track timestep and reward for each step
        self._timestep_rewards: List[tuple[int, float]] = []
        self._timestep_key_press_rewards: List[tuple[int, float]] = []
        self._timestep_energy_rewards: List[tuple[int, float]] = []
        self._timestep_finger_movement_rewards: List[tuple[int, float]] = []
        self._current_timestep = 0

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        # Log timestep and reward
        # print(timestep)
        # return
        reward = timestep.reward if timestep.reward is not None else 0.0 # Handle None reward
        key_press_reward = self._environment.task._reward_fn.reward_fns["key_press_reward"](self._environment.physics)
        energy_reward = self._environment.task._reward_fn.reward_fns["energy_reward"](self._environment.physics)
        finger_movement_reward = self._environment.task._reward_fn.reward_fns["finger_movement_reward"](self._environment.physics)
        self._timestep_key_press_rewards.append((self._current_timestep, key_press_reward))
        self._timestep_energy_rewards.append((self._current_timestep, energy_reward))
        self._timestep_finger_movement_rewards.append((self._current_timestep, finger_movement_reward))

        self._timestep_rewards.append((self._current_timestep, reward))
        self._current_timestep += 1

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))
        sustain_activation = self._environment.task.piano.sustain_activation
        self._sustain_presses.append(sustain_activation.astype(np.float64))

        if timestep.last():
            self._episode_count += 1
            self._episode_numbers.append(self._episode_count)

            key_press_metrics = self._compute_key_press_metrics()
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            # sustain_metrics = self._compute_sustain_metrics()
            # self._sustain_precisions.append(sustain_metrics.precision)
            # self._sustain_recalls.append(sustain_metrics.recall)
            # self._sustain_f1s.append(sustain_metrics.f1)

            self._key_presses = []
            self._sustain_presses = []
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        self._sustain_presses = []
        self._timestep_rewards = []
        self._current_timestep = 0
        return self._environment.reset()

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean precision/recall/F1 over the last `deque_size` episodes."""
        if not self._key_press_precisions:
            raise ValueError("No episode metrics available yet.")

        def _mean(seq: Sequence[float]) -> float:
            return sum(seq) / len(seq)

        return {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
            "sustain_precision": _mean(self._sustain_precisions),
            "sustain_recall": _mean(self._sustain_recalls),
            "sustain_f1": _mean(self._sustain_f1s),
        }
    
    def save_timestep_rewards(self, filename: str) -> None:
        """Saves the timestep and reward data to a CSV file."""
        if not self._timestep_rewards:
            print("No timestep rewards to save.")
            return
        
        data = {
            "Timestep": [t for t, r in self._timestep_rewards],
            "Reward": [r for t, r in self._timestep_rewards],
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Timestep rewards saved to {filename}.")

    def save_timestep_key_press_rewards(self, filename: str) -> None:
        """Saves the timestep and key press reward data to a CSV file."""
        if not self._timestep_key_press_rewards:
            print("No timestep key press rewards to save.")
            return
        
        data = {
            "Timestep": [t for t, r in self._timestep_key_press_rewards],
            "Reward": [r for t, r in self._timestep_key_press_rewards],
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Timestep key press rewards saved to {filename}.")

    def save_timestep_energy_rewards(self, filename: str) -> None:
        """Saves the timestep and energy reward data to a CSV file."""
        if not self._timestep_energy_rewards:
            print("No timestep energy rewards to save.")
            return
        
        data = {
            "Timestep": [t for t, r in self._timestep_energy_rewards],
            "Reward": [r for t, r in self._timestep_energy_rewards],
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Timestep energy rewards saved to {filename}.")

    def save_timestep_finger_movement_rewards(self, filename: str) -> None:
        """Saves the timestep and finger movement reward data to a CSV file."""
        if not self._timestep_finger_movement_rewards:
            print("No timestep finger movement rewards to save.")
            return
        
        data = {
            "Timestep": [t for t, r in self._timestep_finger_movement_rewards],
            "Reward": [r for t, r in self._timestep_finger_movement_rewards],
        }

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Timestep finger movement rewards saved to {filename}.")

    # Helper methods.

    def _compute_key_press_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for key presses over the episode."""
        # Get the ground truth key presses.
        note_seq = self._environment.task._notes
        ground_truth = []
        for notes in note_seq:
            presses = np.zeros((self._environment.task.piano.n_keys,), dtype=np.float64)
            keys = [note.pitch - 21 for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # Deal with the case where the episode gets truncated due to a failure. In this
        # case, the length of the key presses will be less than or equal to the length
        # of the ground truth.
        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._key_presses)]

        assert len(ground_truth) == len(self._key_presses)

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._key_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)

    def _compute_sustain_metrics(self) -> EpisodeMetrics:
        """Computes precision/recall/F1 for sustain presses over the episode."""
        # Get the ground truth sustain presses.
        ground_truth = [
            np.atleast_1d(v).astype(float) for v in self._environment.task._sustains
        ]

        if hasattr(self._environment.task, "_wrong_press_termination"):
            failure_termination = self._environment.task._wrong_press_termination
            if failure_termination:
                ground_truth = ground_truth[: len(self._sustain_presses)]

        precisions = []
        recalls = []
        f1s = []
        for y_true, y_pred in zip(ground_truth, self._sustain_presses):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

        return EpisodeMetrics(precision, recall, f1)
