# evaluation.py

import numpy as np
from pathlib import Path
import mido
from dm_control import mujoco
from dm_control.composer import Environment
from robopianist.music.midi_file import MidiFile
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
from robopianist.models.hands.base import HandSide

# Constants
CONTROL_TIMESTEP = 0.05
INITIAL_BUFFER_TIME = 0.5
OUTPUT_NPY_PATH = Path("action_sequence.npy")

def create_simple_midi_file(path: Path) -> None:
    """Create a simple MIDI file for 'Twinkle, Twinkle, Little Star'."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Define the notes for "Twinkle, Twinkle, Little Star"
    notes = [
        (60, 500),  # C4
        (60, 500),  # C4
        (67, 500),  # G4
        (67, 500),  # G4
        (69, 500),  # A4
        (69, 500),  # A4
        (67, 500),  # G4
    ]

    # Add note_on and note_off messages with proper timing
    current_time = 0
    for pitch, duration in notes:
        # Note on
        track.append(mido.Message("note_on", note=pitch, velocity=64, time=current_time))
        current_time = duration  # Time until note_off
        # Note off
        track.append(mido.Message("note_off", note=pitch, velocity=64, time=current_time))
        current_time = 0  # Reset time for the next note_on

    mid.save(str(path))
    print(f"Created MIDI file at {path}")

def evaluate_action_sequence(env, action_sequence):
    """Run the action sequence in the environment and collect evaluation metrics."""
    print("Starting evaluation...")

    # Reset the environment
    time_step = env.reset()
    total_reward = 0.0
    correct_key_presses = 0
    total_keys_to_press = 0
    correct_fingerings = 0
    total_fingerings = 0
    wrong_presses = 0
    episode_length = 0

    # Expected keys and timesteps from the MIDI file
    # Adjusted for initial_buffer_time = 0.5s, control_timestep = 0.05s
    # Each note starts 0.5s apart, initial buffer adds 0.5s (10 timesteps)
    expected_keys = {
        10: 39,  # Pitch 60 (C4) -> key 39 at t=0.5s (timestep 10)
        20: 39,  # Pitch 60 (C4) -> key 39 at t=1.0s (timestep 20)
        30: 46,  # Pitch 67 (G4) -> key 46 at t=1.5s (timestep 30)
        40: 46,  # Pitch 67 (G4) -> key 46 at t=2.0s (timestep 40)
        50: 48,  # Pitch 69 (A4) -> key 48 at t=2.5s (timestep 50)
        60: 48,  # Pitch 69 (A4) -> key 48 at t=3.0s (timestep 60)
        70: 46,  # Pitch 67 (G4) -> key 46 at t=3.5s (timestep 70)
    }
    expected_fingers = {
        10: 3,  # Key 39 -> finger 3
        20: 3,  # Key 39 -> finger 3
        30: 3,  # Key 46 -> finger 3
        40: 3,  # Key 46 -> finger 3
        50: 4,  # Key 48 -> finger 4
        60: 4,  # Key 48 -> finger 4
        70: 3,  # Key 46 -> finger 3
    }

    for t, action in enumerate(action_sequence):
        if time_step.last():
            print(f"Episode terminated early at timestep {t}")
            break

        # Step the environment with the action
        time_step = env.step(action)
        episode_length += 1

        # Collect reward
        reward = time_step.reward if time_step.reward is not None else 0.0
        total_reward += reward

        # Check key presses and fingering (using task's internal state)
        task = env.task
        keys_pressed = task._keys_current  # List of (key, finger) tuples
        print(f"Timestep {t}: Keys pressed: {keys_pressed}, Reward: {reward}")

        # Evaluate key press accuracy
        expected_key = expected_keys.get(t, None)
        if expected_key is not None:
            total_keys_to_press += 1
            key_pressed = None
            for key, _ in keys_pressed:
                if key == expected_key:
                    key_pressed = key
                    correct_key_presses += 1
                    break
            if key_pressed is None:
                print(f"Timestep {t}: Missed key {expected_key}")

        # Evaluate fingering accuracy
        expected_finger = expected_fingers.get(t, None)
        if expected_finger is not None:
            total_fingerings += 1
            for key, finger in keys_pressed:
                if key == expected_key and finger == expected_finger:
                    correct_fingerings += 1
                    break

        # Check for wrong presses
        if expected_key is None and keys_pressed:
            wrong_presses += len(keys_pressed)
            print(f"Timestep {t}: Wrong key press: {keys_pressed}")

    # Compute metrics
    key_accuracy = (correct_key_presses / total_keys_to_press * 100) if total_keys_to_press > 0 else 0
    fingering_accuracy = (correct_fingerings / total_fingerings * 100) if total_fingerings > 0 else 0

    # Print evaluation summary
    print("\nEvaluation Summary:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Episode Length: {episode_length} timesteps")
    print(f"Key Press Accuracy: {key_accuracy:.2f}% ({correct_key_presses}/{total_keys_to_press})")
    print(f"Fingering Accuracy: {fingering_accuracy:.2f}% ({correct_fingerings}/{total_fingerings})")
    print(f"Wrong Presses: {wrong_presses}")

    return {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "key_accuracy": key_accuracy,
        "fingering_accuracy": fingering_accuracy,
        "wrong_presses": wrong_presses,
    }

def main():
    # Step 1: Create a test MIDI file
    test_midi_path = Path("test_twinkle.mid")
    create_simple_midi_file(test_midi_path)

    # Step 2: Load and parse the MIDI file manually
    try:
        # First, attempt to load the MIDI file using MidiFile
        midi = MidiFile(test_midi_path)
        print(f"Loaded MIDI file: {midi}")

        # Check if the MIDI file was parsed correctly
        if not hasattr(midi, 'seq') or not hasattr(midi.seq, 'notes'):
            print("MidiFile did not parse the MIDI file correctly. Parsing manually...")
            # Manually parse the MIDI file using mido
            mido_midi = mido.MidiFile(str(test_midi_path))
            notes = []
            current_time = 0
            note_on_times = {}  # To track start times of notes

            # Convert ticks to seconds (assuming default tempo of 500,000 microseconds per beat, 120 BPM)
            ticks_per_beat = mido_midi.ticks_per_beat
            seconds_per_tick = 500000 / 1000000 / ticks_per_beat  # 500,000 us/beat = 0.5 s/beat at 120 BPM

            for msg in mido_midi:
                current_time += msg.time * seconds_per_tick  # Convert delta time to seconds
                if msg.type == "note_on" and msg.velocity > 0:
                    note_on_times[msg.note] = current_time
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in note_on_times:
                        start_time = note_on_times[msg.note]
                        end_time = current_time
                        # Create a note object matching the expected structure
                        note = type('Note', (), {
                            'start_time': start_time,
                            'end_time': end_time,
                            'pitch': msg.note,
                            'velocity': msg.velocity if msg.type == "note_on" else 0
                        })()
                        notes.append(note)
                        del note_on_times[msg.note]

            # Create a sequence object
            sequence = type('Sequence', (), {
                'notes': notes,
                'events': []  # Add empty events list for sustain pedal (if needed)
            })()

            # Create a new MidiFile object with the parsed sequence
            # Since we cannot assign to midi.seq, we create a mock MidiFile object
            class MockMidiFile:
                def __init__(self, seq):
                    self.seq = seq

            midi = MockMidiFile(sequence)
            print(f"Manually parsed MIDI notes: {midi.seq.notes}")
        else:
            print(f"MIDI notes: {midi.seq.notes}")
    except Exception as e:
        print(f"Failed to load MIDI file: {e}")
        return

    # Step 3: Initialize the PianoWithOneShadowHand task
    try:
        task = PianoWithOneShadowHand(
            midi=midi,
            hand_side=HandSide.RIGHT,
            control_timestep=CONTROL_TIMESTEP,
            initial_buffer_time=INITIAL_BUFFER_TIME,
            trim_silence=False,
            wrong_press_termination=False,
            disable_fingering_reward=False,
            disable_colorization=False,
            n_steps_lookahead=1,
            n_seconds_lookahead=None,
            augmentations=None,
        )
        print("Initialized PianoWithOneShadowHand task")
    except Exception as e:
        print(f"Failed to initialize task: {e}")
        return

    # Step 4: Create a physics instance and environment
    try:
        physics = mujoco.Physics.from_xml_string(task.root_entity.mjcf_model.to_xml_string())
        env = Environment(task, time_limit=10.0, random_state=np.random.RandomState(42))
        print("Created environment")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return

    # Step 5: Generate the action sequence
    try:
        action_sequence = task.get_action_trajectory(physics)
        action_sequence = np.array(action_sequence)
        print(f"Generated action sequence with shape: {action_sequence.shape}")
        print(f"First few actions:\n{action_sequence[:5]}")
    except Exception as e:
        print(f"Failed to generate action sequence: {e}")
        return

    # Step 6: Save the action sequence to a .npy file
    try:
        np.save(OUTPUT_NPY_PATH, action_sequence)
        print(f"Saved action sequence to {OUTPUT_NPY_PATH}")
    except Exception as e:
        print(f"Failed to save action sequence: {e}")
        return

    # Step 7: Evaluate the action sequence
    try:
        metrics = evaluate_action_sequence(env, action_sequence)
    except Exception as e:
        print(f"Failed to evaluate action sequence: {e}")
        return

if __name__ == "__main__":
    main()