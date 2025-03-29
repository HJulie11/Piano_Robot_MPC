import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from robopianist.models.hands.base import HandSide

from matplotlib.animation import FuncAnimation

# plot helper functions:

def plot_fingertip_positions(trajectory, physics, hand_task, hand_side, timesteps):
    # Collect fingertip positions over time
    fingertip_positions = {finger: [] for finger in range(5)}
    for t in range(len(timesteps)):
        for finger in range(5):
            site_name = hand_task.fingertip_sites[finger].name
            if hand_side == HandSide.LEFT:
                full_site_name = f"lh_shadow_hand/{site_name}"
            else:
                full_site_name = f"rh_shadow_hand/{site_name}"
            fingertip_pos = physics.named.data.site_xpos[full_site_name]
            fingertip_positions[finger].append(fingertip_pos)
    
    # Conver to numpy arrays for easier plotting
    for finger in fingertip_positions:
        fingertip_positions[finger] = np.array(fingertip_positions[finger])

    # Plot x, y, z coordinates over time for each finger
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].set_title("Fingertip Positions Over Time")
    axes[0].set_ylabel("X Position (m)")
    axes[1].set_ylabel("Y Position (m)")
    axes[2].set_ylabel("Z Position (m)")
    axes[2].set_xlabel("Timestep")

    colors = ['b', 'g', 'r', 'c', 'm']  # One color per finger
    for finger in range(5):
        axes[0].plot(timesteps, fingertip_positions[finger][:, 0], label=f"Finger {finger}", color=colors[finger])
        axes[1].plot(timesteps, fingertip_positions[finger][:, 1], label=f"Finger {finger}", color=colors[finger])
        axes[2].plot(timesteps, fingertip_positions[finger][:, 2], label=f"Finger {finger}", color=colors[finger])

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_distance_to_keys(simulation_data): # trajectory, physics, task, hand_side, keys_current_history, timesteps
    # Collect distances to target keys over time
    # distances = {finger: [] for finger in range(5)}
    # for t in range(len(timesteps)):
    #     keys_current = keys_current_history[t]
    #     for finger in range(5):
    #         site_name = task._hand.fingertip_sites[finger].name
    #         if hand_side == HandSide.LEFT:
    #             full_site_name = f"lh_shadow_hand/{site_name}"
    #         else:
    #             full_site_name = f"rh_shadow_hand/{site_name}"
    #         fingertip_pos = physics.named.data.site_xpos[full_site_name].copy()

    #         # Find the target key for this finger
    #         target_key = None
    #         for key, mjcf_fingering in keys_current:
    #             if mjcf_fingering == finger:
    #                 target_key = key
    #                 break

    #         if target_key is not None:
    #             key_site = task.piano.keys[target_key].site[0]
    #             key_pos = physics.bind(key_site).xpos.copy()
    #             distance = np.linalg.norm(fingertip_pos - key_pos)
    #             distances[finger].append(distance)
    #         else:
    #             distances[finger].append(np.nan)  # No target key for this finger

    distances = simulation_data["distances"]
    timesteps = simulation_data["timesteps"]

    # Plot distances
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Distance to Target Keys Over Time")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance (m)")

    colors = ['b', 'g', 'r', 'c', 'm']
    for finger in range(5):
        ax.plot(timesteps, distances[finger], label=f"Finger {finger}", color=colors[finger])

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fingertip_trajectories_3d(trajectory, physics, hand_side, timesteps):
    # Collect fingertip positions over time
    fingertip_positions = {finger: [] for finger in range(5)}
    for t in range(len(timesteps)):
        for finger in range(5):
            site_name = physics.model.fingertip_sites[finger].name
            if hand_side == HandSide.LEFT:
                full_site_name = f"lh_shadow_hand/{site_name}"
            else:
                full_site_name = f"rh_shadow_hand/{site_name}"
            fingertip_pos = physics.named.data.site_xpos[full_site_name]
            fingertip_positions[finger].append(fingertip_pos)

    # Convert to numpy arrays for easier plotting
    for finger in fingertip_positions:
        fingertip_positions[finger] = np.array(fingertip_positions[finger])
    
    # Plot 3D trajectories of fingertip positions
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Fingertip Trajectories")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")

    colors = ['b', 'g', 'r', 'c', 'm']
    for finger in range(5):
        ax.plot(
            fingertip_positions[finger][:, 0],
            fingertip_positions[finger][:, 1],
            fingertip_positions[finger][:, 2],
            label=f"Finger {finger}",
            color=colors[finger]
        )
        # Mark the start and end points
        ax.scatter(
            fingertip_positions[finger][0, 0],
            fingertip_positions[finger][0, 1],
            fingertip_positions[finger][0, 2],
            color=colors[finger],
            marker='o',
            s=100,
            label=f"Finger {finger} Start"
        )
        ax.scatter(
            fingertip_positions[finger][-1, 0],
            fingertip_positions[finger][-1, 1],
            fingertip_positions[finger][-1, 2],
            color=colors[finger],
            marker='x',
            s=100,
            label=f"Finger {finger} End"
        )

    ax.legend()
    plt.show()

def plot_joint_positions(trajectory, physics, full_actuator, timesteps):
    # Collect joint positions over time
    joint_positions = {finger: {joint: [] for joint in full_actuator[finger]} for finger in range(5)}
    for t in range(len(timesteps)):
        for finger in range(5):
            for joint_name in full_actuator[finger]:
                if "J0" in joint_name:
                    joint_idx = physics.model.name2id(joint_name, "tendon")
                else:
                    joint_idx = physics.model.name2id(joint_name, "joint")
                joint_pos = physics.data.qpos[joint_idx]
                joint_positions[finger][joint_name].append(joint_pos)

    # Plot joint positions for each finger
    for finger in range(5):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Finger {finger} Joint Positions Over Time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Joint Position (rad)")

        for joint_name in joint_positions[finger]:
            ax.plot(timesteps, joint_positions[finger][joint_name], label=joint_name)

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_actions(trajectory, full_actuator, timesteps):
    # Convert trajectory to numpy array
    trajectory = np.array(trajectory)  # Shape: (timesteps, n_actuators + 1)

    # Plot actions for each finger
    for finger in range(5):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Finger {finger} Actions Over Time")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Action Value")

        for i, joint_name in enumerate(full_actuator[finger]):
            try:
                if "J0" in joint_name:
                    action_idx = next(
                        i for i, act in enumerate(self._hand.actuators)
                        if (hasattr(act, "tendon") and act.tendon is not None and act.tendon.name in joint_name)
                    )
                else:
                    action_idx = next(
                        i for i, act in enumerate(self._hand.actuators)
                        if (hasattr(act, "joint") and act.joint is not None and act.joint.name in joint_name)
                    )
                ax.plot(timesteps, trajectory[:, action_idx], label=joint_name)
            except StopIteration:
                print(f"No actuator found for joint {joint_name}")
                continue

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()
    
def animate_hand_motion(trajectory, physics, hand_side, timesteps):
    # Collect fingertip positions over time
    fingertip_positions = {finger: [] for finger in range(5)}
    for t in range(len(timesteps)):
        for finger in range(5):
            site_name = physics.model.fingertip_sites[finger].name
            if hand_side == HandSide.LEFT:
                full_site_name = f"lh_shadow_hand/{site_name}"
            else:
                full_site_name = f"rh_shadow_hand/{site_name}"
            fingertip_pos = physics.named.data.site_xpos[full_site_name].copy()
            fingertip_positions[finger].append(fingertip_pos)

    # Convert to numpy arrays
    for finger in fingertip_positions:
        fingertip_positions[finger] = np.array(fingertip_positions[finger])

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Hand Motion Animation")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")

    # Set axis limits based on the data
    all_positions = np.concatenate([fingertip_positions[finger] for finger in range(5)], axis=0)
    ax.set_xlim(np.min(all_positions[:, 0]) - 0.1, np.max(all_positions[:, 0]) + 0.1)
    ax.set_ylim(np.min(all_positions[:, 1]) - 0.1, np.max(all_positions[:, 1]) + 0.1)
    ax.set_zlim(np.min(all_positions[:, 2]) - 0.1, np.max(all_positions[:, 2]) + 0.1)

    # Initialize scatter plots for each fingertip
    colors = ['b', 'g', 'r', 'c', 'm']
    scatters = [ax.scatter([], [], [], color=colors[finger], label=f"Finger {finger}") for finger in range(5)]

    def update(frame):
        for finger in range(5):
            scatters[finger]._offsets3d = (
                [fingertip_positions[finger][frame, 0]],
                [fingertip_positions[finger][frame, 1]],
                [fingertip_positions[finger][frame, 2]]
            )
        ax.set_title(f"Hand Motion Animation (Timestep {frame})")
        return scatters

    ani = FuncAnimation(fig, update, frames=len(timesteps), interval=200, blit=False)
    plt.legend()
    plt.show()