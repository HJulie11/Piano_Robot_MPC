import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# from fastdtw import fastdtw
# from scipy.spatial.distance import euclidean

# Load the timestep rewards
df_reward = pd.read_csv("results/timestep_rewards.csv")
df_key_press = pd.read_csv("results/timestep_key_press_rewards.csv")
df_finger_move = pd.read_csv("results/timestep_finger_movement_rewards.csv")
df_energy = pd.read_csv("results/timestep_energy_rewards.csv")

plt.figure(figsize=(12, 6))
plt.plot(df_reward["Timestep"], df_reward["Reward"], label="Reward", color="blue")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Reward Over Time")
plt.legend()
plt.grid()
plt.savefig("results/reward_over_time.png", format="png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_key_press["Timestep"], df_key_press["Reward"], label="Reward", color="orange")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Key Press Reward Over Time")
plt.legend()
plt.grid()
plt.savefig("results/key_press_over_time.png", format="png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_energy["Timestep"], df_energy["Reward"], label="Reward", color="red")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Energy Reward Over Time")
plt.legend()
plt.grid()
plt.savefig("results/energy_over_time.png", format="png", dpi=300)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_finger_move["Timestep"], df_finger_move["Reward"], label="Reward", color="green")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("Finger Movement Reward Over Time")
plt.legend()
plt.grid()
plt.savefig("results/finger_over_time.png", format="png", dpi=300)
plt.show()

# Plot the fingertip trajectories in 3D

thumb_trajectory = np.load("results/trajectories/thumb_trajectory.npy")
index_trajectory = np.load("results/trajectories/index_trajectory.npy")
middle_trajectory = np.load("results/trajectories/middle_trajectory.npy")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Get the first 10 colors from tab10
finger_colors = {0: colors[0], 1: colors[1], 2: colors[2]}  # Assign to thumb, index, middle

ax.plot(thumb_trajectory[:, 0], thumb_trajectory[:, 1], thumb_trajectory[:, 2], label="Thumb", color=finger_colors[0])
ax.scatter(thumb_trajectory[0][0], thumb_trajectory[0][1], thumb_trajectory[0][2], color=finger_colors[0], marker='o', s=50)  # Start point
ax.scatter(thumb_trajectory[0][0], thumb_trajectory[0][1], thumb_trajectory[0][2], color=finger_colors[0], marker='x', s=50)  # Start point

ax.plot(index_trajectory[:, 0], index_trajectory[:, 1], index_trajectory[:, 2], label="Index", color=finger_colors[1])
ax.scatter(index_trajectory[0][0], index_trajectory[0][1], index_trajectory[0][2], color=finger_colors[1], marker='o', s=50)  # Start point
ax.scatter(index_trajectory[0][0], index_trajectory[0][1], index_trajectory[0][2], color=finger_colors[1], marker='x', s=50)  # Start point

ax.plot(middle_trajectory[:, 0], middle_trajectory[:, 1], middle_trajectory[:, 2], label="Middle", color=finger_colors[2])
ax.scatter(middle_trajectory[0][0], middle_trajectory[0][1], middle_trajectory[0][2], color=finger_colors[2], marker='o', s=50)  # Start point
ax.scatter(middle_trajectory[0][0], middle_trajectory[0][1], middle_trajectory[0][2], color=finger_colors[2], marker='x', s=50)  # Start point

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.title("Fingertip Trajectories in 3D Space")
plt.savefig("results/trajectories/fingertip_trajectories_controller_executed.png", format="png", dpi=300)
plt.show()

timesteps = [i for i in range(len(thumb_trajectory))]
plt.figure(figsize=(12, 6))
plt.plot(timesteps, thumb_trajectory[:, 2], label="Thumb", color=finger_colors[0])
plt.plot(timesteps, index_trajectory[:, 2], label="Index", color=finger_colors[1])
plt.plot(timesteps, middle_trajectory[:, 2], label="Middle", color=finger_colors[2])
plt.xlabel("Timestep")
plt.ylabel("Z")
plt.title("Fingertip Trajectories in z over timesteps")
plt.legend()
plt.grid()
plt.savefig("results/trajectories/fingertip_trajectories_z_controller_executed.png", format="png", dpi=300)
plt.show()

