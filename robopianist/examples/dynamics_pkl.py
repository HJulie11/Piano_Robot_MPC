import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the dynamics data
with open("dynamics_data.pkl", "rb") as f:
    data = pickle.load(f)

# Extract fingertip velocities for the index finger (finger 1)
fingertip_velocities = np.array([step[1] for step in data["fingertip_velocities"]])  # Finger 1
velocity_magnitudes = np.linalg.norm(fingertip_velocities, axis=1)

# Plot fingertip velocity over time
plt.figure(figsize=(10, 6))
plt.plot(velocity_magnitudes, label="Index Finger Velocity")
plt.xlabel("Timestep")
plt.ylabel("Fingertip Velocity (m/s)")
plt.title("Fingertip Velocity Over Time")
plt.legend()
plt.grid()
plt.savefig("fingertip_velocity.png")
plt.show()

# Compute total energy consumption
total_energy = np.sum(data["energy"])
print(f"Total energy consumption: {total_energy:.2f} J")

# Analyze contact forces for the index finger
contact_forces = np.array([step[1] for step in data["contact_forces"]])
force_magnitudes = np.linalg.norm(contact_forces, axis=1)
plt.figure(figsize=(10, 6))
plt.plot(force_magnitudes, label="Index Finger Contact Force")
plt.xlabel("Timestep")
plt.ylabel("Contact Force (N)")
plt.title("Contact Force Over Time")
plt.legend()
plt.grid()
plt.savefig("contact_force.png")
plt.show()