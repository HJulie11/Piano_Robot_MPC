import matplotlib.pyplot as plt
import numpy as np

def plot_traj(traj, ax=None, title=None):
    plt.figure(figsize=(8, 6))

    # if len(traj) > 0 and len(traj[0][0]) == 3:
    #     ax = fig.add_subplot(111, projection='3d')
    #     for path in traj:
    #         path = np.array(path)
    #         ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')
        
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')
    #     ax.set_title(f"3D {title}")
    # else:
    for path in traj:
        path = np.array(path[-3:-1])
        plt.plot(path[0], path[1], marker='o', linestyle='-', color='b')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f"2D {title}")
    plt.grid(True)

    plt.show()