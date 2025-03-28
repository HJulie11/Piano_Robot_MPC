import numpy as np
from ..constants.constraints import KINEMATIC_CONSTRAINTS

class KinematicConstraints:
    def __init__(self):
        self.constraints = KINEMATIC_CONSTRAINTS

    def get_joint_limits(self, finger_name, joint_name):
        return self.constraints["fingers"].get(finger_name, {}).get(joint_name, None)
    
    def get_wrist_constraints(self):
        return self.constraints["wrist"]
    
    def enforce_joint_limits(self, joint_positions):
        """clamp joint positions within limits"""
        for finger, joints in self.constraints["fingers"].items():
            for joint, (min_val, max_val) in joints.items():
                joint_positions[finger][joint] = np.clip(joint_positions[finger][joint], min_val, max_val)
        return joint_positions
    
    def enforce_velocity_limits(self, joint_velocities):
        max_velocity = self.constraints["velocity_limits"]["max_finger_velocity"]
        return np.clip(joint_velocities, -max_velocity, max_velocity)