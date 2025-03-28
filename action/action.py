class HandController: 
    def __init__(self, env, fingertips, qpos_inds):
        """
        Initialises the hand controller with environment and fingertip identifiers
        """
        self.env = env
        self.fingertips = fingertips
        self.qpos_inds = qpos_inds
    
    def move_fingertip(self, target_position):
        """
        Moves the fingertip to the target position in a controlled manner.
        1. Lift the finger before moving horizontally (break contact)
        2. Move laterally toward the new key
        3. Lower the finger back to press the key
        """
        current_pos = self.env.physics.data.site_xpos[self.fingertips].copy()

        lift_target = current_pos.copy()

        # Lift the finger
        lift_target[2] += 0.05 # lift 5cm
        self.execute_movement(lift_target)

        # Move laterally
        self.execute_movement(target_position)

        # lower the fingertip back down to press
        press_target = target_position.copy()
        press_target[2] -= 0.02 # slightly press down
        self.execute_movement(press_target)
    
    def execute_movement(self, target_position):
        """
        Executes inverse kinematics to move to the targget position
        """

        ik_joint_config, success = self.inverse_kinematics(target_position)
        if success:
            self.env.physics.data.qpos[self.qpos_inds] = ik_joint_config
            self.env.step()