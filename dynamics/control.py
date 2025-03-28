import mujoco 
import numpy as np

Kp = 1000
Kv = 2 * np.sqrt(Kp) # Derivative gain 

# the shadow hand model has "actuator_velocity_sensors", "actuator_force_sensors", mjcf elements.
def apply_force(env, sensor_name, body_name, force, torque, point, qfrc_target):
    """Apply torque to the fingers."""

    print("velocity sensors: ", env.task._hand.actuator_velocity_sensors)
    print("force sensors: ", env.task._hand.actuator_force_sensors)
    print("joint torque sensors: ", env.task._hand.joint_torque_sensors)
    print("velocity sensors from the env.physics", env.physics.model.sensor(sensor_name))
    print("reading sensordata from env.data", env.physics.data.sensordata[sensor_id])
    print("bodies from the env.model: ", env.physics.model.body(body_name))

    sensor_id = env.physics.model.sensor(sensor_name).id

    # Get the correct body ID
    bodyid = env.physics.model.body(body_name).id

    # Ensure qfrc_target is of the correct shape and writeable
    qfrc_target.setflags(write=True)  # Ensure it is writeable

    # Apply force and torque
    mujoco.mj_applyFT(
        env.physics.model._model, 
        env.physics.data._data, 
        force, 
        torque, 
        point, 
        bodyid, 
        qfrc_target
    )

    print("after force applied", env.physics.data.sensordata[sensor_id])

def control_law(env, trajectory, tcurrent, fingertip_site_name, qfrc_target):
    """Control law for the fingers. Controlling the torque when a finger press a key."""
    qpos = env.physics.data.qpos.copy()
    qvel = env.physics.data.qvel.copy()

    qpos_t, qvel_t, qacc_t = trajectory

    qacc = qacc_t(tcurrent) + Kp * (qpos_t(tcurrent) - qpos) + Kv * (qvel_t(tcurrent) - qvel)

    force_value = -400
    force_value_z = -200
    force = np.array([0, force_value, force_value_z, 0, 0, 0]) # this is for 6 dof but i have 22 dof

    mujoco.mj_jacJoint(env.physics.model._model, env.physics.data._data, qpos)
    
    fingertip_site_id = env.physics.model.site(fingertip_site_name).id
    jacp = np.zeros((3, env.physics.model.nv))
    jacr = np.zeros((3, env.physics.model.nv))
    mujoco.mj_jacSite(env.physics.model._model, env.physics.data._data, jacp, jacr, fingertip_site_id)
    J = np.vstack((jacp, jacr))
    
    force_joint = np.dot(J.T, force)

    # mj_rnea(model: mjModel, data: mjData, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray, qfrc: np.ndarray) -> None
    torques = mujoco.mj_rnea(env.physics.model._model, env.physics.data._data, qpos, qvel, qacc, qfrc_target) + force_joint

    return torques



    
