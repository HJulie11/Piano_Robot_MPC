import sys
sys.path.append("/Users/shjulie/Downloads/BEng_Hons_Diss_TMP-main/robopianist/robopianist")

import robopianist
from robopianist.suite.tasks.piano_with_one_shadow_hand import PianoWithOneShadowHand
import robopianist.models.hands.shadow_hand_constants as hand_consts

def _compute_ot_finger_pos(env, model, data, target_qpos, qpos_inds, finger_name):
    """Compute the desired finger position for the given joint configuration."""
    hand_finger = env.task._hand.fingers[finger_name]
    hand_finger_id = model.body("rh_shadow_hand/" + hand_finger.name).id
    data.qpos[qpos_inds] = target_qpos
    mujoco.mj_forward(model, data)
    hand_finger_pos = data.xpos[hand_finger_id]
    return hand_finger_pos


fingertip_names = [finger.name for finger in env.task._hand.fingertip_sites]

piano_key_site_names = env.task.piano._sites # ✅
key_names = [key.name for key in piano_key_site_names] # ✅
key_inds = [env.physics.model.site("piano/" + key).id for key in key_names]
key_frame_position = env.physics.bind(env.task.piano.keys).xpos[38]

fingertip_names = [finger.name for finger in env.task._hand.fingertip_sites]
fingertip_inds = [env.physics.model.site("rh_shadow_hand/" + finger).id for finger in fingertip_names]

fingertip_pos_in_key = []
for fingertip_id in fingertip_inds:
    fingertip_frame_pos = env.physics.bind(env.task._hand.fingertip_sites).xpos[fingertip_id].ravel()
    fingertip_position = env.physics.data.site_xpos[fingertip_id]
    fingertip_rotation = env.physics.data.site_xmat[fingertip_id].reshape(3, 3)
    
    pos_in_key_frame = fingertip_position + np.dot(fingertip_rotation, fingertip_frame_pos)
    fingertip_pos_in_key.append(pos_in_key_frame)
    
