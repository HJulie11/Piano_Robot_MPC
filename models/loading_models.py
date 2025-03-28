import mujoco
from mujoco import viewer
from dm_control import mjcf
import os
import numpy as np
import glfw
import time
import xml.etree.ElementTree as ET

import robopianist
from robopianist.models.hands.base import Hand, HandSide
from robopianist.models.hands.shadow_hand import ShadowHand
import robopianist.models.hands.shadow_hand as shadow_hand

# help(shadow_hand.ShadowHand)
def load_shadow_hand_model(hand_side, primitive_fingertip_collisions, restrict_wrist_yaw_range, reduced_action_space, forearm_dofs):
    shadow_hand_instance = ShadowHand(
        side=hand_side,
        primitive_fingertip_collisions=primitive_fingertip_collisions,
        restrict_wrist_yaw_range=restrict_wrist_yaw_range,
        reduced_action_space=reduced_action_space,
        forearm_dofs=forearm_dofs,
    )

    
    shadow_hand_mjcf = shadow_hand_instance.mjcf_model
    # print(shadow_hand_instance._mjcf_root.to_xml_string())
    print(type(shadow_hand_mjcf))

    return shadow_hand_mjcf

def remove_default_class_tag(xml_data):
    # Parse the XML data using ElementTree
    tree = ET.ElementTree(ET.fromstring(xml_data))
    root = tree.getroot()

    # Print the XML structure for debugging
    print("XML Structure:")
    ET.dump(root)  # Dump the entire XML tree for inspection

    # Check for namespaces in the XML (adjust the namespace if needed)
    namespaces = {'': ''}  # Adjust if there is a namespace
    # Find all <default> elements with class attribute equal to "/"
    defaults_to_remove = root.findall(".//default[@class='/']", namespaces=namespaces)
    
    # Debugging: Check if matching elements are found
    print("Defaults to remove:", defaults_to_remove)

    # Iterate through all found elements and remove them
    for default_element in defaults_to_remove:
        print("Removing element:", ET.tostring(default_element, encoding='unicode'))  # Debugging: Print element to remove
        
        # Get the parent element by iterating through root elements and matching the parent
        parent = default_element.getparent() if hasattr(default_element, 'getparent') else None
        
        if parent is not None and default_element in parent:
            parent.remove(default_element)
        else:
            print("Error: Parent element not found for removal.")

        default_element.set('class', 'none')
        print("Updated element:", ET.tostring(default_element, encoding='unicode'))  # Debugging: Print updated element

    # Convert the modified tree back to a string
    return ET.tostring(root, encoding='unicode')

def load_piano_model(piano_model_path):
    # Read the piano model XML
    with open(piano_model_path, "r") as f:
        xml_data = f.read()

    # Parse the updated XML string into a MJCF model
    piano_model = mjcf.from_xml_string(xml_data)
    print(type(piano_model))

    return piano_model



def combine_models(shadow_hand_model, piano_model):
    for child in piano_model.root._children:
        if child.tag in {"compiler", "option", "size", "visual", "statistic", "default", "extension", "custom", "asset", "worldbody", "deformable", "contact", "equality", "tendon", "actuator", "sensor", "keyframe"}:
            continue
        shadow_hand_model.root.add(child)
    # shadow_hand_model.root.add(piano_model.root)
    return shadow_hand_model

hand_side = HandSide.RIGHT
primitive_fingertip_collisions = False
restrict_wrist_yaw_range = False
reduced_action_space = False
forearm_dofs = shadow_hand._DEFAULT_FOREARM_DOFS

piano_model_path = "piano.xml"
# shadow_hand_path = "../mujoco_menagerie/shadow_hand"

shadow_hand_model = load_shadow_hand_model(hand_side, primitive_fingertip_collisions, restrict_wrist_yaw_range, reduced_action_space, forearm_dofs)
piano_model = load_piano_model(piano_model_path)

combined_model = combine_models(shadow_hand_model, piano_model)

# combined_model.save("combined_model.xml")
with open("combined_model.xml", "w") as f:
    f.write(combined_model.to_xml_string())

print("Combined model has been created.")