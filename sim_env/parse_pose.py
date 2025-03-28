import xml.etree.ElementTree as ET
import numpy as np

def parse_pose(xml_file_path: str, keyframe_name: str) -> np.ndarray:
    # Load and parse the XML file
    tree = ET.parse(xml_file_path)  # Replace with your XML file path
    root = tree.getroot()

    # Find keyframes
    keyframes = root.find("keyframe")

    # Extract qpos from each keyframe
    qpos_dict = {}
    for key in keyframes.findall("key"):
        name = key.get("name")  # Get keyframe name
        qpos_values = np.array([float(x) for x in key.get("qpos").split()])  # Convert to ndarray
        qpos_dict[name] = qpos_values

    return qpos_dict.get(keyframe_name)

    # Example: Access qpos values
    # print(f"qpos for '{keyframe_name}':", qpos_dict.get(keyframe_name))