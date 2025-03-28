import mujoco
from mujoco.usd import exporter
from pxr import Sdf  # Ensure you have pxr installed for USD handling

def sanitize_path(obj_name):
    """Ensure the object name is properly formatted for USD paths."""
    if not obj_name or obj_name.strip() == "":
        obj_name = "unnamed_object"  # Assign a default name if empty

    obj_name = obj_name.strip().replace("//", "/")  # Remove double slashes
    obj_name = obj_name.replace(" ", "_")  # Replace spaces with underscores (optional)
    
    
    # Ensure the path is absolute and properly formatted
    return Sdf.Path(f"/World/Mesh_Xform_piano/{obj_name}")

def export_model(model_path):
    print("Loading model from:", model_path)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(model)

    # Create the USDExporter
    exp = exporter.USDExporter(model=model)

    duration = 5
    framerate = 60

    while d.time < duration:
        mujoco.mj_step(model, d)

        if exp.frame_count < d.time * framerate:
            # help(model)
            # Sanitize object paths before updating the scene
            for i in range(model.ngeom):
                geom_name = model.geom(i).name
                sanitized_path = sanitize_path(geom_name)
                print(f"Sanitized path: {sanitized_path}")  # Debug output
                
                if isinstance(sanitized_path, Sdf.Path):  
                    sanitized_path = str(sanitized_path)  # Convert Sdf.Path to string
                if not sanitized_path.startswith("/World"):
                    sanitized_path = "/World" + sanitized_path  # Ensure it's an absolute path
                return sanitized_path
                
            # Update the USD with a new frame
            exp.update_scene(data=d)

    # Export the USD file
    exp.save_scene(filetype="usd")
    print("Export complete!")

if __name__ == "__main__":
    model_path = "combined_model.xml"
    export_model(model_path)