import os
import subprocess

def run_headless_sanitization():
    # 1. Exact path to your Blender 5.1 launcher
    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
    
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(f"Blender executable not found at specified path: {BLENDER_PATH}")

    # Explicit raw string paths to protect Windows backslashes
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\female_medium_average"
    TARGET_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized\female_medium_average"
    
    os.makedirs(TARGET_DIR, exist_ok=True)

    # 2. Hardened internal script using strict pathing for Blender
    blender_script = f"""
import bpy
import os

source_dir = os.path.normpath(r"{SOURCE_DIR}")
target_dir = os.path.normpath(r"{TARGET_DIR}")

if not os.path.exists(source_dir):
    print(f"[ERROR] Source folder does not exist inside Blender: {{source_dir}}")

fbx_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".fbx")]
print(f"--- BLENDER WORKER: FOUND {{len(fbx_files)}} FBX FILES TO PROCESS ---")

for file in fbx_files:
    # Clear memory cache completely
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the rigged fbx container
    fbx_path = os.path.join(source_dir, file)
    print(f"Importing: {{file}}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # Select all mesh components (Body skin, clothes, parts)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if mesh_objects:
        # Deselect anything else, select all mesh elements
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        
        # Merge pieces into a single continuous body shell
        bpy.ops.object.join()
        
        # Export unified static shape via Blender 5.x C++ backend Wavefront engine
        base_name = os.path.splitext(file)[0]
        output_obj_path = os.path.join(target_dir, base_name + ".obj")
        
        bpy.ops.wm.obj_export(
            filepath=output_obj_path,
            export_selected_objects=True,
            export_materials=False
        )
        print(f"BAKE_CONFIRMED: {{base_name}}.obj")
    else:
        print(f"WARNING: No mesh geometry layers detected in {{file}}")
"""

    temp_script_path = "temp_blender_worker.py"
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(blender_script)

    print("Launching background Blender 5.1 sanitization subprocess...")
    
    # Run Blender headlessly and route output directly to the terminal screen
    command = [BLENDER_PATH, "--background", "--python", temp_script_path]
    process = subprocess.run(command, capture_output=False, text=False)
    
    if os.path.exists(temp_script_path):
        os.remove(temp_script_path)
        
    print("\n[STATUS] Headless Blender execution cycle completed.")

if __name__ == "__main__":
    run_headless_sanitization()