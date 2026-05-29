import os
import subprocess

def run_headless_male_sanitization():
    # Path to Blender launcher
    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
    
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(f"Blender executable not found at specified path: {BLENDER_PATH}")

    # Isolated male pipeline workspace paths
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\male_medium_average"
    TARGET_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized\male_medium_average"
    
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Internal Blender automation script block
    blender_script = f"""
import bpy
import os

source_dir = os.path.normpath(r"{SOURCE_DIR}")
target_dir = os.path.normpath(r"{TARGET_DIR}")

fbx_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".fbx")]
print(f"\\n--- BLENDER WORKER: PROCESSING {{len(fbx_files)}} MALE FBX ASSETS ---")

for file in fbx_files:
    # Clear Blender environment cache
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import the newly generated, solid male fbx asset
    fbx_path = os.path.join(source_dir, file)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # Gather all imported component mesh segments
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if mesh_objects:
        # Deselect non-mesh elements and select all mesh pieces
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        
        # --- MIRROR FEMALE PROTOCOL: Join all elements into a single body block ---
        bpy.ops.object.join()
        
        # Export via the Wavefront OBJ engine
        base_name = os.path.splitext(file)[0]
        output_obj_path = os.path.join(target_dir, base_name + ".obj")
        
        bpy.ops.wm.obj_export(
            filepath=output_obj_path,
            export_selected_objects=True,
            export_materials=False
        )
        print(f"MALE_BAKE_SUCCESS: {{base_name}}.obj")
    else:
        print(f"WARNING: No valid mesh layers discovered inside {{file}}")
"""

    temp_script_path = "temp_blender_male_worker.py"
    with open(temp_script_path, "w", encoding="utf-8") as f:
        f.write(blender_script)

    print("[+] Launching background Blender instance for male sanitization loop...")
    command = [BLENDER_PATH, "--background", "--python", temp_script_path]
    subprocess.run(command, capture_output=False, text=False)
    
    if os.path.exists(temp_script_path):
        os.remove(temp_script_path)
        
    print("\n[STATUS] Headless male sanitization cycle finished.")

if __name__ == "__main__":
    run_headless_male_sanitization()