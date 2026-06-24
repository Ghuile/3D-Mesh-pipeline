import os
import subprocess

def run_stress_test_sanitization():
    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
    
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(f"Blender launcher not found at: {BLENDER_PATH}")

    # Configure path maps for both cohorts
    workspace = {
        "female_extreme_stress": r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized_stress_test\female_extreme_stress",
        "male_extreme_stress": r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized_stress_test\male_extreme_stress"
    }
    
    source_root = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_stress_test"

    for cohort, target_dir in workspace.items():
        source_dir = os.path.join(source_root, cohort)
        os.makedirs(target_dir, exist_ok=True)
        
        blender_script = f"""
import bpy
import os

source_dir = os.path.normpath(r"{source_dir}")
target_dir = os.path.normpath(r"{target_dir}")

fbx_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".fbx")]
print(f"\\n--- BLENDER WORKER: SANITIZING {{len(fbx_files)}} STRESS ASSETS FOR {cohort.upper()} ---")

for file in fbx_files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    fbx_path = os.path.join(source_dir, file)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if mesh_objects:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        
        # Merge all separate component layers into a single continuous body shell
        bpy.ops.object.join()
        
        base_name = os.path.splitext(file)[0]
        output_obj_path = os.path.join(target_dir, base_name + ".obj")
        
        bpy.ops.wm.obj_export(
            filepath=output_obj_path,
            export_selected_objects=True,
            export_materials=False
        )
        print(f"SANITIZED: {{base_name}}.obj")
"""
        temp_script = "temp_stress_blender_worker.py"
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(blender_script)

        print(f"[+] Launching background Blender instance for {cohort}...")
        subprocess.run([BLENDER_PATH, "--background", "--python", temp_script])
        
        if os.path.exists(temp_script):
            os.remove(temp_script)

    print("\n[STATUS] Headless stress-test sanitization cycle completed successfully.")

if __name__ == "__main__":
    run_stress_test_sanitization()