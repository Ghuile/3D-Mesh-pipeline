import os
import sys
import bpy

def run_dataset_validation():
    # Find the output folder where our 3D meshes live
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "synthetic_output")
    
    if not os.path.exists(output_dir):
        print(f"[-] Error: Cannot find the synthetic_output folder at {output_dir}")
        return

    # Get a list of all .obj files inside the folder
    mesh_files = [f for f in os.listdir(output_dir) if f.endswith('.obj')]
    
    if not mesh_files:
        print("[-] Error: No generated .obj files found to validate.")
        return

    print(f"[+] Found {len(mesh_files)} files. Starting geometric health check...")
    
    # Define our strict expected baseline vertex count
    EXPECTED_VERTEX_COUNT = None 
    passed_count = 0
    failed_count = 0

    for file_name in mesh_files:
        file_path = os.path.join(output_dir, file_name)
        
        # Clear Blender and import the generated mesh
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.wm.obj_import(filepath=file_path)
        
        # Find the imported mesh object
        mesh_obj = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                mesh_obj = obj
                break
                
        if not mesh_obj:
            print(f"[-] CRITICAL FAILURE: {file_name} could not be read as a 3D mesh.")
            failed_count += 1
            continue

        # --- CHECK 1: VERTEX INVARIANCE CHECK ---
        current_vertex_count = len(mesh_obj.data.vertices)
        
        # Lock in the first file's vertex count as our absolute target baseline
        if EXPECTED_VERTEX_COUNT is None:
            EXPECTED_VERTEX_COUNT = current_vertex_count
            print(f"[I] Baseline vertex count established at: {EXPECTED_VERTEX_COUNT} vertices.")

        if current_vertex_count != EXPECTED_VERTEX_COUNT:
            print(f"[-] MONSTER DETECTED ({file_name}): Vertex structure broke! Has {current_vertex_count} vertices.")
            failed_count += 1
            continue

        # --- CHECK 2: BOUNDING BOX ANATOMY CHECK ---
        # Calculate dimensions: [Width (X), Depth (Y), Height (Z)]
        dimensions = mesh_obj.dimensions
        width = dimensions[0]
        height = dimensions[2]

        # Human Proportions Rule: A person cannot be wider than they are tall!
        if width >= height:
            print(f"[-] MONSTER DETECTED ({file_name}): Extreme horizontal distortion! Width ({width:.2f}m) >= Height ({height:.2f}m).")
            failed_count += 1
            continue

        # If it passes both checks, it is a clean human shape
        passed_count += 1

    print("\n--- FINAL DATASET METRIC LOGS ---")
    print(f"[+] Total Verified Clean Human Shapes: {passed_count}/{len(mesh_files)}")
    print(f"[!] Total Flagged Anomalies/Monsters: {failed_count}")
    
    if failed_count == 0:
        print("[+] SUCCESS: Dataset is academically sound, stable, and ready for deep learning training!")

if __name__ == "__main__":
    run_dataset_validation()