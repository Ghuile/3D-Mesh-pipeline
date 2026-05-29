import os
import numpy as np

def verify_female_mesh_footprint():
    SANITIZED_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized\female_medium_average"
    PARSED_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed\female_medium_average"
    
    print("=" * 75)
    print("               METAHUMAN FEMALE COHORT TOPOLOGY CHECKER")
    print("=" * 75)
    
    # Pathway A: Check existing parsed binary structures if they exist
    if os.path.exists(PARSED_DIR):
        npz_files = [f for f in os.listdir(PARSED_DIR) if f.endswith('.npz')]
        if npz_files:
            npz_files.sort()
            sample_file = npz_files[0]
            try:
                data = np.load(os.path.join(PARSED_DIR, sample_file))
                v_count = data['vertices'].shape[0]
                f_count = data['faces'].shape[0]
                print(f"[SUCCESS] Discovered pre-existing parsed array data.")
                print(f"  -> Sample Binary:            {sample_file}")
                print(f"  -> Invariant Body Vertices:  {v_count}")
                print(f"  -> Invariant Body Faces:     {f_count}")
                print("=" * 75)
                return
            except Exception as e:
                print(f"[-] Encountered reading error on binary file: {e}")

    # Pathway B: Fallback pass - Test parse the first sanitized OBJ file on the fly
    print("[-] No parsed binaries discovered. Evaluating sanitized OBJ geometry...")
    if not os.path.exists(SANITIZED_DIR):
        print(f"[ERROR] Target female folder does not exist at: {SANITIZED_DIR}")
        return
        
    obj_files = [f for f in os.listdir(SANITIZED_DIR) if f.endswith('.obj')]
    if not obj_files:
        print("[ERROR] No sanitized female .obj assets found to analyze.")
        return
        
    obj_files.sort()
    target_obj = os.path.join(SANITIZED_DIR, obj_files[0])
    print(f"[+] Inspecting structural arrangement of asset: {obj_files[0]}")
    
    current_object = None
    object_vertices = {}
    global_vertices_count = 0
    
    with open(target_obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                global_vertices_count += 1
                if current_object is not None:
                    object_vertices[current_object].append(global_vertices_count - 1)
            elif line.startswith('o ') or line.startswith('g '):
                current_object = line.split()[1]
                object_vertices[current_object] = []
                
    if not object_vertices:
        print(f"  -> Object tags unassigned. Total mesh footprint: {global_vertices_count} vertices.")
    else:
        body_object_name = max(object_vertices, key=lambda k: len(object_vertices[k]))
        isolated_v_count = len(object_vertices[body_object_name])
        print("\n[SUCCESS] Structural layers parsed successfully:")
        print(f"  -> Primary Isolated Body Block Name: '{body_object_name}'")
        print(f"  -> Invariant Body Vertices:          {isolated_v_count}")
    print("=" * 75)

if __name__ == "__main__":
    verify_female_mesh_footprint()