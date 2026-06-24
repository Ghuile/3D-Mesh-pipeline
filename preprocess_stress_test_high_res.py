import os
import re
import numpy as np

def process_high_res_obj(obj_path, output_dir):
    current_object = None
    object_vertices = {}
    global_vertices = []
    global_faces = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v_coords = [float(x) for x in line.split()[1:4]]
                global_vertices.append(v_coords)
                if current_object is not None:
                    object_vertices[current_object].append(len(global_vertices) - 1)
            elif line.startswith('o ') or line.startswith('g '):
                current_object = line.split()[1]
                object_vertices[current_object] = []
            elif line.startswith('f '):
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                global_faces.append(face)

    if not object_vertices:
        v_arr = np.array(global_vertices, dtype=np.float32)
        f_arr = np.array(global_faces, dtype=np.int32)
    else:
        body_object_name = max(object_vertices, key=lambda k: len(object_vertices[k]))
        body_vertex_indices = set(object_vertices[body_object_name])
        
        body_faces = []
        for face in global_faces:
            if face[0] in body_vertex_indices and face[1] in body_vertex_indices and face[2] in body_vertex_indices:
                body_faces.append(face)
                
        unique_indices = sorted(list(body_vertex_indices))
        index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        filtered_vertices = [global_vertices[idx] for idx in unique_indices]
        filtered_faces = [[index_remap[v] for v in face] for face in body_faces]
        
        v_arr = np.array(filtered_vertices, dtype=np.float32)
        f_arr = np.array(filtered_faces, dtype=np.int32)

    filename = os.path.basename(obj_path)
    # Fixes regex bug to accurately grab targets instead of index boundaries
    pattern = r"val_h(\d+\.\d+)_f(\d+\.\d+)"
    match = re.search(pattern, filename)
    
    actual_height = float(match.group(1))
    actual_fat = float(match.group(2))
    
    base_name = os.path.splitext(filename)[0]
    np.savez_compressed(
        os.path.join(output_dir, f"{base_name}_compressed.npz"), 
        vertices=v_arr, 
        faces=f_arr, 
        actual_height_scale=actual_height,
        actual_fat_scale=actual_fat
    )
    return v_arr.shape

def run_stress_parsing():
    source_root = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized_stress_test"
    export_root = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    
    cohorts = ["female_extreme_stress", "male_extreme_stress"]
    for cohort in cohorts:
        source_dir = os.path.join(source_root, cohort)
        export_dir = os.path.join(export_root, cohort)
        os.makedirs(export_dir, exist_ok=True)
        
        if not os.path.exists(source_dir): continue
        print(f"\nParsing Topology Matrices for {cohort}...")
        
        obj_files = [f for f in os.listdir(source_dir) if f.endswith('.obj')]
        for file in obj_files:
            v_shape = process_high_res_obj(os.path.join(source_dir, file), export_dir)
            print(f"Processed {file} | Invariant Target Vertices: {v_shape[0]}")

if __name__ == "__main__":
    run_stress_parsing()