import os
import re
import numpy as np

def process_sanitized_obj(obj_path, output_dir):
    # Dictionary to track vertices per mesh object block
    current_object = None
    object_vertices = {}
    
    # We still need global face tracking because OBJ faces reference the global vertex index list
    global_vertices = []
    global_faces = []
    
    # Low-level structural block parsing
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                v_coords = [float(x) for x in line.split()[1:4]]
                global_vertices.append(v_coords)
                
                # Assign this vertex index to the active mesh object block
                if current_object is not None:
                    object_vertices[current_object].append(len(global_vertices) - 1)
                    
            elif line.startswith('o ') or line.startswith('g '):
                # Detected a new sub-mesh component boundary (e.g., body, apparel, lashes)
                current_object = line.split()[1]
                object_vertices[current_object] = []
                
            elif line.startswith('f '):
                # OBJ faces are 1-indexed, convert to 0-indexed
                face = [int(x.split('/')[0]) - 1 for x in line.split()[1:4]]
                global_faces.append(face)

    # If no explicit object tokens were found, default to treating the whole file as a single object
    if not object_vertices:
        v_arr = np.array(global_vertices, dtype=np.float32)
        f_arr = np.array(global_faces, dtype=np.int32)
    else:
        # --- STRATEGY A: ACADEMIC COMPONENT FILTERING ---
        # Find the primary body shell by locating the sub-mesh block with the largest vertex footprint
        body_object_name = max(object_vertices, key=lambda k: len(object_vertices[k]))
        body_vertex_indices = set(object_vertices[body_object_name])
        
        # Filter out faces that belong exclusively to our target body shell manifold
        body_faces = []
        for face in global_faces:
            if face[0] in body_vertex_indices and face[1] in body_vertex_indices and face[2] in body_vertex_indices:
                body_faces.append(face)
                
        # Remap the indices so the final array is contiguous and doesn't contain holes
        unique_indices = sorted(list(body_vertex_indices))
        index_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_indices)}
        
        filtered_vertices = [global_vertices[idx] for idx in unique_indices]
        filtered_faces = [[index_remap[v] for v in face] for face in body_faces]
        
        v_arr = np.array(filtered_vertices, dtype=np.float32)
        f_arr = np.array(filtered_faces, dtype=np.int32)

    # Metadata parsing from filename
    filename = os.path.basename(obj_path)
    pattern = r"h([\d\.]+)_f([\d\.]+)"
    match = re.search(pattern, filename)
    
    metadata = {
        "actual_height_scale": float(match.group(1)), 
        "actual_fat_scale": float(match.group(2))
    }
    
    # Save as compressed NumPy archive containing standalone root keys for your dataloader
    base_name = os.path.splitext(filename)[0]
    np.savez_compressed(
        os.path.join(output_dir, f"{base_name}_compressed.npz"), 
        vertices=v_arr, 
        faces=f_arr, 
        actual_height_scale=metadata["actual_height_scale"],
        actual_fat_scale=metadata["actual_fat_scale"]
    )
    return v_arr.shape

def run_extraction():
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized\male_medium_average"
    EXPORT_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed\male_medium_average"
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    print("Starting topology-invariant geometry extraction...")
    for file in os.listdir(SOURCE_DIR):
        if file.endswith('.obj'):
            v_shape = process_sanitized_obj(os.path.join(SOURCE_DIR, file), EXPORT_DIR)
            print(f"Processed {file} | Invariant Body Vertices: {v_shape[0]}")

if __name__ == "__main__":
    run_extraction()