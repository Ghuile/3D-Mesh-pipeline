import os
import re
import numpy as np
import open3d as o3d

def parse_biometric_metadata(filename: str) -> dict:
    """
    Parses step matrix indices and continuous physical metrics 
    directly from the stress-test file naming convention.
    """
    metadata = {}
    complex_pattern = r"step_h(\d+)_f(\d+)_val_h(\d+\.\d+)_f(\d+\.\d+)"
    complex_match = re.search(complex_pattern, filename)
    
    if complex_match:
        metadata["height_index"] = int(complex_match.group(1))
        metadata["fat_index"] = int(complex_match.group(2))
        metadata["actual_height_scale"] = float(complex_match.group(3))
        metadata["actual_fat_scale"] = float(complex_match.group(4))
    return metadata

def process_single_mesh(fbx_path: str, output_dir: str) -> tuple:
    """
    Opens the stress-test FBX, isolates the core Body skin mesh layer 
    by vertex bounds, and compresses spatial matrices to .npz format.
    """
    filename = os.path.basename(fbx_path)
    metadata = parse_biometric_metadata(filename)
    
    # Read the FBX model natively using Open3D's asset engine
    model = o3d.io.read_triangle_model(fbx_path)
    body_mesh = None
    
    # Locate the active Body mesh shell by isolating vertex density bounds
    for mesh_info in model.meshes:
        vertex_count = len(mesh_info.mesh.vertices)
        if 8000 <= vertex_count <= 15000:
            body_mesh = mesh_info.mesh
            break
            
    if body_mesh is None:
        raise ValueError(f"No geometry layer matching the Body vertex count profile found in {filename}")
        
    # Cast Open3D structures directly to high-performance NumPy arrays
    vertices = np.asarray(body_mesh.vertices, dtype=np.float32)  # Coordinate Matrix
    faces = np.asarray(body_mesh.triangles, dtype=np.int32)     # Face Element Matrix
    
    base_name = os.path.splitext(filename)[0]
    compressed_path = os.path.join(output_dir, f"{base_name}_compressed.npz")
    
    # Save clean arrays and paired metadata metrics into an isolated zipped file
    np.savez_compressed(
        compressed_path, 
        vertices=vertices, 
        faces=faces,
        **metadata
    )
    return vertices.shape, faces.shape, compressed_path

def run_preprocessing_pipeline(source_root_dir: str, export_root_dir: str):
    # Set to your new standalone stress folders
    cohorts = ["female_extreme_stress", "male_extreme_stress"]
    
    for cohort in cohorts:
        source_cohort_dir = os.path.join(source_root_dir, cohort)
        export_cohort_dir = os.path.join(export_root_dir, cohort)
        os.makedirs(export_cohort_dir, exist_ok=True)
        
        if not os.path.exists(source_cohort_dir):
            print(f"Skipping {cohort}: Directory not found.")
            continue
            
        print(f"\nProcessing Isolated Stress-Test Cohort: {cohort}...")
        
        for file in os.listdir(source_cohort_dir):
            if file.lower().endswith(".fbx"):
                full_fbx_path = os.path.join(source_cohort_dir, file)
                try:
                    v_shape, f_shape, _ = process_single_mesh(full_fbx_path, export_cohort_dir)
                    print(f" -> Processed {file} | Vertices: {v_shape[0]} | Faces: {f_shape[0]}")
                except Exception as e:
                    print(f" [ERROR] Failed to process {file}: {str(e)}")

if __name__ == "__main__":
    # Isolated folders to safeguard baseline data matrices
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_stress_test"
    EXPORT_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    
    run_preprocessing_pipeline(SOURCE_DIR, EXPORT_DIR)