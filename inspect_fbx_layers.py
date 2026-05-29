import os
import open3d as o3d

def inspect_raw_fbx():
    # Target a single raw asset from your generation directory
    RAW_FBX_PATH = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\female_medium_average"
    
    if not os.path.exists(RAW_FBX_PATH) or len(os.listdir(RAW_FBX_PATH)) == 0:
        print(f"[ERROR] Raw data directory missing or empty: {RAW_FBX_PATH}")
        return

    # Grab the first raw FBX file
    fbx_files = [f for f in os.listdir(RAW_FBX_PATH) if f.lower().endswith('.fbx')]
    first_file = os.path.join(RAW_FBX_PATH, fbx_files[0])
    
    print(f"Opening raw asset for structural analysis: {fbx_files[0]}\n")
    print(f"{'Mesh Node Name / Identifier':<45} | {'Vertex Count':<12}")
    print("-" * 65)
    
    # Load the asset using the Open3D triangle model scene parser
    model = o3d.io.read_triangle_model(first_file)
    
    # Iterate through every isolated geometric component inside the FBX container
    for mesh_info in model.meshes:
        mesh_name = mesh_info.mesh_name
        mesh_geometry = mesh_info.mesh
        vertex_count = len(mesh_geometry.vertices)
        
        print(f"{mesh_name:<45} | {vertex_count:<12}")

if __name__ == "__main__":
    inspect_raw_fbx()