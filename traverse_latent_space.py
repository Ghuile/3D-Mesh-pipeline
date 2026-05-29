import os
import numpy as np
import torch
import trimesh
from train_3dae import DisentangledMeshAE

def execute_latent_traversal():
    WEIGHTS_PATH = "disentangled_mesh_ae.pth"
    SAMPLE_MESH = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed\female_medium_average\step_h0_f0_val_h0.85_f0.75_compressed.npz"
    OUTPUT_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\traversal_outputs"
    
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dummy_template = torch.zeros(30651)
    model = DisentangledMeshAE(mean_template=dummy_template)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    with np.load(SAMPLE_MESH) as data:
        raw_vertices = data['vertices']
        faces = data['faces']
        
        centroid = np.mean(raw_vertices, axis=0)
        centered_vertices = raw_vertices - centroid
        
        # Normalize reference mesh to meter scale to match training parameters
        normalized_vertices = centered_vertices / 100.0
        inputs = torch.tensor(normalized_vertices, dtype=torch.float32).flatten().unsqueeze(0).to(device)

    print("[LATENT TRACE] Encoding normalized reference mesh...")
    with torch.no_grad():
        z_baseline = model.encoder(inputs)
    
    # Standard traversal step sizes
    height_tweaks = {
        "short_deformed": -0.1, 
        "baseline_normal": 0.0, 
        "tall_deformed": 0.1
    }

    print("[LATENT TRACE] Executing traversal step changes...")
    for label, scalar_shift in height_tweaks.items():
        z_manipulated = z_baseline.clone()
        z_manipulated[0, 0:8] += scalar_shift
        
        with torch.no_grad():
            delta_v = model.decoder(z_manipulated)
            reconstructed_vector = model.mean_template + delta_v
            
        # Reshape the output tensor back to [10217, 3]
        output_vertices_meters = reconstructed_vector.squeeze(0).cpu().numpy().reshape(10217, 3)
        
        # CONVERSION FIX: Re-scale back to centimeters for accurate 3D viewport rendering
        output_vertices_cm = output_vertices_meters * 100.0
        
        export_mesh = trimesh.Trimesh(vertices=output_vertices_cm, faces=faces)
        output_path = os.path.join(OUTPUT_DIR, f"traversal_{label}.ply")
        export_mesh.export(output_path)
        print(f" -> Exported clean asset: {output_path}")

    print("\n[STATUS] Traversal pipeline completed successfully.")

if __name__ == "__main__":
    execute_latent_traversal()