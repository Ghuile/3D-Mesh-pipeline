import os
import torch
import torch.nn as nn
import numpy as np

class MorphologicalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(MorphologicalEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )
    def forward(self, x):
        return self.network(x)

def export_colorized_ply(filename, vertices, normalized_gradients):
    """
    Saves a 3D PLY file strictly as a Point Cloud with vertex colors mapped 
    from sensitivity gradients. Eliminates dummy face artifacts entirely.
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write out raw vertex locations paired with Jet colormap values
        for i, v in enumerate(vertices):
            grad_val = normalized_gradients[i]
            # Smooth Jet-style color ramp conversion (Blue -> Green -> Red)
            r = int(max(0, min(255, (grad_val - 0.5) * 2 * 255 if grad_val > 0.5 else 0)))
            g = int(max(0, min(255, (1 - 2 * abs(grad_val - 0.5)) * 255)))
            b = int(max(0, min(255, (0.5 - grad_val) * 2 * 255 if grad_val < 0.5 else 0)))
            f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")

def generate_isolated_heatmap(cohort_name, specs, weights_dir, output_dir):
    print(f"\n🚀 Generating 3D Attention Heatmap for: {cohort_name.upper()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Initialize network topology matching strict cohort dimensions
    input_size = specs['vertices'] * 3
    model = MorphologicalEncoder(input_dim=input_size, latent_dim=32)
    
    # 2. Load dedicated cohort weights
    weight_path = os.path.join(weights_dir, specs['weight_file'])
    if not os.path.exists(weight_path):
        print(f"❌ Error: Missing weights file at {weight_path}")
        return
        
    state_dict = torch.load(weight_path, map_location=device)
    encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    if not encoder_dict: encoder_dict = state_dict  
    model.load_state_dict(encoder_dict, strict=False)
    model.to(device).eval()
    
    # 3. Load baseline template mesh arrays
    template_path = os.path.join(weights_dir, specs['template_file'])
    if not os.path.exists(template_path):
        print(f"❌ Error: Missing baseline template matrix at {template_path}")
        return
    
    try:
        raw_data = np.load(template_path, allow_pickle=True)
        if isinstance(raw_data, np.ndarray):
            vertices = raw_data
        else:
            vertices = raw_data['vertices'] if 'vertices' in raw_data.files else raw_data
    except Exception as e:
        print(f"⚠️ Template read warning ({e}). Processing default layout structures.")
        vertices = np.zeros((specs['vertices'], 3))

    # Reshape vertices from flat 1D to structured 3D coordinates (N, 3)
    vertices = vertices.reshape(-1, 3)

    # 4. Execute Backpropagation Gradient Sensitivity Sweep (Saliency Map)
    input_tensor = torch.tensor(vertices.flatten(), dtype=torch.float32, device=device).requires_grad_(True)
    
    latent_output = model(input_tensor.unsqueeze(0))
    
    # Isolate attention trajectory with respect to adiposity proxy dimension (Index 1)
    target_dimension = latent_output[0, 1] 
    model.zero_grad()
    target_dimension.backward()
    
    raw_gradients = input_tensor.grad.cpu().numpy().reshape(-1, 3)
    
    # 5. Process coordinate gradients into absolute vertex sensitivities
    spatial_sensitivities = np.linalg.norm(raw_gradients, axis=1)
    
    # NEW ROBUST FIX: Use percentiles to clip extreme outlier gradient spikes.
    # This prevents a single outlier vertex from squashing the color profile of the rest of the body.
    v_min = np.percentile(spatial_sensitivities, 2)
    v_max = np.percentile(spatial_sensitivities, 98)
    clipped_sensitivities = np.clip(spatial_sensitivities, v_min, v_max)
    
    # Normalize between 0.0 and 1.0 based on clipped boundaries
    norm_sensitivities = (clipped_sensitivities - v_min) / (v_max - v_min + 1e-8)
    
    # 6. Export colorized asset strictly as a Point Cloud
    output_path = os.path.join(output_dir, f"{cohort_name}_biometric_attention_heatmap.ply")
    export_colorized_ply(output_path, vertices, norm_sensitivities)
    print(f"✅ Pristine {cohort_name.upper()} surface map successfully generated.")
    print(f"   Saved to: {output_path}")

def run_split_heatmap_pipeline():
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    output_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    
    cohort_specs = {
        'female': {
            'vertices': 66993, 
            'weight_file': 'fmetahuman_autoencoder_trained.pth',
            'template_file': 'female_mean_template.npy'
        },
        'male': {
            'vertices': 66991, 
            'weight_file': 'mmetahuman_autoencoder_trained.pth',
            'template_file': 'male_mean_template.npy'
        }
    }
    
    for cohort_label, specs in cohort_specs.items():
        generate_isolated_heatmap(cohort_label, specs, weights_dir, output_dir)

if __name__ == '__main__':
    run_split_heatmap_pipeline()