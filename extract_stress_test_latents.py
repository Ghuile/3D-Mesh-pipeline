import os
import torch
import torch.nn as nn
import numpy as np
from metahuman_extract_latents import MorphologicalEncoder

def run_stress_latent_extraction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    
    # Baseline constraints used to train the original networks
    fat_min, fat_max = 0.75, 1.60
    bfp_bounds = {
        'female': {'bfp_min': 12.0, 'bfp_max': 48.0},
        'male': {'bfp_min': 5.0, 'bfp_max': 40.0}
    }
    
    cohort_specs = {
        'female_extreme_stress': {'vertices': 66993, 'weight_file': 'fmetahuman_autoencoder_trained.pth', 'cohort_label': 'female'},
        'male_extreme_stress': {'vertices': 66991, 'weight_file': 'mmetahuman_autoencoder_trained.pth', 'cohort_label': 'male'}
    }
    
    networks = {}
    for cohort, specs in cohort_specs.items():
        input_size = specs['vertices'] * 3
        model = MorphologicalEncoder(input_dim=input_size, latent_dim=32)
        weight_path = os.path.join(weights_dir, specs['weight_file'])
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=device)
            encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
            if not encoder_dict: encoder_dict = state_dict  
            model.load_state_dict(encoder_dict, strict=False)
        model.to(device).eval()
        networks[cohort] = model

    extracted_latents = []
    extracted_targets = []
    extracted_cohorts = []
    
    for cohort, specs in cohort_specs.items():
        source_dir = os.path.join(parsed_dir, cohort)
        if not os.path.exists(source_dir): continue
        
        cohort_key = specs['cohort_label']
        bounds = bfp_bounds[cohort_key]
        
        for file in os.listdir(source_dir):
            if not file.endswith('.npz'): continue
            full_mesh_path = os.path.join(source_dir, file)
            
            mesh_data = np.load(full_mesh_path)
            vertices = mesh_data['vertices'].flatten()
            
            expected_size = specs['vertices'] * 3
            if len(vertices) != expected_size:
                if len(vertices) > expected_size:
                    vertices = vertices[:expected_size]
                else:
                    vertices = np.pad(vertices, (0, expected_size - len(vertices)), 'edge')
                    
            input_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                latent_vector = networks[cohort](input_tensor).cpu().numpy().flatten()
                
            # FIXED: Linearly extrapolate raw multiplier directly into Body Fat Percentage bounds
            f_scale = mesh_data['actual_fat_scale']
            normalized_fat = (f_scale - fat_min) / (fat_max - fat_min)
            calculated_bfp = bounds['bfp_min'] + (normalized_fat * (bounds['bfp_max'] - bounds['bfp_min']))
            
            extracted_latents.append(latent_vector)
            extracted_targets.append(calculated_bfp) # Appending correct BFP targets
            extracted_cohorts.append(cohort_key)
        
    np.savez(
        os.path.join(parsed_dir, "metahuman_stress_extracted_features.npz"),
        latents=np.array(extracted_latents),
        targets=np.array(extracted_targets),
        cohorts=np.array(extracted_cohorts)
    )
    print("✅ Fixed Stress-test latent feature database generated successfully.")

if __name__ == '__main__':
    run_stress_latent_extraction()