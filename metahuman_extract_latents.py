import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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

def run_latent_extraction():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    registry_path = os.path.join(parsed_dir, "metahuman_master_registry.csv")
    
    df_registry = pd.read_csv(registry_path)
    
    folder_mapping = {'female': 'female_medium_average', 'male': 'male_medium_average'}
    cohort_specs = {
        'female': {'vertices': 66993, 'weight_file': 'fmetahuman_autoencoder_trained.pth'},
        'male': {'vertices': 66991, 'weight_file': 'mmetahuman_autoencoder_trained.pth'}
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
    extracted_cohorts = [] # Added tracking array
    
    for idx, row in df_registry.iterrows():
        cohort = row['cohort']
        filename = row['filename']
        actual_folder = folder_mapping[cohort]
        full_mesh_path = os.path.join(parsed_dir, actual_folder, filename)
        
        if not os.path.exists(full_mesh_path): continue
            
        mesh_data = np.load(full_mesh_path)
        vertices = mesh_data['vertices'].flatten()
        input_tensor = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            latent_vector = networks[cohort](input_tensor).cpu().numpy().flatten()
            
        extracted_latents.append(latent_vector)
        extracted_targets.append(row['target_body_fat_percentage'])
        extracted_cohorts.append(cohort)
        
    np.savez(
        os.path.join(parsed_dir, "metahuman_extracted_features.npz"),
        latents=np.array(extracted_latents),
        targets=np.array(extracted_targets),
        cohorts=np.array(extracted_cohorts) # Saved cohort tracking arrays cleanly
    )
    print("✅ Latent database serialized successfully.")

if __name__ == '__main__':
    run_latent_extraction()