import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ... [Keep your existing build_sparse_laplacian and apply_sparse_laplacian functions] ...

# ==========================================
# 3. MEMORY-OPTIMIZED AUTOENCODER
# ==========================================
class DisentangledMeshAE(nn.Module):
    def __init__(self, mean_template, input_dim=200979, latent_dim=32):
        super().__init__()
        self.register_buffer('mean_template', mean_template)
        # Leaner architecture to fit 6GB VRAM
        self.encoder = nn.Sequential(nn.Linear(input_dim, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.height_regressor = nn.Linear(8, 1)
        self.fat_regressor = nn.Linear(8, 1)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, input_dim))

    def forward(self, x):
        z = self.encoder(x)
        pred_h = self.height_regressor(z[:, 0:8]).squeeze(-1)
        pred_f = self.fat_regressor(z[:, 8:16]).squeeze(-1)
        return self.mean_template + self.decoder(z), pred_h, pred_f

# ==========================================
# 4. MEMORY-EFFICIENT TRAINING LOOP
# ==========================================
def train_model():
    DATA_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed\female_medium_average"
    BATCH_SIZE = 1  # Memory bottleneck fix
    ACCUMULATION_STEPS = 8 # Mimics batch size of 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ... [Load data and sparse L same as before] ...
    
    model = DisentangledMeshAE(mean_template=mean_template).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("Starting Memory-Optimized Training...")
    
    for epoch in range(50):
        optimizer.zero_grad() # Clear gradients at start of accumulation
        for i, batch in enumerate(dataloader):
            inputs, true_h, true_f = batch["vertices"].to(device), batch["height_scale"].to(device), batch["fat_scale"].to(device)
            
            recon, pred_h, pred_f = model(inputs)
            # Laplacian Loss
            lap_loss = criterion(apply_sparse_laplacian(adj, degree_sum, recon.view(-1, 66993, 3)), 
                                 apply_sparse_laplacian(adj, degree_sum, inputs.view(-1, 66993, 3)))
            
            loss = (criterion(recon, inputs) + 10.0*criterion(pred_h, true_h) + 10.0*criterion(pred_f, true_f) + 5.0*lap_loss) / ACCUMULATION_STEPS
            loss.backward()
            
            if (i + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        print(f"Epoch {epoch} complete.")