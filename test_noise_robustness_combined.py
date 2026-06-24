import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================================================
# 1. ARCHITECTURE DEFINITION (Matches metahuman_extract_latents.py exactly)
# ==========================================================================
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

# ==========================================================================
# 2. NOISE INJECTION ENGINE (Simulates real-world scanner jitter)
# ==========================================================================
def inject_mesh_noise(flat_coords, sigma):
    """
    Injects random high-frequency Gaussian coordinate jitter to simulate 
    optical 3D scanner surface noise artifacts.
    """
    if sigma == 0.0:
        return flat_coords.copy()
    
    noise = np.random.normal(0, sigma, size=flat_coords.shape)
    return flat_coords + noise

# ==========================================================================
# 3. COMBINED MASTER EXECUTION PIPELINE
# ==========================================================================
def run_combined_robustness_pipeline():
    print("==========================================================================")
    print("INITIALIZING UNIFIED NOISE ROBUSTNESS PIPELINE: DUAL-BRANCH COHORTS")
    print("==========================================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing computations on device target: {device}")
    
    # Establish absolute verified pathing parameters
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    registry_path = os.path.join(parsed_dir, "metahuman_master_registry.csv")
    output_png_path = os.path.join(parsed_dir, "metahuman_unified_noise_robustness.png")
    
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Master registry database file missing at: {registry_path}")
    df_registry = pd.read_csv(registry_path)
    
    # Structural configuration maps matching both archetype branches cleanly
    cohort_configurations = {
        'female': {
            'vertices': 66993,
            'weight_file': 'fmetahuman_autoencoder_trained.pth',
            'folder': 'female_medium_average',
            'color': '#e377c2',
            'display_name': 'Female Cohort (Ada)'
        },
        'male': {
            'vertices': 66991,
            'weight_file': 'mmetahuman_autoencoder_trained.pth',
            'folder': 'male_medium_average',
            'color': '#1f77b4',
            'display_name': 'Male Cohort (Aoi)'
        }
    }
    
    # Perturbation parameter limits (assuming mesh is scaled in centimeters)
    noise_levels_mm = [0, 2, 5, 10, 20]
    noise_sigmas = [0.0, 0.002, 0.005, 0.010, 0.020]
    
    # Storage dictionary to hold output metrics for plotting
    results_history = {}
    
    # Iterate sequentially through both independent topological branches
    for cohort_key, specs in cohort_configurations.items():
        print(f"\n------------------------------------------------------------------")
        # Ensure clean text representation tracking plural variables matching formatting constraints
        print(f"PROCESSING EXPERIMENTAL CYCLES FOR: {specs['display_name'].upper()}")
        print(f"------------------------------------------------------------------")
        
        cohort_df = df_registry[df_registry['cohort'] == cohort_key]
        input_size = specs['vertices'] * 3
        
        # Load raw localized mesh array structures
        X_raw_list = []
        y_list = []
        
        for _, row in cohort_df.iterrows():
            filename = row['filename']
            full_mesh_path = os.path.join(parsed_dir, specs['folder'], filename)
            
            if os.path.exists(full_mesh_path):
                mesh_data = np.load(full_mesh_path)
                flat_vertices = mesh_data['vertices'].flatten()
                X_raw_list.append(flat_vertices)
                y_list.append(row['target_body_fat_percentage'])
                
        X_raw = np.array(X_raw_list)
        y = np.array(y_list)
        print(f" -> Mesh arrays compiled. Data dimensions: {X_raw.shape}")
        
        # Initialize isolated Morphological Network model weights to prevent cross-bleeding shape errors
        encoder = MorphologicalEncoder(input_dim=input_size, latent_dim=32)
        weight_path = os.path.join(weights_dir, specs['weight_file'])
        
        state_dict = torch.load(weight_path, map_location=device)
        encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        if not encoder_dict: 
            encoder_dict = state_dict  
        encoder.load_state_dict(encoder_dict, strict=False)
        encoder.to(device).eval()
        
        # Enforce an isolated 80/20 partition matrix split
        X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)
        
        # Extract clean baseline latents to fit downstream regressors
        train_tensor = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
        with torch.no_grad():
            X_latent_train = encoder(train_tensor).cpu().numpy()
            
        scaler = StandardScaler()
        X_latent_train_scaled = scaler.fit_transform(X_latent_train)
        
        # Optimize and fit baseline network heads
        mlp_head = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                                alpha=1e-4, max_iter=3000, early_stopping=True, n_iter_no_change=20, random_state=42)
        mlp_head.fit(X_latent_train_scaled, y_train)
        
        gpr_kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gpr_head = GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-5, n_restarts_optimizer=15, random_state=42)
        gpr_head.fit(X_latent_train, y_train)
        
        # Initialize internal historical metrics tracking dictionaries
        cohort_metrics = {'gpr_r2': [], 'gpr_mae': [], 'mlp_r2': [], 'mlp_mae': []}
        
        # Execute the perturbation sweeps over noisy out-of-distribution matrices
        for mm, sigma in zip(noise_levels_mm, noise_sigmas):
            X_raw_test_noisy = inject_mesh_noise(X_raw_test, sigma)
            
            test_tensor = torch.tensor(X_raw_test_noisy, dtype=torch.float32).to(device)
            with torch.no_grad():
                X_latent_test_noisy = encoder(test_tensor).cpu().numpy()
                
            # Run inference evaluations
            X_latent_test_noisy_scaled = scaler.transform(X_latent_test_noisy)
            mlp_preds = mlp_head.predict(X_latent_test_noisy_scaled)
            gpr_preds = gpr_head.predict(X_latent_test_noisy)
            
            cohort_metrics['gpr_r2'].append(r2_score(y_test, gpr_preds))
            cohort_metrics['gpr_mae'].append(mean_absolute_error(y_test, gpr_preds))
            cohort_metrics['mlp_r2'].append(r2_score(y_test, mlp_preds))
            cohort_metrics['mlp_mae'].append(mean_absolute_error(y_test, mlp_preds))
            
        # Log completed loop metrics array into memory
        results_history[cohort_key] = cohort_metrics
        print(f" -> Successfully finished stress loop evaluation sweeps for {cohort_key} branch.")

    # ==========================================================================
    # 4. UNIFIED VISUALIZATION GRAPH COMPILER (Publication-Grade 2x2 Layout)
    # ==========================================================================
    print("\nStep 4: Compiling high-resolution consolidated visual subplots matrix...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    
    # Map row indices to match cohort loops explicitly
    row_mapping = {'female': 0, 'male': 1}
    
    for cohort_key, metrics in results_history.items():
        row = row_mapping[cohort_key]
        specs = cohort_configurations[cohort_key]
        
        # Column 1 Subplots: R² Score Evolution Curves
        ax_r2 = axes[row, 0]
        ax_r2.plot(noise_levels_mm, metrics['gpr_r2'], marker='o', color=specs['color'], linewidth=2.5, label='GPR Bayesian Head')
        ax_r2.plot(noise_levels_mm, metrics['mlp_r2'], marker='s', linestyle='--', color='#7f7f7f', linewidth=2, label='MLP Regressor Head')
        ax_r2.set_title(f"{specs['display_name']}: Variance Accountability ($R^2$)", fontsize=11, fontweight='bold')
        ax_r2.set_xlabel("Simulated Sensor Jitter ($\sigma$ in mm)", fontsize=10)
        ax_r2.set_ylabel("Coefficient of Determination ($R^2$)", fontsize=10)
        ax_r2.set_ylim([-0.05, 1.05])
        ax_r2.grid(True, linestyle=':', alpha=0.6)
        ax_r2.legend(frameon=True, loc='lower left')
        
        # Column 2 Subplots: MAE Score Inflation Curves
        ax_mae = axes[row, 1]
        ax_mae.plot(noise_levels_mm, metrics['gpr_mae'], marker='o', color=specs['color'], linewidth=2.5, label='GPR Bayesian Head')
        ax_mae.plot(noise_levels_mm, metrics['mlp_mae'], marker='s', linestyle='--', color='#7f7f7f', linewidth=2, label='MLP Regressor Head')
        ax_mae.set_title(f"{specs['display_name']}: Estimation Error Inflation (MAE)", fontsize=11, fontweight='bold')
        ax_mae.set_xlabel("Simulated Sensor Jitter ($\sigma$ in mm)", fontsize=10)
        ax_mae.set_ylabel("Mean Absolute Error (MAE in % BFP)", fontsize=10)
        ax_mae.grid(True, linestyle=':', alpha=0.6)
        ax_mae.legend(frameon=True, loc='upper left')
        
    plt.suptitle("CPSI 2026 Submission 108: Parametric Model Degradation Metrics Under Spatial Coordinate Stress", 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n==========================================================================")
    print(f"✅ PIPELINE MATRIX COMPLETED. Master graphic layout exported to:\n -> {output_png_path}")
    print("==========================================================================")

if __name__ == '__main__':
    run_combined_robustness_pipeline()