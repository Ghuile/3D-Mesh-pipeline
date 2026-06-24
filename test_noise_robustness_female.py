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
    
    # Generate Gaussian noise matching the dimensionality of the flattened mesh
    noise = np.random.normal(0, sigma, size=flat_coords.shape)
    return flat_coords + noise

# ==========================================================================
# 3. EXPERIMENTAL EXECUTION LOOP
# ==========================================================================
def run_female_robustness_pipeline():
    print("==========================================================================")
    print("INITIALIZING NOISE ROBUSTNESS STRESS-TEST ENGINE: FEMALE COHORT (ADA)")
    print("==========================================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing computations on device target: {device}")
    
    # Define verified absolute paths 
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    registry_path = os.path.join(parsed_dir, "metahuman_master_registry.csv")
    output_plot_path = os.path.join(parsed_dir, "female_noise_degradation_profile.png")
    
    # Set fixed architectural attributes for female branch (Ada)
    num_vertices = 66993
    input_size = num_vertices * 3  # 200,979 input channels
    weight_file = 'fmetahuman_autoencoder_trained.pth'
    female_folder = 'female_medium_average'
    
    # Load Master Registry
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Master registry file not found at: {registry_path}")
    df_registry = pd.read_csv(registry_path)
    
    # Filter registry exclusively for the female cohort
    female_df = df_registry[df_registry['cohort'] == 'female']
    print(f" -> Detected {len(female_df)} valid entries for the Female Cohort.")
    
    # ----------------------------------------------------------------------
    # STEP A: Ingest and Compile Clean Raw Structural Coordinate Arrays
    # ----------------------------------------------------------------------
    print("\nStep A: Loading raw 3D spatial points from data directory...")
    X_raw_list = []
    y_list = []
    
    for _, row in female_df.iterrows():
        filename = row['filename']
        full_mesh_path = os.path.join(parsed_dir, female_folder, filename)
        
        if os.path.exists(full_mesh_path):
            mesh_data = np.load(full_mesh_path)
            # Pull 'vertices' key matching metahuman_direct_regressor.py implementation
            flat_vertices = mesh_data['vertices'].flatten()
            X_raw_list.append(flat_vertices)
            y_list.append(row['target_body_fat_percentage'])
            
    X_raw = np.array(X_raw_list)
    y = np.array(y_list)
    print(f" -> Done. Raw Input Data Shape: {X_raw.shape}")
    
    # ----------------------------------------------------------------------
    # STEP B: Load and Initialize Pre-trained Morphological Encoder Weights
    # ----------------------------------------------------------------------
    print("\nStep B: Instantiating DSMAE Morphological Encoder Network...")
    encoder = MorphologicalEncoder(input_dim=input_size, latent_dim=32)
    weight_path = os.path.join(weights_dir, weight_file)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Trained autoencoder weights not found at: {weight_path}")
        
    state_dict = torch.load(weight_path, map_location=device)
    # Filter prefix keys out if saved inside a lightning or top-level wrapper module
    encoder_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    if not encoder_dict: 
        encoder_dict = state_dict  
    encoder.load_state_dict(encoder_dict, strict=False)
    encoder.to(device).eval()
    print(" -> Encoder weights loaded successfully and network toggled to .eval() mode.")
    
    # ----------------------------------------------------------------------
    # STEP C: Perform Strict Train/Test Separation (80/20 Partition Matrix)
    # ----------------------------------------------------------------------
    # Split the raw data first to ensure clean training elements are separate from testing targets
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)
    print(f"\nStep C: Splitting Cohort Matrix -> Train: {X_raw_train.shape[0]} | Test: {X_raw_test.shape[0]}")
    
    # ----------------------------------------------------------------------
    # STEP D: Extract Pristine Latents to Train Downstream Regressor Heads
    # ----------------------------------------------------------------------
    print("\nStep D: Extracting clean training latents via DSMAE bottleneck...")
    train_tensor = torch.tensor(X_raw_train, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_latent_train = encoder(train_tensor).cpu().numpy()
        
    # Scale features for the MLP network head to prevent gradient explosion
    scaler = StandardScaler()
    X_latent_train_scaled = scaler.fit_transform(X_latent_train)
    
    # Instantiate MLP Head matching metahuman_regressor_bench.py specs
    print(" -> Training baseline MLP Regressor Head on clean features...")
    mlp_head = MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                            alpha=1e-4, max_iter=3000, early_stopping=True, n_iter_no_change=20, random_state=42)
    mlp_head.fit(X_latent_train_scaled, y_train)
    
    # Instantiate GPR Head matching metahuman_regressor_bench.py specs
    print(" -> Training baseline GPR Bayesian Head on clean features...")
    gpr_kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpr_head = GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-5, n_restarts_optimizer=15, random_state=42)
    gpr_head.fit(X_latent_train, y_train)
    
    # ==========================================================================
    # STEP E: RUN PERTURBATION SWEEP ON THE HELD-OUT TEST MESHES
    # ==========================================================================
    # 0.001 standard deviation noise scale = 1 mm if mesh units are centimeter scaled (adjust if scaled to meters)
    noise_levels_mm = [0, 2, 5, 10, 20]
    noise_sigmas = [0.0, 0.002, 0.005, 0.010, 0.020] 
    
    gpr_r2_scores = []
    gpr_mae_scores = []
    mlp_r2_scores = []
    mlp_mae_scores = []
    
    print("\n==================================================")
    print("   STARTING ARTIFACT PERTURBATION EXPERIMENT SWEEP ")
    print("==================================================")
    
    for mm, sigma in zip(noise_levels_mm, noise_sigmas):
        # 1. Inject noise directly into the raw test spatial positions 
        X_raw_test_noisy = inject_mesh_noise(X_raw_test, sigma)
        
        # 2. Extract out-of-distribution noisy bottleneck vectors via the frozen network
        test_tensor = torch.tensor(X_raw_test_noisy, dtype=torch.float32).to(device)
        with torch.no_grad():
            X_latent_test_noisy = encoder(test_tensor).cpu().numpy()
            
        # 3. Downstream MLP Inference Pass
        X_latent_test_noisy_scaled = scaler.transform(X_latent_test_noisy)
        mlp_preds = mlp_head.predict(X_latent_test_noisy_scaled)
        mlp_r2 = r2_score(y_test, mlp_preds)
        mlp_mae = mean_absolute_error(y_test, mlp_preds)
        
        # 4. Downstream GPR Inference Pass
        gpr_preds = gpr_head.predict(X_latent_test_noisy)
        gpr_r2 = r2_score(y_test, gpr_preds)
        gpr_mae = mean_absolute_error(y_test, gpr_preds)
        
        # Track results over intervals
        gpr_r2_scores.append(gpr_r2)
        gpr_mae_scores.append(gpr_mae)
        mlp_r2_scores.append(mlp_r2)
        mlp_mae_scores.append(mlp_mae)
        
        print(f"Noise: {mm:2d}mm | GPR R²: {gpr_r2:.4f}, MAE: {gpr_mae:.4f}% | MLP R²: {mlp_r2:.4f}, MAE: {mlp_mae:.4f}%")
        
    # ==========================================================================
    # STEP F: COMPILE ACADEMIC PLOT CHART FOR COHORT SUBMISSION
    # ==========================================================================
    print("\nStep F: Compiling high-resolution visualization figure layout...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: R² Performance Trends
    ax1.plot(noise_levels_mm, gpr_r2_scores, marker='o', color='#e377c2', linewidth=2.5, label='GPR Bayesian Head')
    ax1.plot(noise_levels_mm, mlp_r2_scores, marker='s', linestyle='--', color='#7f7f7f', linewidth=2, label='MLP Regressor Head')
    ax1.set_title("Variance Accountability vs. Optical Scanner Noise", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Simulated Scanner Sensor Distortion Jitter ($\sigma$ in mm)", fontsize=11)
    ax1.set_ylabel("Coefficient of Determination ($R^2$)", fontsize=11)
    ax1.set_ylim([-0.1, 1.05])
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, loc='lower left')
    
    # Subplot 2: MAE Deviation Trends
    ax2.plot(noise_levels_mm, gpr_mae_scores, marker='o', color='#e377c2', linewidth=2.5, label='GPR Bayesian Head')
    ax2.plot(noise_levels_mm, mlp_mae_scores, marker='s', linestyle='--', color='#7f7f7f', linewidth=2, label='MLP Regressor Head')
    ax2.set_title("Downstream Estimation Error Inflation Profiles", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Simulated Scanner Sensor Distortion Jitter ($\sigma$ in mm)", fontsize=11)
    ax2.set_ylabel("Mean Absolute Error (MAE in % BFP)", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, loc='upper left')
    
    plt.suptitle("FEMALE COHORT BRANCH (ADA): Downstream Model Performance Degradation Under Noise Stress", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n==========================================================================")
    print(f"✅ STRESS EXPERIMENT PLOT CHARTS GENERATED: \n -> {output_plot_path}")
    print("==========================================================================")

if __name__ == '__main__':
    run_female_robustness_pipeline()