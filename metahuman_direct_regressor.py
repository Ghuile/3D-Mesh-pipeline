import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

def execute_direct_biometric_pipeline():
    print("==========================================================================")
    print("1. INITIALIZING DIRECT GEOMETRIC ATTENTION REGRESSION ENGINE")
    print("==========================================================================\n")
    
    # Establish absolute path limits
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    registry_path = os.path.join(parsed_dir, "metahuman_master_registry.csv")
    output_image_path = os.path.join(parsed_dir, "metahuman_publication_direct_evaluation.png")
    
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Master registry database not found at: {registry_path}")
        
    df_registry = pd.read_csv(registry_path)
    
    # Configure absolute folder name maps
    folder_mapping = {
        'female': 'female_medium_average',
        'male': 'male_medium_average'
    }
    
    # Plot configuration setting up a peer-reviewed double-column layout
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, cohort_name in enumerate(['female', 'male']):
        print(f"Processing structural coordinate arrays for [{cohort_name.upper()}] cohort...")
        cohort_df = df_registry[df_registry['cohort'] == cohort_name]
        
        X_list = []
        y_list = []
        
        for _, row in cohort_df.iterrows():
            filename = row['filename']
            actual_folder = folder_mapping[cohort_name]
            full_mesh_path = os.path.join(parsed_dir, actual_folder, filename)
            
            if os.path.exists(full_mesh_path):
                # Load the raw 3D spatial points directly
                mesh_data = np.load(full_mesh_path)
                flat_vertices = mesh_data['vertices'].flatten() # Ingest full raw structural matrix
                X_list.append(flat_vertices)
                y_list.append(row['target_body_fat_percentage'])
                
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f" -> Successfully compiled data matrix. Dimensions shape: {X.shape}")
        
        # Split into an 80/20 train/test partition matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        # Instantiate a regularized Ridge linear probe to map spatial offsets directly to BFP
        regressor_head = Ridge(alpha=1.0)
        regressor_head.fit(X_train, y_train)
        predictions = regressor_head.predict(X_test)
        
        # Compute publication scoring benchmarks
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f" -> Metrics: MAE = {mae:.4f}% BFP | R² = {r2:.4f}\n")
        
        # Render scatter charts onto the master figure canvas
        ax = axes[idx]
        ax.scatter(y_test, predictions, color='#1f77b4' if cohort_name == 'male' else '#e377c2', 
                   alpha=0.8, edgecolors='k', s=60, zorder=3, label='Estimated Body Mesh')
        
        # Draw perfect identity reference line
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect Alignment Line')
        
        ax.set_title(f"{cohort_name.upper()} Cohort: Direct Coordinate Probe\n$R^2$ = {r2:.4f} | MAE = {mae:.3f}% BFP", 
                     fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Ground-Truth Body Fat Percentage (%)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Estimated Body Fat Percentage (%)", fontsize=11)
            
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("==========================================================================")
    print(f"✅ PHASE 4 COMPLETED. High-resolution figure exported to:\n -> {output_image_path}")
    print("==========================================================================")

if __name__ == '__main__':
    execute_direct_biometric_pipeline()