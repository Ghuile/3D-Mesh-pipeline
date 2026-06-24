import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def build_publication_stress_benchmarks():
    print("==================================================")
    print("  GENERATING FINAL OOD ROBUSTNESS EVALUATION PLOT ")
    print("==================================================")
    
    base_parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    stress_parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    output_image_path = os.path.join(base_parsed_dir, "metahuman_publication_ood_robustness.png")
    
    # Load targets and demographic tracking markers
    baseline_features = np.load(os.path.join(base_parsed_dir, "metahuman_extracted_features.npz"), allow_pickle=True)
    y_base = baseline_features['targets']
    cohorts_base = baseline_features['cohorts']
    
    stress_features = np.load(os.path.join(stress_parsed_dir, "metahuman_stress_extracted_features.npz"), allow_pickle=True)
    y_stress = stress_features['targets']
    cohorts_stress = stress_features['cohorts']

    baseline_folders = {'female': 'female_medium_average', 'male': 'male_medium_average'}
    stress_folders = {'female': 'female_extreme_stress', 'male': 'male_extreme_stress'}

    # Configure publication chart style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, target_cohort in enumerate(['female', 'male']):
        print(f"Processing raw coordinate arrays for [{target_cohort.upper()}]...")
        
        # Reconstruct X_train matrix
        X_train_list = []
        base_cohort_dir = os.path.join(base_parsed_dir, baseline_folders[target_cohort])
        base_files = sorted([f for f in os.listdir(base_cohort_dir) if f.endswith('.npz')])
        for file in base_files:
            mesh_data = np.load(os.path.join(base_cohort_dir, file))
            X_train_list.append(mesh_data['vertices'].flatten())
        X_train = np.array(X_train_list)
        y_train = y_base[cohorts_base == target_cohort][:len(X_train)]
        
        # Reconstruct X_test matrix
        X_test_list = []
        stress_cohort_dir = os.path.join(stress_parsed_dir, stress_folders[target_cohort])
        stress_files = sorted([f for f in os.listdir(stress_cohort_dir) if f.endswith('.npz')])
        for file in stress_files:
            mesh_data = np.load(os.path.join(stress_cohort_dir, file))
            X_test_list.append(mesh_data['vertices'].flatten())
        X_test = np.array(X_test_list)
        y_test = y_stress[cohorts_stress == target_cohort][:len(X_test)]
        
        # Structural alignment check
        min_features = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_features]
        X_test = X_test[:, :min_features]
        
        # Normalize and fit direct coordinate linear probe
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        linear_probe = Ridge(alpha=10.0)
        linear_probe.fit(X_train_scaled, y_train)
        predictions = linear_probe.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # Render scatter charts onto panel canvas
        ax = axes[idx]
        color_marker = '#e377c2' if target_cohort == 'female' else '#1f77b4'
        
        ax.scatter(y_test, predictions, color=color_marker, alpha=0.75, 
                   edgecolors='k', s=55, zorder=3, label='OOD Stress-Test Extremes')
        
        # Render a perfect identity reference line
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Identity Line')
        
        ax.set_title(f"{target_cohort.upper()} Cohort: Direct Coordinate Extrapolation\n$R^2$ = {r2:.4f} | MAE = {mae:.3f}% BFP", 
                     fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Ground-Truth Body Fat Percentage (%)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Estimated Body Fat Percentage (%)", fontsize=11)
            
        ax.legend(loc='upper left', frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ SUCCESS! Publication-grade chart exported directly to:\n ➔ {output_image_path}")

if __name__ == '__main__':
    build_publication_stress_benchmarks()