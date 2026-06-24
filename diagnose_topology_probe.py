import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def execute_diagnostic_probe():
    print("==================================================")
    print("      RUNNING DIRECT GEOMETRIC COORDINATE PROBE   ")
    print("==================================================")
    
    base_parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    stress_parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    
    # Load baseline feature data to extract raw training targets
    baseline_features = np.load(os.path.join(base_parsed_dir, "metahuman_extracted_features.npz"), allow_pickle=True)
    y_base = baseline_features['targets']
    cohorts_base = baseline_features['cohorts']
    
    # Load stress feature data to extract testing targets
    stress_features = np.load(os.path.join(stress_parsed_dir, "metahuman_stress_extracted_features.npz"), allow_pickle=True)
    y_stress = stress_features['targets']
    cohorts_stress = stress_features['cohorts']

    # Map folders to locate raw coordinate arrays (.npz)
    baseline_folders = {'female': 'female_medium_average', 'male': 'male_medium_average'}
    stress_folders = {'female': 'female_extreme_stress', 'male': 'male_extreme_stress'}

    for target_cohort in ['female', 'male']:
        print(f"\nEvaluating raw vertex alignments for [{target_cohort.upper()}]...")
        
        # 1. Reconstruct raw X_train matrix from baseline coordinate files
        X_train_list = []
        base_cohort_dir = os.path.join(base_parsed_dir, baseline_folders[target_cohort])
        base_files = sorted([f for f in os.listdir(base_cohort_dir) if f.endswith('.npz')])
        
        for file in base_files:
            mesh_data = np.load(os.path.join(base_cohort_dir, file))
            X_train_list.append(mesh_data['vertices'].flatten())
            
        X_train = np.array(X_train_list)
        y_train = y_base[cohorts_base == target_cohort][:len(X_train)]
        
        # 2. Reconstruct raw X_test matrix from stress-test coordinate files
        X_test_list = []
        stress_cohort_dir = os.path.join(stress_parsed_dir, stress_folders[target_cohort])
        stress_files = sorted([f for f in os.listdir(stress_cohort_dir) if f.endswith('.npz')])
        
        for file in stress_files:
            mesh_data = np.load(os.path.join(stress_cohort_dir, file))
            X_test_list.append(mesh_data['vertices'].flatten())
            
        X_test = np.array(X_test_list)
        y_test = y_stress[cohorts_stress == target_cohort][:len(X_test)]
        
        # Enforce dimensionality alignment safety
        min_features = min(X_train.shape[1], X_test.shape[1])
        X_train = X_train[:, :min_features]
        X_test = X_test[:, :min_features]
        
        # Scale spatial data coordinates
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Fit direct regularized linear probe head
        linear_probe = Ridge(alpha=10.0)
        linear_probe.fit(X_train_scaled, y_train)
        predictions = linear_probe.predict(X_test_scaled)
        
        # Compute indicator scores
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"  ➔ Direct Linear Probe Metrics:")
        print(f"    - Mean Absolute Error (MAE):     {mae:.4f}% BFP")
        print(f"    - Coeff. of Determination (R²):  {r2:.4f}")

if __name__ == '__main__':
    execute_diagnostic_probe()