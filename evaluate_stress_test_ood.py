import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_ood_robustness():
    base_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    stress_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed_stress_test"
    
    # 1. Ingest clean training baseline data
    baseline_source = np.load(os.path.join(base_dir, "metahuman_extracted_features.npz"), allow_pickle=True)
    X_base = baseline_source['latents']
    y_base = baseline_source['targets']
    cohorts_base = baseline_source['cohorts']
    
    # 2. Ingest expanded stress test testing data
    stress_source = np.load(os.path.join(stress_dir, "metahuman_stress_extracted_features.npz"), allow_pickle=True)
    X_stress = stress_source['latents']
    y_stress = stress_source['targets']
    cohorts_stress = stress_source['cohorts']
    
    for target_cohort in ['female', 'male']:
        print(f"\n==================================================")
        # Check training size vs target evaluation matrix size
        print(f"   OOD STRESS EVALUATION: {target_cohort.upper()} COHORT BRANCH   ")
        print(f"==================================================")
        
        # Filter baseline for training
        idx_train = (cohorts_base == target_cohort)
        X_train, y_train = X_base[idx_train], y_base[idx_train]
        
        # Filter stress test data for testing evaluation bounds
        idx_test = (cohorts_stress == target_cohort)
        X_test, y_test = X_stress[idx_test], y_stress[idx_test]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Multi-Layer Perceptron Head
        mlp_head = MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=3000, early_stopping=True, random_state=42)
        mlp_head.fit(X_train_scaled, y_train)
        mlp_preds = mlp_head.predict(X_test_scaled)
        
        # Gaussian Process Head
        gpr_kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gpr_head = GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-5, n_restarts_optimizer=10, random_state=42)
        gpr_head.fit(X_train, y_train)
        gpr_preds, gpr_std = gpr_head.predict(X_test, return_std=True)
        
        # Compute Performance Metrics
        mlp_mae = mean_absolute_error(y_test, mlp_preds)
        mlp_r2 = r2_score(y_test, mlp_preds)
        
        gpr_mae = mean_absolute_error(y_test, gpr_preds)
        gpr_r2 = r2_score(y_test, gpr_preds)
        
        print(f"\n📈 {target_cohort.upper()} OUT-OF-DISTRIBUTION BENCHMARK RESULTS:")
        print(f"  • MLP Regressor Head:")
        print(f"    - Mean Absolute Error (MAE):     {mlp_mae:.4f}")
        print(f"    - Coeff. of Determination (R²):  {mlp_r2:.4f}")
        print(f"  • GPR Bayesian Head:")
        print(f"    - Mean Absolute Error (MAE):     {gpr_mae:.4f}")
        print(f"    - Coeff. of Determination (R²):  {gpr_r2:.4f}")
        print(f"    - Avg. Prediction Uncertainty (σ): ±{np.mean(gpr_std):.4f}")

if __name__ == '__main__':
    evaluate_ood_robustness()