import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_isolated_cohort(X, y, cohort_name):
    print(f"\n==================================================")
    print(f"       EVALUATING COHORT BRANCH: {cohort_name.upper()}         ")
    print(f"==================================================")
    
    # 1. Isolate and perform cohort-specific train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(f"[{cohort_name}] Dataset Split -> Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    
    # 2. Apply strict isolated Feature Scaling to prevent cross-gender blending noise
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Optimize and train the Multi-Layer Perceptron (MLP) Head
    print(f"[{cohort_name}] Fitting Multi-Layer Perceptron Regressor...")
    mlp_head = MLPRegressor(
        hidden_layer_sizes=(256, 128),  # Expanded layer width for high-dimensional resolution
        activation='relu',
        solver='adam',
        alpha=1e-4,                     # Regularization penalty to prevent flattening
        max_iter=3000,                  # Ample iteration window for absolute convergence
        early_stopping=True,            # Prevents validation stagnation
        n_iter_no_change=20,
        random_state=42
    )
    mlp_head.fit(X_train_scaled, y_train)
    mlp_preds = mlp_head.predict(X_test_scaled)
    
    # 4. Optimize and train the Gaussian Process Bayesian Regressor (GPR) Head
    print(f"[{cohort_name}] Fitting Gaussian Process Regressor...")
    # Expanding bounds and adding slight alpha noise value natively absorbs convergence instability 
    gpr_kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
    gpr_head = GaussianProcessRegressor(
        kernel=gpr_kernel, 
        alpha=1e-5,                     # Regularizes small noise to prevent abnormal optimization cliffs
        n_restarts_optimizer=15, 
        random_state=42
    )
    gpr_head.fit(X_train, y_train)
    gpr_preds, gpr_std = gpr_head.predict(X_test, return_std=True)
    
    # 5. Compute isolated performance indicators
    mlp_mae = mean_absolute_error(y_test, mlp_preds)
    mlp_r2 = r2_score(y_test, mlp_preds)
    
    gpr_mae = mean_absolute_error(y_test, gpr_preds)
    gpr_r2 = r2_score(y_test, gpr_preds)
    
    # Print results to console
    print(f"\n📈 {cohort_name.upper()} COHORT METRIC PROFILE:")
    print(f"  • MLP Architecture Head:")
    print(f"    - Mean Absolute Error (MAE):     {mlp_mae:.4f}% BFP")
    print(f"    - Coeff. of Determination (R²):  {mlp_r2:.4f}")
    print(f"  • GPR Bayesian Head:")
    print(f"    - Mean Absolute Error (MAE):     {gpr_mae:.4f}% BFP")
    print(f"    - Coeff. of Determination (R²):  {gpr_r2:.4f}")
    print(f"    - Avg. Prediction Confidence (σ): ±{np.mean(gpr_std):.4f}%")
    
    return {
        "y_test": y_test,
        "mlp_preds": mlp_preds,
        "gpr_preds": gpr_preds,
        "gpr_std": gpr_std
    }

def run_split_pipeline():
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    database_path = os.path.join(parsed_dir, "metahuman_extracted_features.npz")
    
    if not os.path.exists(database_path):
        raise FileNotFoundError("Extracted features package not detected. Execute metahuman_extract_latents.py.")
        
    data_source = np.load(database_path)
    X = data_source['latents']  
    y = data_source['targets']  
    
    # Stratify into pure female and male segments
    X_female, y_female = X[:100], y[:100]
    X_male, y_male = X[100:], y[100:]
    
    # Run evaluations independently
    female_results = evaluate_isolated_cohort(X_female, y_female, "Female")
    male_results = evaluate_isolated_cohort(X_male, y_male, "Male")
    
    # Save un-contaminated results for the visualization engine
    np.savez(
        os.path.join(parsed_dir, "separated_evaluation_results.npz"),
        female_y=female_results["y_test"], female_mlp=female_results["mlp_preds"], female_gpr=female_results["gpr_preds"],
        male_y=male_results["y_test"], male_mlp=male_results["mlp_preds"], male_gpr=male_results["gpr_preds"]
    )
    print("\n✅ Clean separated evaluation structures saved for visualization mapping.")

if __name__ == '__main__':
    run_split_pipeline()