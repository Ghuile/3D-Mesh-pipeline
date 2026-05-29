import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

def execute_biometric_benchmarking():
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    database_path = os.path.join(parsed_dir, "metahuman_extracted_features.npz")
    
    if not os.path.exists(database_path):
        raise FileNotFoundError("Missing extracted features database! Run metahuman_extract_latents.py first.")
        
    # Load feature vectors and target labels
    data_source = np.load(database_path)
    X = data_source['latents']   # Shapes matrix [200 samples, 32 latent dimensions]
    y = data_source['targets']   # Body Fat Percentages array [200 samples]
    
    # Perform an 80/20 stratified split for robust scientific validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    print(f"Dataset configurations loaded successfully.")
    print(f"Training Matrices Scale: {X_train.shape} | Evaluation Scale: {X_test.shape}\n")
    print("==========================================================================")
    print("             PHASE 4 BIOMETRIC ESTIMATION BENCHMARK RUN                   ")
    print("==========================================================================\n")
    
    # --------------------------------------------------------------------------
    # MODEL BRANCH 1: Multi-Layer Perceptron (MLP) Head
    # --------------------------------------------------------------------------
    print("Training Multi-Layer Perceptron Regressor Model...")
    mlp_head = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    mlp_head.fit(X_train, y_train)
    mlp_predictions = mlp_head.predict(X_test)
    
    mlp_mae = mean_absolute_error(y_test, mlp_predictions)
    mlp_r2 = r2_score(y_test, mlp_predictions)
    
    # --------------------------------------------------------------------------
    # MODEL BRANCH 2: Gaussian Process Regression (GPR) Head with Uncertainty Bounds
    # --------------------------------------------------------------------------
    print("Training Gaussian Process Bayesian Regressor Model...")
    gpr_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gpr_head = GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10, random_state=42)
    
    gpr_head.fit(X_train, y_train)
    gpr_predictions, gpr_std = gpr_head.predict(X_test, return_std=True)
    
    gpr_mae = mean_absolute_error(y_test, gpr_predictions)
    gpr_r2 = r2_score(y_test, gpr_predictions)
    
    # --------------------------------------------------------------------------
    # FINAL METRIC EVALUATION CONSOLE REPORT
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("             FINAL METRIC PROFILES SUMMARY                 ")
    print("="*50)
    print(f"Architectural Head A: Multi-Layer Perceptron (MLP)")
    print(f"  -> Mean Absolute Error (MAE):     {mlp_mae:.4f}% BFP")
    print(f"  -> Coefficient Determination (R²): {mlp_r2:.4f}")
    print("-"*50)
    print(f"Architectural Head B: Gaussian Process Regression (GPR)")
    print(f"  -> Mean Absolute Error (MAE):     {gpr_mae:.4f}% BFP")
    print(f"  -> Coefficient Determination (R²): {gpr_r2:.4f}")
    print(f"  -> Average Predictive Confidence Bounds (σ): ±{np.mean(gpr_std):.4f}%")
    print("="*50)

if __name__ == '__main__':
    execute_biometric_benchmarking()