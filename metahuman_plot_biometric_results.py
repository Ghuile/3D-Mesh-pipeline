import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

def generate_isolated_plots():
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    database_path = os.path.join(parsed_dir, "metahuman_extracted_features.npz")
    output_image_path = os.path.join(parsed_dir, "metahuman_prediction_evaluation.png")
    
    data_source = np.load(database_path)
    X_all = data_source['latents']
    y_all = data_source['targets']
    cohorts_all = data_source['cohorts']
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Process both branches completely isolated from each other
    for row_idx, cohort_name in enumerate(['female', 'male']):
        mask = (cohorts_all == cohort_name)
        X = X_all[mask]
        y = y_all[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        # Train MLP Head specifically for this gender branch
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000, random_state=42)
        mlp.fit(X_train, y_train)
        mlp_preds = mlp.predict(X_test)
        
        # Train GPR Head specifically for this gender branch
        gpr_kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gpr = GaussianProcessRegressor(kernel=gpr_kernel, n_restarts_optimizer=10, random_state=42)
        gpr.fit(X_train, y_train)
        gpr_preds, gpr_std = gpr.predict(X_test, return_std=True)
        
        # Subplot Column 1: MLP Evaluation
        ax_mlp = axes[row_idx, 0]
        ax_mlp.scatter(y_test, mlp_preds, color='#d95f02', alpha=0.8, edgecolors='k', s=50)
        ax_mlp.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        ax_mlp.set_title(f"{cohort_name.upper()} Branch: MLP Regressor\n$R^2$ = {r2_score(y_test, mlp_preds):.4f} | MAE = {mean_absolute_error(y_test, mlp_preds):.2f}% BFP", fontsize=11, fontweight='bold')
        ax_mlp.set_xlabel("Ground-Truth BFP (%)")
        ax_mlp.set_ylabel("Predicted BFP (%)")
        
        # Subplot Column 2: GPR Evaluation
        ax_gpr = axes[row_idx, 1]
        ax_gpr.scatter(y_test, gpr_preds, color='#1b9e77', alpha=0.8, edgecolors='k', s=50)
        ax_gpr.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        
        # Generate clean sorted bounds for uncertainty intervals
        sort_idx = np.argsort(y_test)
        ax_gpr.fill_between(
            y_test[sort_idx], 
            gpr_preds[sort_idx] - 1.96 * gpr_std[sort_idx], 
            gpr_preds[sort_idx] + 1.96 * gpr_std[sort_idx], 
            color='#1b9e77', alpha=0.15
        )
        ax_gpr.set_title(f"{cohort_name.upper()} Branch: GPR Regressor\n$R^2$ = {r2_score(y_test, gpr_preds):.4f} | MAE = {mean_absolute_error(y_test, gpr_preds):.2f}% BFP", fontsize=11, fontweight='bold')
        ax_gpr.set_xlabel("Ground-Truth BFP (%)")
        
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Clean dual-branch publication chart exported to: {output_image_path}")

if __name__ == '__main__':
    generate_isolated_plots()