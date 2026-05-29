import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

def generate_publication_plots():
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    data_path = os.path.join(parsed_dir, "separated_evaluation_results.npz")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Missing split evaluation data! Execute metahuman_regressor_bench.py first.")
        
    data = np.load(data_path)
    
    # Set up publication styling aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    
    cohorts = {
        'female': {
            'y_true': data['female_y'], 'mlp': data['female_mlp'], 'gpr': data['female_gpr'],
            'color_mlp': '#d62728', 'color_gpr': '#bcbd22', 'row': 0, 'title': 'Female Cohort'
        },
        'male': {
            'y_true': data['male_y'], 'mlp': data['male_mlp'], 'gpr': data['male_gpr'],
            'color_mlp': '#1f77b4', 'color_gpr': '#2ca02c', 'row': 1, 'title': 'Male Cohort'
        }
    }
    
    for key, c in cohorts.items():
        y_true = c['y_true']
        row = c['row']
        
        # --- Multi-Layer Perceptron Subplot ---
        ax_mlp = axes[row, 0]
        r2_mlp = r2_score(y_true, c['mlp'])
        mae_mlp = mean_absolute_error(y_true, c['mlp'])
        
        ax_mlp.scatter(y_true, c['mlp'], color=c['color_mlp'], alpha=0.7, edgecolors='k', label='Predicted')
        ax_mlp.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1.5, label='Ideal Alignment')
        ax_mlp.set_title(f"{c['title']} - MLP Regressor Head")
        ax_mlp.set_xlabel("Ground Truth BFP (%)")
        ax_mlp.set_ylabel("Predicted BFP (%)")
        ax_mlp.grid(True, linestyle=':', alpha=0.6)
        ax_mlp.text(0.05, 0.85, f"$R^2$: {r2_mlp:.4f}\nMAE: {mae_mlp:.4f}%", 
                    transform=ax_mlp.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        ax_mlp.legend(loc='lower right')
        
        # --- Gaussian Process Subplot ---
        ax_gpr = axes[row, 1]
        r2_gpr = r2_score(y_true, c['gpr'])
        mae_gpr = mean_absolute_error(y_true, c['gpr'])
        
        ax_gpr.scatter(y_true, c['gpr'], color=c['color_gpr'], alpha=0.7, edgecolors='k', label='Predicted')
        ax_gpr.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=1.5, label='Ideal Alignment')
        ax_gpr.set_title(f"{c['title']} - Gaussian Process Bayesian Head")
        ax_gpr.set_xlabel("Ground Truth BFP (%)")
        ax_gpr.set_ylabel("Predicted BFP (%)")
        ax_gpr.grid(True, linestyle=':', alpha=0.6)
        ax_gpr.text(0.05, 0.85, f"$R^2$: {r2_gpr:.4f}\nMAE: {mae_gpr:.4f}%", 
                    transform=ax_gpr.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        ax_gpr.legend(loc='lower right')

    plt.tight_layout()
    output_img = os.path.join(parsed_dir, "metahuman_separated_prediction_evaluation.jpg")
    plt.savefig(output_img, dpi=300)
    plt.close()
    print(f"✅ Clean, non-contaminated publication graphics successfully saved to:\n   {output_img}")

if __name__ == '__main__':
    generate_publication_plots()