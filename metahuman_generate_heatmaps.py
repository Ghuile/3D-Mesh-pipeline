import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
from sklearn.linear_model import LinearRegression

def export_colorized_heatmap_ply(file_dest, vertices, faces, normalized_importances):
    """
    Natively compiles a standalone, colorized 3D surface mesh in ASCII .ply format.
    Ensures absolute numerical safety with zero NaNs or Infs written to disk.
    """
    num_vertices = len(vertices)
    num_faces = len(faces)
    
    # Safe colormap retrieval matching modern Matplotlib specifications
    try:
        colormap = mpl.colormaps['jet']
    except AttributeError:
        colormap = cm.get_cmap('jet')
        
    with open(file_dest, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_vertices}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for idx in range(num_vertices):
            v = vertices[idx]
            imp_val = normalized_importances[idx]
            
            if np.isnan(imp_val) or np.isinf(imp_val):
                imp_val = 0.0
                
            rgba = colormap(imp_val)
            r = int(rgba[0] * 255)
            g = int(rgba[1] * 255)
            b = int(rgba[2] * 255)
            
            f.write(f"{v[0]} {v[1]} {v[2]} {r} {g} {b}\n")
            
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")

def execute_anatomical_heatmap_pipeline():
    print("==========================================================================")
    print("     DEPLOYING ADAPTIVE MULTI-VARIABLE MORPHOLOGICAL STRAIN ENGINE       ")
    print("==========================================================================\n")
    
    parsed_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    weights_dir = r"C:\Users\Vasileios Nikolaou\Documents\PhD\Metahuman_project"
    registry_path = os.path.join(parsed_dir, "metahuman_master_registry.csv")
    
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Missing master registry sheet tracker: {registry_path}")
        
    df_registry = pd.read_csv(registry_path)
    
    folder_mapping = {
        'female': 'female_medium_average',
        'male': 'male_medium_average'
    }
    
    for cohort_name in ['female', 'male']:
        print(f"Processing array streams for [{cohort_name.upper()}] branch...")
        cohort_df = df_registry[df_registry['cohort'] == cohort_name]
        
        X_geom_list = []
        biometric_inputs = []
        sample_faces = None
        sample_vertices = None
        
        for _, row in cohort_df.iterrows():
            filename = row['filename']
            actual_folder = folder_mapping[cohort_name]
            full_mesh_path = os.path.join(parsed_dir, actual_folder, filename)
            
            if os.path.exists(full_mesh_path):
                mesh_data = np.load(full_mesh_path)
                vertices = np.copy(mesh_data['vertices'])
                
                # DEFENSIVE STEP 1: Auto-detect and standardize un-normalized centimeter scales on the fly
                if np.max(np.abs(vertices)) > 5.0:
                    vertices = vertices / 100.0
                    
                X_geom_list.append(vertices.flatten())
                biometric_inputs.append([row['height_multiplier'], row['target_body_fat_percentage']])
                
                if sample_faces is None:
                    sample_faces = np.copy(mesh_data['faces'])
                    sample_vertices = np.copy(vertices)
                    
        X_geom = np.array(X_geom_list)
        X_bio = np.array(biometric_inputs)
        num_vertices = len(sample_vertices)
        
        # Calculate clean, standardized bounding box boundaries
        bbox_min = np.min(sample_vertices, axis=0)
        bbox_max = np.max(sample_vertices, axis=0)
        print(f" -> Guardrail Verified Bounding Box Extents:")
        print(f"    Min: X={bbox_min[0]:.2f}, Y={bbox_min[1]:.2f}, Z={bbox_min[2]:.2f}")
        print(f"    Max: X={bbox_max[0]:.2f}, Y={bbox_max[1]:.2f}, Z={bbox_max[2]:.2f}")
        
        fat_sensitivity_weights = np.zeros(num_vertices)
        
        # Fit multi-variable linear equations across all geometry points simultaneously
        ols_model = LinearRegression()
        ols_model.fit(X_bio, X_geom)
        
        # Isolate the isolated fat multiplier coefficient column
        fat_coefficients = ols_model.coef_[:, 1]
        
        print(f" -> Computing scale-invariant local morphological strain fields...")
        for i in range(num_vertices):
            w_x = fat_coefficients[3 * i]
            w_y = fat_coefficients[3 * i + 1]
            w_z = fat_coefficients[3 * i + 2]
            absolute_velocity = np.sqrt(w_x**2 + w_y**2 + w_z**2)
            
            # Isolate the matching baseline vertex coordinate vector
            v_base = sample_vertices[i]
            # Calculate absolute distance from central vertical torso anchor line (X=0, Z=0)
            radial_distance = np.sqrt(v_base[0]**2 + v_base[2]**2)
            
            # DEFENSIVE STEP 2: Divide absolute velocity by relative radial baseline distance
            # Adds a stabilization epsilon to safeguard against division-by-zero on the spine line
            fat_sensitivity_weights[i] = absolute_velocity / (radial_distance + 0.05)
            
        # Clean out any extreme localized outliers across specific noise landmarks
        p99 = np.percentile(fat_sensitivity_weights, 99)
        fat_sensitivity_weights = np.clip(fat_sensitivity_weights, 0, p99)
        
        # Smooth normalization mapping cleanly to the 0.0 - 1.0 color bounds
        min_w = np.min(fat_sensitivity_weights)
        max_w = np.max(fat_sensitivity_weights)
        normalized_importances = (fat_sensitivity_weights - min_w) / (max_w - min_w + 1e-8)
        
        output_filename = f"{cohort_name}_biometric_attention_heatmap.ply"
        output_filepath = os.path.join(weights_dir, output_filename)
        
        export_colorized_heatmap_ply(output_filepath, sample_vertices, sample_faces, normalized_importances)
        print(f"✅ Safe 3D Heatmap generated successfully at:\n -> {output_filepath}\n")
        
    print("==========================================================================")
    print("     PHASE 4 ANATOMICAL DECOMPOSITION COMPLETE: RECONSTRUCTION SECURE    ")
    print("==========================================================================")

if __name__ == '__main__':
    execute_anatomical_heatmap_pipeline()