import os
import re
import pandas as pd

def generate_metahuman_tracking_registry(base_dir):
    """
    Scans parsed data folders, extracts actual continuous geometric float parameters,
    maps them onto realistic human BFP distributions, and writes out a master tracking CSV.
    """
    # The actual geometric bounds used during data generation
    fat_min, fat_max = 0.75, 1.60
    
    # Biological target ranges for demographic profiles
    mapping_bounds = {
        'female': {'bfp_min': 12.0, 'bfp_max': 48.0},
        'male': {'bfp_min': 5.0, 'bfp_max': 40.0}
    }
    
    registry_records = []
    
    # REGEX UPDATE: Explicitly matches the literal floating-point patterns at the end of the filename
    # Looks for "_val_h" followed by digits/decimals, and "_f" followed by digits/decimals
    filename_parser = re.compile(r'_val_h(?P<height>[0-9.]+)_f(?P<fat>[0-9.]+)')
    
    cohort_directories = {
        'female': os.path.join(base_dir, 'female_medium_average'),
        'male': os.path.join(base_dir, 'male_medium_average')
    }
    
    for cohort_label, folder_path in cohort_directories.items():
        if not os.path.exists(folder_path):
            print(f"⚠️ Directory path missing: {folder_path}")
            continue
            
        print(f"Parsing filenames for cohort branch: {cohort_label}...")
        bounds = mapping_bounds[cohort_label]
        
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.npz'):
                match = filename_parser.search(file_name)
                if match:
                    # Isolate true float scaling metrics
                    h_scale = float(match.group('height'))
                    f_scale = float(match.group('fat'))
                    
                    # Compute continuous Body Fat Percentage target
                    normalized_fat = (f_scale - fat_min) / (fat_max - fat_min)
                    calculated_bfp = bounds['bfp_min'] + (normalized_fat * (bounds['bfp_max'] - bounds['bfp_min']))
                    
                    registry_records.append({
                        'filename': file_name,
                        'cohort': cohort_label,
                        'relative_path': os.path.join(cohort_label, file_name),
                        'height_multiplier': h_scale,
                        'fat_multiplier': f_scale,
                        'target_body_fat_percentage': round(calculated_bfp, 2)
                    })
                    
    if registry_records:
        df_registry = pd.DataFrame(registry_records)
        output_dest = os.path.join(base_dir, 'metahuman_master_registry.csv')
        df_registry.to_csv(output_dest, index=False)
        print(f"\n✅ Master registry successfully updated with {len(df_registry)} entries.")
        print(f"Saved directly to: {output_dest}")
        print("\nPreview of verified calculations:")
        print(df_registry[['filename', 'height_multiplier', 'fat_multiplier', 'target_body_fat_percentage']].head(5))
    else:
        print("❌ Error: No matching metadata found. Double check filenames.")

if __name__ == '__main__':
    research_path = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_parsed"
    generate_metahuman_tracking_registry(research_path)