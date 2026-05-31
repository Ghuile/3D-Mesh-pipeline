import os
import sys
import math
import bpy

def generate_biologically_accurate_anchors():
    # Establish local workspace paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_fbx_path = os.path.join(script_dir, "metahuman_base.fbx")
    output_dir = os.path.join(script_dir, "phenotype_test_anchors")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(base_fbx_path):
        print(f"[-] Error: metahuman_base.fbx is missing from {script_dir}")
        return

    # Define the 4 target research phenotypes using accurate scaling recipes
    phenotype_recipes = [
        {"name": "obese_male",      "height_scale": 1.02, "mass_mod": 0.45, "dist_type": "android"},
        {"name": "obese_female",    "height_scale": 0.94, "mass_mod": 0.42, "dist_type": "gynoid"},
        {"name": "athletic_male",   "height_scale": 1.05, "mass_mod": -0.05, "dist_type": "v_taper"},
        {"name": "athletic_female", "height_scale": 0.98, "mass_mod": -0.08, "dist_type": "hourglass"}
    ]

    print("[+] Initializing Gaussian Soft-Tissue Deformation Engine...")

    for target in phenotype_recipes:
        # Wipe Blender scene clean to prevent asset bleeding
        bpy.ops.wm.read_factory_settings(use_empty=True)

        # Import the true MetaHuman base vertex shell
        bpy.ops.import_scene.fbx(filepath=base_fbx_path)

        # Find the imported mesh object
        human_mesh = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                human_mesh = obj
                break

        if not human_mesh:
            print(f"[-] Skipping {target['name']}: No mesh layout detected.")
            continue

        human_mesh.name = target["name"]
        
        # Apply global vertical skeletal height adjustment
        human_mesh.scale[2] = target["height_scale"]
        bpy.context.view_layer.objects.active = human_mesh
        human_mesh.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        mesh_data = human_mesh.data
        mass = target["mass_mod"]

        # --- GAUSSIAN RADIAL BASIS TRANSFORMATION LOOP ---
        for vertex in mesh_data.vertices:
            x, y, z = vertex.co[0], vertex.co[1], vertex.co[2]

            # 1. Base Android Adiposity Vector (Stomach/Abdomen Center)
            # Landmark height = 1.05m, Spread standard deviation = 0.22m
            w_stomach = math.exp(-((z - 1.05) ** 2) / (2 * (0.22 ** 2)))
            
            # 2. Base Gynoid Adiposity Vector (Hips/Thighs Center)
            # Landmark height = 0.82m, Spread standard deviation = 0.18m
            w_hips = math.exp(-((z - 0.82) ** 2) / (2 * (0.18 ** 2)))

            # 3. Shoulder/Clavicle Athletic Width Vector
            # Landmark height = 1.45m, Spread standard deviation = 0.15m
            w_shoulders = math.exp(-((z - 1.45) ** 2) / (2 * (0.15 ** 2)))

            # Execute phenotype-specific localized variations
            if target["dist_type"] == "android":
                # Obese Male: Heavy expansion around the waist core, moderate limb padding
                vertex.co[0] *= (1.0 + (mass * 1.1 * w_stomach) + 0.12)
                vertex.co[1] *= (1.0 + (mass * 1.5 * w_stomach) + 0.12) # Major belly protrusion
                
            elif target["dist_type"] == "gynoid":
                # Obese Female: High expansion concentrated heavily around the hips/thighs
                vertex.co[0] *= (1.0 + (mass * 1.4 * w_hips) + 0.10)
                vertex.co[1] *= (1.0 + (mass * 1.2 * w_hips) + 0.10)
                
            elif target["dist_type"] == "v_taper":
                # Athletic Male: Broaden shoulders, slim down the waist vertices significantly
                vertex.co[0] *= (1.0 + (0.18 * w_shoulders) - (0.12 * w_stomach))
                vertex.co[1] *= (1.0 + (0.04 * w_shoulders) - (0.16 * w_stomach))
                
            elif target["dist_type"] == "hourglass":
                # Athletic Female: Balanced upper/lower expansion with tight waist compression
                vertex.co[0] *= (1.0 + (0.08 * w_shoulders) + (0.10 * w_hips) - (0.14 * w_stomach))
                vertex.co[1] *= (1.0 + (0.04 * w_shoulders) + (0.06 * w_hips) - (0.18 * w_stomach))

        # Permanently bake the morphological deformations into virtual memory coordinates
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # Export the finalized, clean phenotype asset
        output_file_path = os.path.join(output_dir, f"{target['name']}.obj")
        bpy.ops.wm.obj_export(filepath=output_file_path, export_selected_objects=True)
        print(f"[SUCCESSFUL MORPH] Exported pristine asset: {target['name']}.obj")

    print(f"\n[+] Processing Complete. Human assets safely generated inside: {output_dir}")

if __name__ == "__main__":
    generate_biologically_accurate_anchors()