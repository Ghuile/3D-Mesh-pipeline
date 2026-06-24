import os
import unreal

def execute_extreme_stress_test_dataset():
    # 1. Isolated absolute base path to protect your clean baseline data
    base_output_dir = "C:/Users/Vasileios Nikolaou/Documents/PhD/research_pipeline/data_stress_test/"
    
    # 2. Symmetric demographics: Active passes for BOTH cohorts to satisfy reviewers
    active_presets = [
        ("/Game/MetaHumans/MH_Ada/BP_MH_Ada", "ada", "female_extreme_stress"),
        ("/Game/MetaHumans/MH_Aoi/BP_MH_Aoi", "aoi", "male_extreme_stress"), 
    ]
    
    # 3. Expanded 10x10 Factorial Matrix (100 unique variants per identity = 200 total)
    steps_height = 10
    steps_fat = 10
    
    # 4. Calibrated extreme out-of-distribution physical range bounds
    min_h, max_h = 0.80, 1.22  # Extends range significantly for extreme height thresholds
    min_f, max_f = 0.65, 1.85  # Reaches from severe underweight up to clinical obesity limits

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    print("[+] Initializing Out-of-Distribution Parametric Stress-Test Pipeline...")

    for blueprint_path, subject_name, archetype_folder in active_presets:
        # Generate the destination folder layout inside the isolated folder
        target_dir = os.path.join(base_output_dir, archetype_folder).replace("\\", "/")
        os.makedirs(target_dir, exist_ok=True)
        
        # Ingest the master character blueprint asset
        blueprint_asset = unreal.EditorAssetLibrary.load_asset(blueprint_path)
        if not blueprint_asset:
            unreal.log_error(f"[-] Critical Error: Blueprint asset unlinked or missing at: {blueprint_path}")
            continue

        print(f"\n[🚀] Beginning Extreme Stress Pass for: {subject_name.upper()} ➔ Target Folder: {archetype_folder}")

        # --- 2D FACTORIAL MATRIX EXECUTION LOOP ---
        for h_idx in range(steps_height):
            h_fraction = h_idx / float(steps_height - 1) if steps_height > 1 else 0.5
            target_height = min_h + (h_fraction * (max_h - min_h))
            
            for f_idx in range(steps_fat):
                f_fraction = f_idx / float(steps_fat - 1) if steps_fat > 1 else 0.5
                target_fat = min_f + (f_fraction * (max_f - min_f))
                
                # A. Spawn character instance at the level center origin
                spawn_loc = unreal.Vector(0.0, 0.0, 0.0)
                spawn_rot = unreal.Rotator(0.0, 0.0, 0.0)
                spawned_actor = actor_subsystem.spawn_actor_from_object(blueprint_asset, spawn_loc, spawn_rot)
                
                if not spawned_actor:
                    unreal.log_error("[-] Viewport execution context blocked.")
                    return

                # B. Find skeletal components and strip clothing objects
                components = spawned_actor.get_components_by_class(unreal.SkeletalMeshComponent)
                body_component = None
                
                for comp in components:
                    comp_name = comp.get_name().lower()
                    if any(clothing in comp_name for clothing in ["torso", "legs", "feet", "shoes"]):
                        comp.set_skeletal_mesh(None)
                    elif comp_name == "body":
                        body_component = comp

                if body_component:
                    # Apply independent 3D bounding transformations
                    body_component.set_world_scale3d(unreal.Vector(target_fat, target_fat, target_height))
                
                # C. Select actor to isolate for the file exporter task
                actor_subsystem.set_actor_selection_state(spawned_actor, True)
                
                # D. Build unique name string (incorporates 'val_' layout to dodge regex extraction bugs)
                filename = f"step_h{h_idx}_f{f_idx}_val_h{target_height:.2f}_f{target_fat:.2f}.fbx"
                full_export_path = os.path.join(target_dir, filename).replace("\\", "/")
                
                # E. Configure automated export properties
                export_task = unreal.AssetExportTask()
                export_task.object = spawned_actor.get_world()
                export_task.filename = full_export_path
                export_task.automated = True
                export_task.selected = True
                
                fbx_options = unreal.FbxExportOption()
                fbx_options.level_of_detail = False # Keeps LOD 0 high-density surface details pristine
                fbx_options.export_morph_targets = True
                fbx_options.collision = False
                export_task.options = fbx_options
                
                # Run engine asset exporter
                unreal.Exporter.run_asset_export_task(export_task)
                
                # F. Clear active level instance from virtual memory
                actor_subsystem.destroy_actor(spawned_actor)
                
    print(f"\n[+] STRESS-TEST GENERATION COMPLETED! Files saved directly to: {base_output_dir}")

# Trigger script processing loop inside editor viewport context
execute_extreme_stress_test_dataset()