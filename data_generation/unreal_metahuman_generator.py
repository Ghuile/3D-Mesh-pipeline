import os
import unreal

def execute_high_density_factorial_dataset():
    # 1. Base absolute output path
    base_output_dir = "C:/Users/Vasileios Nikolaou/Documents/PhD/research_pipeline/data_generation/"
    
    # 2. COMPLETE TARGET ARRAY: Paths mapped exactly from your local in-engine assets
    # Format: (Blueprint Asset Path, Subject Identifier String, Destination Archetype Folder)
    active_presets = [
        #("/Game/MetaHumans/MH_Ada/BP_MH_Ada", "ada", "female_medium_average"),
        ("/Game/MetaHumans/MH_Aoi/BP_MH_Aoi", "aoi", "male_medium_average"), 
    ]
    
    # 3. Dense 10x10 Sampling Configurations (100 unique morphs per identity pass)
    steps_height = 10
    steps_fat = 10
    
    # Invariant physical range bounds
    min_h, max_h = 0.85, 1.15  
    min_f, max_f = 0.75, 1.60  

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    print("[+] Initializing Unified High-Density Parametric Exporter Pipeline...")

    for blueprint_path, subject_name, archetype_folder in active_presets:
        # Generate the destination folder layout directly inside data_generation
        target_dir = os.path.join(base_output_dir, archetype_folder).replace("\\", "/")
        os.makedirs(target_dir, exist_ok=True)
        
        # Load the compiled master character actor template
        blueprint_asset = unreal.EditorAssetLibrary.load_asset(blueprint_path)
        if not blueprint_asset:
            unreal.log_error(f"[-] Critical Error: Blueprint asset unlinked or missing at: {blueprint_path}")
            continue

        print(f"\n[🚀] Beginning Batch Generation Pass for: {subject_name.upper()} ➔ Target Folder: {archetype_folder}")

        # --- 2D FACTORIAL MATRIX EXECUTION LOOP ---
        for h_idx in range(steps_height):
            h_fraction = h_idx / float(steps_height - 1) if steps_height > 1 else 0.5
            target_height = min_h + (h_fraction * (max_h - min_h))
            
            for f_idx in range(steps_fat):
                f_fraction = f_idx / float(steps_fat - 1) if steps_fat > 1 else 0.5
                target_fat = min_f + (f_fraction * (max_f - min_f))
                
                # A. Spawn a clean actor instance at viewport origin
                spawn_loc = unreal.Vector(0.0, 0.0, 0.0)
                spawn_rot = unreal.Rotator(0.0, 0.0, 0.0)
                spawned_actor = actor_subsystem.spawn_actor_from_object(blueprint_asset, spawn_loc, spawn_rot)
                
                if not spawned_actor:
                    unreal.log_error("[-] Viewport state blocked execution.")
                    return

                # B. Parse skeletal layers to automatically strip all garments
                components = spawned_actor.get_components_by_class(unreal.SkeletalMeshComponent)
                body_component = None
                
                for comp in components:
                    comp_name = comp.get_name().lower()
                    if any(clothing in comp_name for clothing in ["torso", "legs", "feet", "shoes"]):
                        comp.set_skeletal_mesh(None)
                    elif comp_name == "body":
                        body_component = comp

                if body_component:
                    # Deform morphological dimensions independently
                    body_component.set_world_scale3d(unreal.Vector(target_fat, target_fat, target_height))
                
                # C. Target selection for export isolation
                actor_subsystem.set_actor_selection_state(spawned_actor, True)
                
                # D. Build tracking filename schemas string (Clean index tagging for data tracking)
                filename = f"step_h{h_idx}_f{f_idx}_val_h{target_height:.2f}_f{target_fat:.2f}.fbx"
                full_export_path = os.path.join(target_dir, filename).replace("\\", "/")
                
                # E. Task Configuration
                export_task = unreal.AssetExportTask()
                export_task.object = spawned_actor.get_world()
                export_task.filename = full_export_path
                export_task.automated = True
                export_task.selected = True
                
                fbx_options = unreal.FbxExportOption()
                fbx_options.level_of_detail = False # Forces isolation of pristine LOD 0 surface shell
                fbx_options.export_morph_targets = True
                fbx_options.collision = False
                export_task.options = fbx_options
                
                # Execute the export command
                unreal.Exporter.run_asset_export_task(export_task)
                
                # F. Destroy active level instance to clean memory cache
                actor_subsystem.destroy_actor(spawned_actor)
                
    print(f"\n[+] EXECUTION BATCH COMPLETED! Targeted files written to: {base_output_dir}")

# Initialize script loop execution block
execute_high_density_factorial_dataset()