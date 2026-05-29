
import bpy
import os

source_dir = os.path.normpath(r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\male_medium_average")
target_dir = os.path.normpath(r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_sanitized\male_medium_average")

fbx_files = [f for f in os.listdir(source_dir) if f.lower().endswith(".fbx")]
print(f"--- BLENDER WORKER: FOUND {len(fbx_files)} FBX FILES TO PROCESS ---")

for file in fbx_files:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    fbx_path = os.path.join(source_dir, file)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if mesh_objects:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        
        # --- MIRROR FEMALE: Join all meshes into a single continuous block ---
        bpy.ops.object.join()
        
        base_name = os.path.splitext(file)[0]
        output_obj_path = os.path.join(target_dir, base_name + ".obj")
        
        bpy.ops.wm.obj_export(
            filepath=output_obj_path,
            export_selected_objects=True,
            export_materials=False
        )
        print(f"BAKE_CONFIRMED: {base_name}.obj")
