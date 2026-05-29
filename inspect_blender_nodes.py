import os
import subprocess

def inspect_blender_nodes():
    # 1. Point directly to your launcher
    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
    
    # 2. Get your raw FBX file
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\female_medium_average"
    first_fbx = os.path.join(SOURCE_DIR, [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.fbx')][0])

    print(f"Executing deep scene inspection on: {first_fbx}\n")

    # 3. Use Blender's native CLI to import and dump the scene structure
    # We load the file and then run a simple Python one-liner to print the objects
    command = [
        BLENDER_PATH, 
        "--background", 
        first_fbx, 
        "--python-expr", 
        "import bpy; print('--- IDENTIFIED OBJECTS ---'); [print(f'{obj.name} | Type: {obj.type} | Verts: {len(obj.data.vertices) if obj.type == 'MESH' else 'N/A'}') for obj in bpy.context.scene.objects if obj.type == 'MESH']"
    ]
    
    # 4. Execute and capture EVERYTHING
    process = subprocess.run(command, capture_output=True, text=True)
    
    # 5. Print the raw output from Blender
    print(process.stdout)
    print(process.stderr)

if __name__ == "__main__":
    inspect_blender_nodes()