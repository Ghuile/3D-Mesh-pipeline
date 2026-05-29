import os
import subprocess

def inspect_blender_nodes():
    BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 5.1\blender-launcher.exe"
    SOURCE_DIR = r"C:\Users\Vasileios Nikolaou\Documents\PhD\research_pipeline\data_generation\female_medium_average"
    first_fbx = os.path.join(SOURCE_DIR, [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith('.fbx')][0])

    blender_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=r"{first_fbx}")
print("\\n" + "="*50)
print("INTERNAL BLENDER MESH NAMES:")
print("="*50)
for obj in bpy.context.scene.objects:
    if obj.type == 'MESH':
        print(f"- {{obj.name}} | Vertices: {{len(obj.data.vertices)}}")
print("="*50 + "\\n")
"""
    with open("temp_inspector.py", "w") as f: f.write(blender_script)
    process = subprocess.run([BLENDER_PATH, "--background", "--python", "temp_inspector.py"], capture_output=True, text=True)
    os.remove("temp_inspector.py")
    print(process.stdout)

if __name__ == "__main__":
    inspect_blender_nodes()