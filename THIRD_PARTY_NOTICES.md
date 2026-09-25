# Third-party notices and acknowledgements

The following existing repository notices were relocated from `LICENSE` without changing their wording.


## Third-Party License Notices

### Unreal Engine

This project utilizes the Unreal Engine for synthetic data generation and MetaHuman model manipulation.

Unreal Engine is licensed under the Unreal Engine End User License Agreement (EULA).
For more information, please visit: https://www.unrealengine.com/en-US/eula

Key points regarding Unreal Engine usage:
- The Unreal Engine scripts in this project (`data_generation/unreal_metahuman_generator.py`) are designed to integrate with Unreal Engine through its Python API
- Users must have a valid Unreal Engine license to use these components
- MetaHuman models generated through Unreal Engine are subject to Epic Games' MetaHuman Terms of Service
- Any outputs generated using Unreal Engine components are subject to the terms and restrictions outlined in the Unreal Engine EULA

### Blender

This project uses Blender for mesh processing and manipulation through the `bpy` Python API.

Blender is licensed under the GNU General Public License (GPL).
For more information, please visit: https://www.blender.org/

The Blender integration scripts (`sanitize_pipeline_*.py`, `data_generation/assemble_synthetic_cohort.py`, etc.) are provided under the MIT License, but their use requires a compatible Blender installation.

---

## Attribution

This research pipeline builds upon:
- PyTorch deep learning framework
- Open3D 3D data processing library
- scikit-learn machine learning library
- Blender 3D modeling software
- Unreal Engine MetaHuman system

Please ensure compliance with all respective licenses when using this project.
