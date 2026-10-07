---
pretty_name: "MetaHuman Baseline — Parsed Mesh Arrays"
viewer: false
size_categories:
  - n<1K
tags:
  - 3d
  - computer-vision
  - synthetic-data
  - metahuman
  - body-shape
  - cpsi-2026
---

# MetaHuman Baseline — Parsed Mesh Arrays

**03 · Train and evaluate** · **200 synthetic cohort meshes** · **CPSI 2026 research companion**

Compressed NumPy mesh arrays, extracted features, and baseline evaluation artifacts for the 3D Mesh Pipeline.

[Code and notebooks](https://github.com/Ghuile/3D-Mesh-pipeline) · [Reproduction guide](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/docs/REPRODUCIBILITY.md) · [Paper](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/CPSI20_CR_Final.pdf) · [Project collection](https://huggingface.co/collections/Ghuile/3d-mesh-pipeline-cpsi-2026)

## Research context

Companion data for **Decoupled Parametric Human Shape Generation: A Fully Synthetic Framework for Biometric and Adiposity Estimation**, by **Vasileios Nikolaou, Daqing Chen, and Perry Xiao**, presented at [CPSI 2026](https://agist.org/CPSI2026/index.html), University of Cambridge, United Kingdom, **11–14 August 2026**.

![3D Mesh Pipeline overview](https://raw.githubusercontent.com/Ghuile/3D-Mesh-pipeline/main/Figure%201.png)

## Files and organization

| Path | Contents |
|---|---|
| `female_medium_average/` | 100 compressed female mesh NPZ files |
| `male_medium_average/` | 100 compressed male mesh NPZ files |
| `metahuman_master_registry.csv` | Mesh filenames, cohort labels, scale parameters, and synthetic BFP targets |
| `metahuman_extracted_features.npz` | Saved `latents` `(200, 32)`, `targets` `(200,)`, and `cohorts` `(200,)` |
| `separated_evaluation_results.npz` | Saved female/male targets and MLP/GPR predictions; 20 held-out entries per cohort |
| `*_biometric_attention_heatmap.ply` | Two gradient-sensitivity point clouds |
| `metahuman_separated_prediction_evaluation.jpg` | Saved baseline prediction plot |

### Mesh array schema

Representative files from each cohort were inspected:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `vertices` | `(N, 3)` | `float32` | XYZ coordinates in the export's coordinate system |
| `faces` | `(M, 3)` | `int32` | Triangle vertex indices |
| `actual_height_scale` | scalar | `float64` | Stored height-scale metadata |
| `actual_fat_scale` | scalar | `float64` | Stored fat-scale metadata |

The inspected female sample has **66,993 vertices** and the male sample **66,991**, each with **124,910 faces**. These are sample observations, not a full geometry audit. Verify topology before training.

```python
from pathlib import Path
import numpy as np

mesh_path = next(Path("data_parsed/female_medium_average").glob("*_compressed.npz"))
with np.load(mesh_path, allow_pickle=False) as mesh:
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    print(vertices.shape, faces.shape)
```

## Training and evaluation

Use the female and male Colab notebooks in GitHub for training. They expect cohort-specific ZIP archives and configured Drive paths; this Hub repository distributes individual NPZ files rather than those ZIP archives. Existing feature and evaluation archives are saved research outputs, not fresh evaluations performed during documentation preparation. No model checkpoints are included in this dataset repository.

The mesh directories do not define a fixed train/validation/test partition. The downstream benchmark uses an 80/20 split with `random_state=42` and assumes the first 100 feature rows are female. Verify registry and feature ordering before reusing results.

## Download

This is a repository of mesh files and research artifacts. The automatic tabular viewer is disabled because it does not represent this mixed mesh-file layout. Use `huggingface_hub` to download files; the examples do not assume a tabular `load_dataset()` interface.

```bash
python -m pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ghuile/metahuman-data-parsed",
    repo_type="dataset",
    revision="f60a3d66c80a809e1ce79d1f93963a3628dc86bd",
    local_dir="data_parsed",
    allow_patterns=["female_medium_average/*.npz", "male_medium_average/*.npz", "*.csv", "*.npz"],
)
```

The example pins the inspected data revision and selects the main inputs for this pipeline stage. Remove `allow_patterns` to download the complete repository. Full file inventory size is approximately **0.23 GB** (decimal bytes, including auxiliary files); a filtered download may be much smaller. Counts and sizes were checked on **7 October 2026**. Pin a revision when reporting experiments.

## Interpretation and limitations

- These are synthetic variations of the project's female and male MetaHuman cohorts, not a representative sample of human participants.
- BFP labels are scale-derived synthetic targets, not clinical measurements. The baseline mapping uses fat scale 0.75–1.60 and cohort-specific target ranges of 12–48 (female) and 5–40 (male); stress-test targets extrapolate that mapping.
- Scale factors are dimensionless. Check actual coordinate units before applying millimetre-valued perturbations or interpreting dimensions.
- The paper and checked-in training settings have documented differences, including cohort vertex counts and batch size. See the reproduction guide before comparing results.
- Suitable uses include studying synthetic geometry, preprocessing, representation learning, and the accompanying experiments. The data do not establish clinical accuracy or population-level validity.

## Related datasets

| Stage | Repository |
|---|---|
| Raw baseline FBX | [metahuman-data-generation](https://huggingface.co/datasets/Ghuile/metahuman-data-generation) |
| Sanitized baseline OBJ | [metahuman-data-sanitized](https://huggingface.co/datasets/Ghuile/metahuman-data-sanitized) |
| Parsed baseline arrays and outputs | [metahuman-data-parsed](https://huggingface.co/datasets/Ghuile/metahuman-data-parsed) |
| OOD assets, arrays, and features | [unreal-metahuman-stress-test](https://huggingface.co/datasets/Ghuile/unreal-metahuman-stress-test) |

## Citation

```bibtex
@conference{nikolaou2026decoupled,
  author = {Nikolaou, Vasileios and Chen, Daqing and Xiao, Perry},
  title = {Decoupled Parametric Human Shape Generation: A Fully Synthetic Framework for Biometric and Adiposity Estimation},
  booktitle = {2026 International Conference on Cyber-Physical Social Intelligence (CPSI 2026)},
  year = {2026},
  address = {Cambridge, United Kingdom},
  url = {https://researchportal.lsbu.ac.uk/en/publications/decoupled-parametric-human-shape-generation-a-fully-synthetic-fra/}
}
```

The [LSBU research record](https://researchportal.lsbu.ac.uk/en/publications/decoupled-parametric-human-shape-generation-a-fully-synthetic-fra/) confirms the conference attribution. A paper DOI has not been verified as of 7 October 2026.

## License and provenance

No dataset-specific license declaration was present in this repository at the inspected revision. This documentation update does not assign a new license or change permissions. The GitHub software license should not be assumed to license these assets. See the project's [existing third-party notices](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/THIRD_PARTY_NOTICES.md) and contact the maintainer for clarification on reuse.

## Maintenance

Maintained by [Vasileios Nikolaou / Ghuile](https://huggingface.co/Ghuile). Report documentation or pipeline issues through [GitHub issues](https://github.com/Ghuile/3D-Mesh-pipeline/issues); use this dataset's Community tab for questions about the hosted files. Include the dataset revision and relevant filename.
