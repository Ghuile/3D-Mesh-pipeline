---
pretty_name: "MetaHuman Baseline — Raw FBX Meshes"
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

# MetaHuman Baseline — Raw FBX Meshes

**01 · Generate** · **200 synthetic cohort meshes** · **CPSI 2026 research companion**

Raw synthetic body meshes for the baseline generation stage of the 3D Mesh Pipeline.

[Code and notebooks](https://github.com/Ghuile/3D-Mesh-pipeline) · [Reproduction guide](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/docs/REPRODUCIBILITY.md) · [Paper](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/CPSI20_CR_Final.pdf) · [Project collection](https://huggingface.co/collections/Ghuile/3d-mesh-pipeline-cpsi-2026)

## Research context

Companion data for **Decoupled Parametric Human Shape Generation: A Fully Synthetic Framework for Biometric and Adiposity Estimation**, by **Vasileios Nikolaou, Daqing Chen, and Perry Xiao**, presented at [CPSI 2026](https://agist.org/CPSI2026/index.html), University of Cambridge, United Kingdom, **11–14 August 2026**.

![3D Mesh Pipeline overview](https://raw.githubusercontent.com/Ghuile/3D-Mesh-pipeline/main/Figure%201.png)

## Files and organization

| Path | Contents |
|---|---|
| `female_medium_average/` | 100 baseline female FBX meshes |
| `male_medium_average/` | 100 baseline male FBX meshes |
| `metahuman_base.fbx` | Additional base asset |
| `population_matrix.csv` | Population metadata for the auxiliary cohort-generation utility |
| `*.py` | Four generation and validation utilities |

The 200 cohort meshes follow a 10 × 10 grid: height scale **0.85–1.15** and fat scale **0.75–1.60**. The extra base FBX is not another grid sample. The CSV supports an auxiliary generation route; it should not be assumed to be the manifest for the 200 baseline meshes.

## Using this stage

Use the FBX assets with the cohort-specific sanitization wrappers in the GitHub repository, which launch Blender and export OBJ meshes. If you only need training inputs, start with the parsed dataset instead of downloading the raw assets.

Generation utilities stored here are historical snapshots. Use GitHub for the maintained code, environment instructions, and current documentation. Unreal generation scripts run inside Unreal Editor; Blender scripts require Blender's Python runtime.

## Download

This is a repository of mesh files and research artifacts. The automatic tabular viewer is disabled because it does not represent this mixed mesh-file layout. Use `huggingface_hub` to download files; the examples do not assume a tabular `load_dataset()` interface.

```bash
python -m pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ghuile/metahuman-data-generation",
    repo_type="dataset",
    revision="cd03bec1bc5b6d5d3347d5cda96850d342ead355",
    local_dir="data_generation",
    allow_patterns=["female_medium_average/*.fbx", "male_medium_average/*.fbx"],
)
```

The example pins the inspected data revision and selects the main inputs for this pipeline stage. Remove `allow_patterns` to download the complete repository. Full file inventory size is approximately **7.46 GB** (decimal bytes, including auxiliary files); a filtered download may be much smaller. Counts and sizes were checked on **7 October 2026**. Pin a revision when reporting experiments.

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
