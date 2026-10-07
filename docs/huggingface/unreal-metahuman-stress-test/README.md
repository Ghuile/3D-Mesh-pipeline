---
pretty_name: "MetaHuman — Out-of-Distribution Stress Test"
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

# MetaHuman — Out-of-Distribution Stress Test

**04 · Stress-test** · **200 synthetic cohort meshes** · **CPSI 2026 research companion**

Expanded-scale synthetic body meshes and features for out-of-distribution evaluation of the 3D Mesh Pipeline.

[Code and notebooks](https://github.com/Ghuile/3D-Mesh-pipeline) · [Reproduction guide](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/docs/REPRODUCIBILITY.md) · [Paper](https://github.com/Ghuile/3D-Mesh-pipeline/blob/main/CPSI20_CR_Final.pdf) · [Project collection](https://huggingface.co/collections/Ghuile/3d-mesh-pipeline-cpsi-2026)

## Research context

Companion data for **Decoupled Parametric Human Shape Generation: A Fully Synthetic Framework for Biometric and Adiposity Estimation**, by **Vasileios Nikolaou, Daqing Chen, and Perry Xiao**, presented at [CPSI 2026](https://agist.org/CPSI2026/index.html), University of Cambridge, United Kingdom, **11–14 August 2026**.

![3D Mesh Pipeline overview](https://raw.githubusercontent.com/Ghuile/3D-Mesh-pipeline/main/Figure%201.png)

## Files and organization

| Path | Format | Meshes |
|---|---|---:|
| `data_stress_test/female_extreme_stress/` | FBX | 100 |
| `data_stress_test/male_extreme_stress/` | FBX | 100 |
| `data_sanitized_stress_test/female_extreme_stress/` | OBJ | 100 |
| `data_sanitized_stress_test/male_extreme_stress/` | OBJ | 100 |
| `data_parsed_stress_test/female_extreme_stress/` | NPZ | 100 |
| `data_parsed_stress_test/male_extreme_stress/` | NPZ | 100 |
| `data_parsed_stress_test/metahuman_stress_extracted_features.npz` | NPZ | Saved feature archive |

The three representations describe **200 variants**, not 600 independent samples. Filenames encode a 10 × 10 grid per cohort with height scale **0.80–1.22** and fat scale **0.65–1.85**, extending the baseline range. This evaluates geometric extrapolation within the synthetic generation setup, not generalization to measured human scans.

## Parsed arrays and saved features

Mesh NPZ files contain `vertices`, `faces`, `actual_height_scale`, and `actual_fat_scale`. Representative female/male files contain `(66993, 3)` / `(66991, 3)` float32 vertices and `(124910, 3)` int32 faces. The feature archive contains `latents` `(200, 32)`, `targets` `(200,)`, and `cohorts` `(200,)`.

## Evaluation workflow

Download into the GitHub checkout root to retain the three existing pipeline directories. If using the supplied parsed meshes, begin with feature extraction or use the saved features after verifying provenance:

```bash
python extract_stress_test_latents.py
python evaluate_stress_test_ood.py
python generate_publication_plots.py
```

Configure baseline-data and checkpoint paths first. The latent evaluator trains regressors on baseline features and tests on this stress cohort. The publication-plot script instead fits a direct-coordinate Ridge probe. OOD extraction can truncate or pad coordinates, and plotting truncates to a common feature length; neither establishes vertex correspondence. Confirm target order against sorted mesh filenames before interpreting plots.

## Download

This is a repository of mesh files and research artifacts. The automatic tabular viewer is disabled because it does not represent this mixed mesh-file layout. Use `huggingface_hub` to download files; the examples do not assume a tabular `load_dataset()` interface.

```bash
python -m pip install huggingface_hub
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Ghuile/unreal-metahuman-stress-test",
    repo_type="dataset",
    revision="395d1cdc06e42a2c090cd5fccf39af0bd5b2c485",
    local_dir=".",
    allow_patterns=["data_parsed_stress_test/**"],
)
```

The example pins the inspected data revision and selects the parsed OOD stage only. Remove `allow_patterns` to download the complete repository. Full file inventory size is approximately **10.05 GB** (decimal bytes, including auxiliary files); a filtered download may be much smaller. Counts and sizes were checked on **7 October 2026**. Pin a revision when reporting experiments.

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
