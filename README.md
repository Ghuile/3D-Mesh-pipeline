# 3D Mesh Pipeline

Research code accompanying **Decoupled Parametric Human Shape Generation: A Fully Synthetic Framework for Biometric and Adiposity Estimation**, by Vasileios Nikolaou, Daqing Chen, and Perry Xiao.

The pipeline generates synthetic MetaHuman body meshes, prepares coordinate arrays, trains cohort-specific morphological autoencoders, and evaluates regression models for synthetic body-fat-percentage targets. It includes coordinate-noise experiments and out-of-distribution (OOD) evaluation.

[Paper](CPSI20_CR_Final.pdf) · [Installation](INSTALLATION.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

![Research pipeline overview from the accompanying paper](Figure%201.png)

## Research scope

The baseline generation design uses a 10 × 10 parameter grid for each of two MetaHuman cohorts. Body-fat-percentage labels are derived from geometric scale parameters; they are synthetic targets, not measured body composition.

The notebooks use a 32-dimensional latent representation. Input sizes are 66,993 vertices × 3 coordinates for the female cohort and 66,991 × 3 for the male cohort: 200,979 and 200,973 input values, respectively.

This repository preserves research scripts with local paths and external asset requirements. Read the [configuration and known differences](docs/REPRODUCIBILITY.md#configuration-and-known-differences) before running experiments. An end-to-end reproduction has not been verified during the documentation cleanup.

## Workflow

| Stage | Entry points | Environment |
|---|---|---|
| Generate baseline assets | `data_generation/unreal_metahuman_generator.py` | Unreal Editor Python |
| Sanitize FBX to OBJ | `sanitize_pipeline_female.py`, `sanitize_pipeline_male.py` | Python launching Blender |
| Parse OBJ coordinates | `preprocess_metahuman_female.py`, `preprocess_metahuman_male.py` | Local Python |
| Validate and build metadata | `verify_topology.py`, `metahuman_generate_registry.py` | Local Python |
| Train and explore latent space | `fmetahuman_cloud_training.ipynb`, `mmetahuman_cloud_training.ipynb` | Google Colab |
| Extract and benchmark | `metahuman_extract_latents.py`, `metahuman_regressor_bench.py` | Python and saved checkpoints |
| Plot baseline results | `metahuman_plot_biometric_results.py`, `metahuman_direct_regressor.py` | Local Python |
| Export sensitivity maps | `metahuman_generate_heatmaps.py` | Python, checkpoints, mean templates |
| Evaluate coordinate noise | `test_noise_robustness_combined.py` and cohort variants | Python and checkpoints |
| Evaluate OOD data | Stress-test generation, sanitization, parsing, and evaluation scripts | Unreal Editor, Blender, Python |

The notebooks create `train_metahuman_autoencoder.py` within Colab. There is no standalone `train_3dae.py` in this checkout. The `test_noise_robustness_*.py` files are experiment scripts, not a unit-test suite.

## Getting started

```bash
git clone https://github.com/Ghuile/3D-Mesh-pipeline.git
cd 3D-Mesh-pipeline
python -m venv .venv
```

Follow [INSTALLATION.md](INSTALLATION.md) to activate the environment and install dependencies. Use the ordered steps in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) after configuring dataset, executable, checkpoint, and Unreal asset paths.

## Data and artifacts

The existing project documentation points to these external dataset repositories:

- [MetaHuman generation data](https://huggingface.co/datasets/Ghuile/metahuman-data-generation)
- [Parsed MetaHuman data](https://huggingface.co/datasets/Ghuile/metahuman-data-parsed)
- [Dataset collection](https://huggingface.co/Ghuile/datasets)

Check file listings and revisions before use. A separate sanitized-data URL, immutable dataset revisions, checksums, and a checkpoint release are not established by this checkout.

| Location | Contents |
|---|---|
| `data_generation/` | Tracked generation scripts and population metadata; local raw assets |
| `data_sanitized/` | Local baseline OBJ meshes |
| `data_parsed/` | Local baseline NPZ arrays, registry, features, and results |
| `data_stress_test/` | Local OOD FBX assets |
| `data_sanitized_stress_test/` | Local OOD OBJ meshes |
| `data_parsed_stress_test/` | Local OOD NPZ arrays and features |

Mesh arrays use `vertices` with shape `(N, 3)` and `faces` with shape `(M, 3)`. See the reproducibility guide for metadata, ordering, and units. Large assets and generated checkpoints are excluded from Git.

## Paper and citation

Use the [camera-ready paper](CPSI20_CR_Final.pdf) as the primary included reference. The [earlier manuscript](CPSI%20Manuscript.pdf) remains for provenance; its figures and results differ from the camera-ready version.

Citation metadata is provided in [CITATION.cff](CITATION.cff). Venue details and a DOI are omitted pending verification.

## License

Repository software is provided under the [MIT License](LICENSE). Existing third-party notices and acknowledgements are retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
