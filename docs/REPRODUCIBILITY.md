# Reproducing the research workflow

This guide describes the checked-in scripts and notebooks, not a newly completed reproduction. The primary included reference is [CPSI20_CR_Final.pdf](../CPSI20_CR_Final.pdf).

## Configuration and known differences

| Item | Checked-in behavior or unresolved requirement |
|---|---|
| Paths | Scripts retain absolute workstation, checkpoint, and Colab Drive paths. Configure these before execution. |
| Cohort generation | The baseline Unreal generator has the female target commented out. Both cohorts require their assets and generation passes. |
| Vertex counts | Female training/extraction uses 66,993 vertices; male uses 66,991. The paper describes a shared 66,993-vertex resolution. |
| Batch size | Both notebooks use 4; the camera-ready paper describes 1. |
| Training settings | Both notebooks specify 80 epochs, initial learning rate `3e-4`, and latent dimension 32. These are source settings, not newly validated results. |
| Targets | Fat multipliers from 0.75 to 1.60 map linearly to 12–48 for female and 5–40 for male BFP targets. These are synthetic proxies. |
| Units | Noise standard deviations `0`, `0.002`, `0.005`, `0.010`, `0.020` are labeled 0, 2, 5, 10, 20 mm, which assumes coordinates in metres. Verify exports; units are not consistently established across the repository. |
| Baseline ordering | The regressor benchmark assigns the first 100 feature rows to female and the remainder to male. Verify counts and ordering. |
| OOD ordering | Publication plotting sorts mesh filenames but takes targets from feature arrays. Verify target/mesh alignment. |
| Checkpoints | Extraction can continue without a checkpoint, leaving an initialized encoder. Loading uses `strict=False`; verify actual weight compatibility before interpreting features. |
| Topology | OOD extraction truncates or pads coordinates; publication plotting truncates to a common feature length. Neither establishes vertex correspondence. |
| Randomness | Several splits/models use `random_state=42`, but not every training or noise operation is seeded. |
| Environment | A verified package lock, exact engine versions, dataset checksums, and checkpoint release provenance are not supplied by this checkout. |

These differences are documented without modifying the paper or experimental settings.

## 1. Prepare baseline meshes

Follow [installation](../INSTALLATION.md), supply external assets, and configure paths. Run `data_generation/unreal_metahuman_generator.py` inside Unreal Editor for the intended cohorts. Then run from the repository root:

```bash
python sanitize_pipeline_female.py
python sanitize_pipeline_male.py
python preprocess_metahuman_female.py
python preprocess_metahuman_male.py
python verify_topology.py
python metahuman_generate_registry.py
```

Sanitizers export OBJ files; cohort preprocessors write NPZ files under `data_parsed/female_medium_average/` and `data_parsed/male_medium_average/`. The registry is `data_parsed/metahuman_master_registry.csv`.

NPZ meshes contain `vertices` `(N, 3)` and `faces` `(M, 3)`, with scale metadata where exported. Filenames carry height/fat parameters, including the `_val_h..._f...` pattern used by the registry. Do not rename data without preserving metadata matching.

`preprocess_pipeline.py` is an alternate direct-FBX route with an 8,000–15,000 vertex-selection heuristic, not a substitute for notebook high-resolution inputs. The Blender phenotype-anchor and synthetic-cohort utilities are additional generation routes, not required steps above.

## 2. Train each cohort

Use the [female notebook](../fmetahuman_cloud_training.ipynb) and [male notebook](../mmetahuman_cloud_training.ipynb) in separate Colab runtimes. Supply `fmetahuman_data_parsed.zip` or `mmetahuman_data_parsed.zip` with the appropriate arrays.

Each notebook mounts Drive, extracts data, writes and executes a training script, and writes/runs a latent-traversal script. The archive-search cell is diagnostic; its result does not automatically update extraction paths.

| Cohort | Checkpoint | Runtime template | Local heatmap template name |
|---|---|---|---|
| Female | `fmetahuman_autoencoder_trained.pth` | `mean_template.npy` | `female_mean_template.npy` |
| Male | `mmetahuman_autoencoder_trained.pth` | `mean_template.npy` | `male_mean_template.npy` |

Preserve artifacts before the runtime ends and copy them into the configured local `weights_dir`. Do not mix cohorts. Both notebooks use shared training/traversal script filenames.

## 3. Evaluate baseline representations

```bash
python metahuman_extract_latents.py
python metahuman_regressor_bench.py
python metahuman_plot_biometric_results.py
python metahuman_direct_regressor.py
python metahuman_generate_heatmaps.py
```

Extraction produces `metahuman_extracted_features.npz` with `latents`, `targets`, and `cohorts`. The benchmark saves `separated_evaluation_results.npz`. Downstream regression uses an 80/20 split with `random_state=42`; MLP features are standardized on the training split and GPR receives unscaled latents. This split alone does not establish that autoencoder training excluded evaluation meshes.

Heatmaps are colored PLY point clouds from coordinate gradients of latent dimension 1. They are sensitivity visualizations, not validated anatomical segmentation.

## 4. Run robustness experiments

For both baseline cohorts:

```bash
python test_noise_robustness_combined.py
```

Female and male variants provide separate plots. Check coordinate units before interpreting millimetre labels.

For OOD data, execute `data_generation/unreal_metahuman_stress_tester.py` inside Unreal Editor, then:

```bash
python sanitize_stress_test.py
python preprocess_stress_test_high_res.py
python extract_stress_test_latents.py
python evaluate_stress_test_ood.py
python generate_publication_plots.py
```

The high-resolution route consumes sanitized OBJ meshes. `preprocess_stress_test.py` is an alternate direct-FBX parser. Extraction creates `metahuman_stress_extracted_features.npz`. The evaluator trains latent regressors on baseline data and evaluates the stress cohort; the publication plot script instead fits a direct-coordinate Ridge probe.

## Paper figures and artifacts

Mappings follow camera-ready captions and script purposes. Exact visual/numerical equivalence has not been rerun or established from an archived run manifest.

| Figure | Included asset | Related source |
|---|---|---|
| 1: Pipeline and attention overview | [Figure 1.png](../Figure%201.png) | Generation, preprocessing, heatmaps; no complete composition script identified |
| 2: Processing flow chart | [Figure 2.png](../Figure%202.png) | No diagram-generation source identified |
| 3: BFP estimation | [Figure 3.jpg](../Figure%203.jpg) | `metahuman_regressor_bench.py`, `metahuman_plot_biometric_results.py` |
| 4: Coordinate-noise degradation | [Figure 4.png](../Figure%204.png) | `test_noise_robustness_combined.py` |
| 5: Direct-coordinate OOD extrapolation | [Figure 5.png](../Figure%205.png) | `generate_publication_plots.py` |

## Record a reproduction

Save the commit ID, dataset revisions/checksums, ordered cohort file lists, coordinate units, environment versions, hardware, seeds, checkpoint hashes, commands, and metrics. Keep run artifacts outside Git unless publishing a documented reference. A successful process exit does not establish reproduction of the paper.
