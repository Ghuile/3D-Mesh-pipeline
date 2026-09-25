# Installation

The project spans ordinary Python, Blender Python, and Unreal Editor Python. Training notebooks additionally use Google Colab and Google Drive.

## Local Python

```bash
git clone https://github.com/Ghuile/3D-Mesh-pipeline.git
cd 3D-Mesh-pipeline
python -m venv .venv
```

Activate the environment in your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS or Linux
source .venv/bin/activate
```

Install dependencies and check the analysis imports:

```bash
python -m pip install -r requirements.txt
python -c "import numpy, scipy, pandas, matplotlib, torch, sklearn; print('Analysis imports available')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The requirements contain historical lower bounds, not a locked reproduction environment. This checkout does not establish a tested minimum Python version or complete package-version record. Record Python, package, GPU, and driver versions for each reproduction. An import check does not validate experiments or checkpoint compatibility.

## Blender

Sanitization wrappers launch an external Blender process. Their paths currently refer to Blender 5.1 on Windows. Configure `BLENDER_PATH` and input/output directories before running them.

Scripts importing `bpy` under `data_generation/` must run inside Blender. Installing analysis requirements into ordinary Python does not provide that runtime. For example, after supplying the required base asset and CSV:

```bash
blender --background --python data_generation/assemble_synthetic_cohort.py -- data_generation/population_matrix.csv
```

The wrappers contain embedded Blender source. Preserve that source in documentation-only changes. Compatibility with other Blender versions has not been established here.

## Unreal Engine and MetaHuman

Run the Unreal generation scripts inside an Unreal Editor project with Python scripting and the referenced MetaHuman assets available. Inspect asset paths, active targets, output directories, and scale ranges first. The baseline generator currently has the female target commented out.

The `unreal` module belongs to the editor runtime. Ordinary Python cannot execute these scripts by installing the requirements alone. Exact engine and MetaHuman versions used for the paper are not recorded in the dependency file.

## Colab training

Open the cohort-specific notebook and read its Markdown instructions before executing cells in order. Supply its parsed-data archive in Google Drive and configure the existing Drive paths. Both notebooks write shared script and artifact filenames, so use separate runtimes or preserve cohort-specific copies between runs.

Training creates a cohort checkpoint and `mean_template.npy` in the runtime working directory. Transfer these to the configured local checkpoint directory. Heatmaps expect the templates as `female_mean_template.npy` and `male_mean_template.npy`.

## Data and configuration

The [reproducibility guide](docs/REPRODUCIBILITY.md) lists the execution sequence and required artifacts. Many scripts retain absolute research-workstation paths. There is no shared configuration file or universal command-line interface.

Open3D is used by alternate direct-FBX preprocessing scripts. Trimesh, Pillow, SciPy, and torchvision remain in the historical requirements; this cleanup does not remove or update package declarations. GPU memory requirements have not been rebenchmarked.
