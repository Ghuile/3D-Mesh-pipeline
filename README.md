# Research Pipeline

A comprehensive research pipeline for 3D human body mesh processing and analysis using machine learning, with focus on biometric attribute prediction and latent space exploration.

## Overview

This project provides a complete pipeline for:
- **3D Mesh Processing**: Parse and process FBX files containing human body models
- **Biometric Extraction**: Extract physical attributes (height, body fat percentage) from mesh data
- **Deep Learning**: Train a disentangled autoencoder for mesh reconstruction and attribute prediction
- **Analysis & Visualization**: Generate heatmaps, registry files, and latent space visualizations

## Project Structure

### Data Processing
- `preprocess_pipeline.py` - Main preprocessing pipeline for mesh data
- `preprocess_metahuman_female.py` - Female-specific preprocessing
- `preprocess_metahuman_male.py` - Male-specific preprocessing
- `sanitize_pipeline_female.py` - Data sanitization for female data
- `sanitize_pipeline_male.py` - Data sanitization for male data

### Data Generation
- `data_generation/unreal_metahuman_generator.py` - Generate synthetic MetaHuman models from Unreal Engine
- `data_generation/generate_phenotype_anchors.py` - Generate phenotypic anchor points for biometric attributes
- `data_generation/assemble_synthetic_cohort.py` - Assemble synthetic cohorts from generated models
- `data_generation/validate_topology_invariants.py` - Validate mesh topology consistency across synthetic models

### Model Training & Inference
- `train_3dae.py` - Train a memory-optimized 3D autoencoder
- `metahuman_extract_latents.py` - Extract latent representations from trained model
- `metahuman_direct_regressor.py` - Direct regression model for biometric prediction

### Evaluation & Visualization
- `metahuman_regressor_bench.py` - Benchmark regression model performance
- `metahuman_generate_heatmaps.py` - Generate visualization heatmaps
- `metahuman_plot_biometric_results.py` - Plot biometric analysis results
- `metahuman_generate_registry.py` - Generate data registry files
- `verify_topology.py` - Verify mesh topology consistency

### Jupyter Notebooks
- `fmetahuman_cloud_training.ipynb` - Female model cloud training workflow
- `mmetahuman_cloud_training.ipynb` - Male model cloud training workflow

### Quality Assurance
- `metahuman_check_female_base.py` - Verify female base model integrity

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Open3D
- Additional dependencies as specified in requirements.txt

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

This project uses large 3D mesh datasets hosted on Hugging Face for reproducibility and easy access. (will upload soon)

### Dataset Availability

All datasets are available on the Hugging Face Hub:

- **Data Generation**: [your-username/metahuman-data-generation](https://huggingface.co/datasets/your-username/metahuman-data-generation)
- **Data Parsed**: [your-username/metahuman-data-parsed](https://huggingface.co/datasets/your-username/metahuman-data-parsed)
- **Data Sanitized**: [your-username/metahuman-data-sanitized](https://huggingface.co/datasets/your-username/metahuman-data-sanitized)

### Dataset Sizes

| Dataset | Size | Contents |
|---------|------|----------|
| Data Generation | ~7 GB | Raw synthetic MetaHuman models, base FBX files, phenotype anchors |
| Data Parsed | ~223 MB | Preprocessed mesh data in NumPy/tensor format |
| Data Sanitized | ~2.2 GB | Cleaned and validated mesh data ready for training |

### Downloading Datasets

#### Option 1: Using Hugging Face Datasets Library (Recommended)

```bash
# Install Hugging Face datasets
pip install datasets

# Download data in Python
from datasets import load_dataset

# Download specific dataset
dataset = load_dataset("your-username/metahuman-data-parsed")

# Access files
data = dataset['train']
```

#### Option 2: Manual Download

Visit the dataset pages on Hugging Face and download files directly:
```
https://huggingface.co/datasets/your-username/metahuman-data-parsed/tree/main
```

#### Option 3: Git LFS (for development)

```bash
# Install Git LFS
git lfs install

# Clone dataset repository
git clone https://huggingface.co/datasets/your-username/metahuman-data-parsed
cd metahuman-data-parsed
```

### Setting Up Datasets Locally

After downloading, organize datasets in your project directory:

```bash
# Create data directories
mkdir -p data_generation data_parsed data_sanitized

# Extract downloaded files
unzip data_generation.zip -d data_generation/
unzip data_parsed.zip -d data_parsed/
unzip data_sanitized.zip -d data_sanitized/
```

### Dataset Documentation

For detailed information about dataset contents, structure, and format specifications, see [DATASETS.md](DATASETS.md) (coming soon)

### Basic Workflow

1. **Preprocess data**:
   ```bash
   python preprocess_pipeline.py
   ```

2. **Train model**:
   ```bash
   python train_3dae.py
   ```

3. **Extract latents**:
   ```bash
   python metahuman_extract_latents.py
   ```

4. **Generate visualizations**:
   ```bash
   python metahuman_generate_heatmaps.py
   ```

### Synthetic Data Generation

To generate synthetic MetaHuman models for expanded datasets:

1. **Generate phenotypic anchors**:
   ```bash
   python data_generation/generate_phenotype_anchors.py
   ```

2. **Generate MetaHuman models**:
   ```bash
   python data_generation/unreal_metahuman_generator.py
   ```

3. **Validate topology consistency**:
   ```bash
   python data_generation/validate_topology_invariants.py
   ```

4. **Assemble synthetic cohorts**:
   ```bash
   python data_generation/assemble_synthetic_cohort.py
   ```

### Cloud Training

For large-scale training on cloud infrastructure, use the provided Jupyter notebooks:
- `fmetahuman_cloud_training.ipynb` - Female model training
- `mmetahuman_cloud_training.ipynb` - Male model training

### Benchmarking & Evaluation

To evaluate model performance:

```bash
python metahuman_regressor_bench.py
python metahuman_plot_biometric_results.py
```

## Data Format

### Naming Convention

The pipeline expects FBX files with standardized naming convention:
```
step_h{H}_f{F}_val_h{H_VAL}_f{F_VAL}
```

Where:
- **H**: Height index
- **F**: Fat index  
- **H_VAL**: Height scale value
- **F_VAL**: Fat scale value

### Directory Structure

```
data_generation/          # Raw synthetic models from Unreal/Blender
├── metahuman_base.fbx
├── population_matrix.csv
└── [synthetic models generated by unreal_metahuman_generator.py]

data_parsed/             # Preprocessed mesh data
├── [numpy arrays and tensors]
└── [parsed biometric attributes]

data_sanitized/          # Cleaned and validated data ready for training
├── [processed meshes]
└── [quality-verified datasets]
```

### File Formats

- **FBX**: Binary 3D mesh format (source data)
- **NPZ**: NumPy compressed arrays (parsed data)
- **PKL**: Pickle serialized Python objects (processed data)
- **CSV**: Metadata and population matrices

## Key Features

- **Synthetic Data Generation**: Create diverse MetaHuman models with controlled phenotypic variations
- **Biometric Metadata Extraction**: Automatically parses physical attributes from filenames
- **Topology Validation**: Ensures mesh consistency and topology invariance across synthetic and real datasets
- **Memory-Optimized Architecture**: Fits 6GB VRAM constraints with 200K+ vertex meshes
- **Disentangled Representation**: Separates mesh variation from biometric attributes in latent space
- **Cloud Training Support**: Jupyter notebooks for distributed training on cloud platforms
- **Comprehensive Benchmarking**: Regression model evaluation with detailed performance metrics
- **Heatmap Visualization**: Generate informative heatmap visualizations for mesh attributes

## Performance

- Input dimension: 200,979 vertices
- Latent dimension: 32
- Optimized for GPU acceleration

## Contributing

This is a research project. For contributions, please maintain code style and ensure all changes are backward compatible.

## License

See LICENSE file for details.

## Citation

If you use this pipeline in your research, please cite appropriately.

## Contact

For questions or issues, please refer to the documentation in individual scripts.
