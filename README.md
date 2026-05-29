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

### Analysis & Inspection
- `inspect_blender_names.py` - Inspect Blender naming conventions
- `inspect_blender_nodes.py` - Inspect Blender node structures
- `inspect_fbx_layers.py` - Inspect FBX layer organization
- `verify_topology.py` - Verify mesh topology consistency

### Model Training & Inference
- `train_3dae.py` - Train a memory-optimized 3D autoencoder
- `metahuman_extract_latents.py` - Extract latent representations from trained model
- `metahuman_direct_regressor.py` - Direct regression model for biometric prediction

### Evaluation & Visualization
- `metahuman_regressor_bench.py` - Benchmark regression model performance
- `metahuman_generate_heatmaps.py` - Generate visualization heatmaps
- `metahuman_plot_biometric_results.py` - Plot biometric analysis results
- `metahuman_generate_registry.py` - Generate data registry files
- `traverse_latent_space.py` - Explore and traverse latent space

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

## Usage

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

## Data Format

The pipeline expects:
- FBX files with standardized naming convention: `step_h{H}_f{F}_val_h{H_VAL}_f{F_VAL}`
  - H: Height index
  - F: Fat index
  - H_VAL: Height scale value
  - F_VAL: Fat scale value

## Key Features

- **Biometric Metadata Extraction**: Automatically parses physical attributes from filenames
- **Memory-Optimized Architecture**: Fits 6GB VRAM constraints
- **Disentangled Representation**: Separates mesh variation from biometric attributes
- **Topology Verification**: Ensures mesh consistency across dataset

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
