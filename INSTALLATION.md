# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)
- GPU with CUDA support (recommended for faster training)

### Optional Prerequisites

For advanced features:
- **Blender 3.x or higher** - For mesh processing and sanitization
- **Unreal Engine 5.x or higher** - For synthetic data generation via MetaHuman
- **CUDA Toolkit** - For GPU acceleration with PyTorch

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd research_pipeline
```

### 2. Create Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Optional: Install Blender Integration

For mesh processing features:

```bash
# Install Blender Python API
blender --python -m pip install bpy
```

### 5. Optional: Install Unreal Engine Integration

For synthetic data generation:
- Install Unreal Engine 5.x
- Enable Python API in Unreal Editor Preferences
- Ensure Python 3.x is available in your UE5 installation

### 6. Verify Installation

```bash
python -c "import torch; import open3d; import pandas; print('Installation successful!')"
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with PyTorch, install the appropriate version for your system:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For macOS with MPS
pip install torch torchvision
```

### Open3D Issues

On some systems, you may need to install additional system dependencies:

```bash
# Ubuntu/Debian
sudo apt-get install libopengl0 libglvnd0

# macOS
brew install open3d
```

### Blender Integration Issues

If Blender integration doesn't work:

```bash
# Try using Blender's built-in Python
/path/to/blender/python/bin/python -m pip install --upgrade pip
/path/to/blender/python/bin/python -m pip install -r requirements.txt
```

### Unreal Engine Integration Issues

For Unreal Engine compatibility:
- Verify UE5 Python plugin is enabled
- Check that Python path matches UE5 configuration
- Ensure MetaHuman plugin is installed

## GPU Setup (Recommended)

For GPU acceleration with PyTorch:

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

### Memory Requirements

- **Minimum**: 6GB VRAM for 3D autoencoder training
- **Recommended**: 8GB+ VRAM for optimal performance
- **CPU Fallback**: Works with CPU but training will be significantly slower

## Data Setup

Create directories for data storage:

```bash
mkdir -p data_generation data_parsed data_sanitized
```

Note: These directories should remain empty when pushing to repository. Data files are managed separately via `.gitignore`.

## Project Structure Verification

After installation, verify the directory structure:

```
research_pipeline/
├── data_generation/          # Synthetic data generation scripts
├── data_parsed/              # Processed mesh data (not committed)
├── data_sanitized/           # Sanitized data output (not committed)
├── requirements.txt
├── README.md
├── INSTALLATION.md
├── CONTRIBUTING.md
├── LICENSE
└── *.py                       # Core pipeline scripts
```

## Next Steps

1. Refer to **README.md** for usage instructions
2. Check **CONTRIBUTING.md** for development guidelines
3. Review script docstrings for detailed parameter documentation
4. Start with basic preprocessing: `python preprocess_pipeline.py`

## System Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| RAM | 8GB | 16GB+ |
| VRAM | 6GB | 12GB+ |
| Storage | 100GB | 500GB+ |
| OS | Windows/macOS/Linux | Linux preferred |
