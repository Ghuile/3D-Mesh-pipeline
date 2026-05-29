# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

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

### 4. Verify Installation

```bash
python -c "import torch; import open3d; print('Installation successful!')"
```

## Troubleshooting

### PyTorch Installation Issues

If you encounter issues with PyTorch, install the appropriate version for your system:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

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

## GPU Setup (Optional)

For GPU acceleration, ensure CUDA toolkit is installed:

```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Data Setup

Create directories for data storage:

```bash
mkdir -p data_generation data_parsed data_sanitized
```

Note: These directories should remain empty when pushing to repository.

## Next Steps

Once installation is complete, refer to README.md for usage instructions.
