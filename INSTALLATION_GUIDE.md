# Standardised Installation Guide for MinkowskiEngine + TorchPoints3D

## 1. System Checks

```bash
uname -a
python3 --version
which python3
```

## 2. Install Miniconda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version
```

## 3. Create Conda Environment

```bash
conda create -n minkowski python=3.8 -y
conda activate minkowski
which python
python --version
```

## 4. CUDA Toolkit (11.8)

Install CUDA 11.8 from NVIDIA’s runfile: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

$CUDA_HOME/bin/nvcc --version
```

## 5. Install PyTorch with CUDA 11.8

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 6. Install OpenBLAS (needed for Minkowski)

```bash
conda install openblas-devel -c anaconda
```

## 7. Build MinkowskiEngine

```bash
# Go to home (or wherever you keep projects)
cd ~  

# Clone MinkowskiEngine repository
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

# Install with OpenBLAS + CUDA support
python setup.py install \
  --blas_include_dirs=${CONDA_PREFIX}/include \
  --blas=openblas \
  --force_cuda

```

## 8. Install TorchPoints3D

```bash
# Dependencies
conda install openblas-devel -c anaconda
pip install hydra-core==1.1 omegaconf==2.1.1 pyyaml

# Get source
git clone https://github.com/nicolas-chaulet/torch-points3d.git
cd torch-points3d

# Requirements (cleaned)
pip install -r requirements_clean.txt
```

## 9. Fix Compatibility Issues

```bash
# Reinstall torch & geometric packages for CUDA 11.8
pip uninstall torch torchvision torchaudio torch-scatter torch-sparse torch-cluster -y
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyG extensions
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Numba + LLVM fixes
conda install -c conda-forge numba=0.56.4 llvmlite=0.39.1
```