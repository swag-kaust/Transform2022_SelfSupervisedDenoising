#!/bin/bash
#
# Installer for PyTorch GPU environment with Pytorch+Cupy+PyTorch and CUDA 10.2
#
# Run: ./install_tt2022ssd.sh
#
# C. Birnie, 16/09/2021

echo 'Creating PyTorch environment with Pytorch and CUDA 10.2'

# create conda env
conda env create -f environment_tt2022ssd.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tt2022ssd
conda env list
echo 'Created and activated environment:' $(which python)

# check cupy works as expected
echo 'Checking cupy version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda"))'

echo 'Done!'