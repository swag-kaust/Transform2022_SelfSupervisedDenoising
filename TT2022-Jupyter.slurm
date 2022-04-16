#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --partition=batch
#SBATCH --constraint=[v100]

# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# GPU jupyter notebook setup on Ibex
# Original Author: Matteo Ravasi
# Adapted for Transform Tutorial by: Claire Birnie
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

## Move to directory:
cd /ibex/scratch/birniece/Transform2022_SelfSupervisedDenoising

## Setup environment
source activate tt2022ssd

# get tunneling info
export XDG_RUNTIME_DIR="" node=$(hostname -s)
user=$(whoami)
submit_host=${SLURM_SUBMIT_HOST}
port=10000
echo $node pinned to port $port
# print tunneling instructions

echo -e "
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1.
Command to create ssh tunnel from you workstation/laptop to glogin:
ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa
Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop "

# Run Jupyter
jupyter notebook --no-browser --port=${port} --port-retries=50 --ip=${node}