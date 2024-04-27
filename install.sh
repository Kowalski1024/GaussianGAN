#!/bin/bash -l
#SBATCH -J stylegan
#SBATCH --mem=32G
#SBATCH --time=05:00:00 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgmultiplannerf-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

module add CUDA/11.8.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/11.8.0

nvidia-smi -L

source activate gauss
# conda env create -p /net/pr2/projects/plgrid/plggtriplane/twojnar/gauss -f environment.yml
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git