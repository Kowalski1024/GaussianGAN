#!/bin/bash -l
#SBATCH -J stylegan
#SBATCH --mem=16G
#SBATCH --time=00:20:00 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgmultiplannerf-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="test.out"
#SBATCH --error="error.err"


module add CUDA/11.7.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/11.7.0

DATA_PATH="/net/pr2/projects/plgrid/plggtriplane/twojnar/cars_128.zip"

nvidia-smi -L
source activate gauss

nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!
python train.py --outdir=training-runs --gpus=1 --cfg="stylegan2" --data=$DATA_PATH --batch=32 --gamma=0.5 --glr 0.001 --dlr 0.0008 --cond=True --name "gnn-athena" --wandb "disabled" & 
P2=$!
wait $P2
kill $P1