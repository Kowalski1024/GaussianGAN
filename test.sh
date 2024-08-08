#!/bin/bash -l
#SBATCH -J testrun
#SBATCH --mem=64G
#SBATCH --time=00:15:00 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -A plgvdgs-gpu-a100
#SBATCH -p plgrid-gpu-a100


module add CUDA/11.7.0
module add Miniconda3/23.3.1-0
module add GCC/10.3.0
module load Ninja/1.10.2
export CUDA_HOME=/net/software/v1/software/CUDA/11.7.0

DATA_PATH="/net/people/plgrid/plgkowalski1024/cars/"
# DATA_PATH="/net/pr2/projects/plgrid/plggtriplane/twojnar/cars_128.zip"

nvidia-smi -L
conda env list
source activate gauss

nvidia-smi -l 60 --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv &
P1=$!
python train.py --outdir=training-runs --gpus=1 --cbase=16384 --cfg="stylegan2" --data=$DATA_PATH --batch=16 --gamma=1 --glr 0.0001 --dlr 0.002 --cond=True --name "gnn-athena" --wandb "disabled" & 
P2=$!
wait $P2
kill $P1