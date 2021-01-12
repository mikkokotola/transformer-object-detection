#!/bin/bash
#SBATCH -o /scratch/project_2000924/detr/output/train_detr_full.txt
#SBATCH --job-name=train_detr_full
#SBATCH --account=project_2000924
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:v100:1

module load pytorch/1.6

srun python3 /scratch/project_2000924/detr/python/model_trainer_fullDetr.py