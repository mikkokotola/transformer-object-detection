#!/bin/bash
#SBATCH -o train_testing.txt
#SBATCH --job-name=train_testing_detr
#SBATCH --account=project_2000924
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2000
#SBATCH --gres=gpu:v100:1

module load pytorch/1.6

srun python3 /scratch/project_2000924/detr/python/model_trainer.py