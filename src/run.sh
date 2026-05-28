#!/bin/bash
#
#SBATCH --job-name=name
#SBATCH --output=./logs/output_%j.txt
#SBATCH --error=./logs/errors_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=aias-compute-4

srun python -m ...


