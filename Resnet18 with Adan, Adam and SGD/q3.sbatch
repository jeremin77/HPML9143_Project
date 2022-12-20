#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB
#SBATCH --job-name=q3
#SBATCH --output=Adam.out

module purge

python3 mainADAM.py
