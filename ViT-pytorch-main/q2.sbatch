#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64GB
#SBATCH --job-name=q2
#SBATCH --output=SGD.out

module purge
module load cuda/11.3.1
module load cudnn/8.6.0.163-cuda11
python3 trainSGD.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16
