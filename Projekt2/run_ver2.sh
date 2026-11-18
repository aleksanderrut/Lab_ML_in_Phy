#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=10gb
#SBATCH --time=0-00:29:00
#SBATCH --job-name=CNN
#SBATCH -p lem-gpu
#SBATCH --gres=gpu:hopper:1
#SBATCH --account=hpc-jarpaw-1791296019

##apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif python -u CNNs_ssh_train.py
apptainer exec --nv /lustre/software-data/container-images/pytorch_latest.sif python -u CNN_ver2.py
