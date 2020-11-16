#!/bin/bash
#SBATCH --partition=SCSEGPU_UG
#SBATCH --job-name="can-lstm"
#SBATCH --output=cnn-lstm.out
#SBATCH --error=cnn-lstm.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M

module load anaconda
python CNN-lstm.py