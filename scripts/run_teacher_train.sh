#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=../logs/train_new.out
#SBATCH --error=../logs/train_new.err
#SBATCH --nodelist=gilbreth-i[000-004],gilbreth-g[000-011]

# Load modules
module load conda
conda activate llama

echo "Running Python Script"

# Run script
python ../src/kldiv_train.py
