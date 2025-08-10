#!/bin/bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/sample.out
#SBATCH --error=../logs/sample.err

# Load modules
module load conda
conda activate llama

echo "Running Python Script"

# Run script
python ../src/sample.py
