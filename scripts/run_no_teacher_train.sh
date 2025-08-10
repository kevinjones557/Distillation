#!/bin/bash
#SBATCH --job-name=no_teacher
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=../logs/no_teacher_train.out
#SBATCH --error=../logs/no_teacher_train.err
#SBATCH --nodelist=gilbreth-g[000-011],gilbreth-i[000-004]

# Load modules
module load conda
conda activate llama

echo "Running Python Script"

# Run script
python ../src/no_teacher_train.py
