#!/bin/bash
#SBATCH --job-name=continued_training_CE
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=../logs/continued_training_CE.out
#SBATCH --error=../logs/continued_training_CE.err
#SBATCH --nodelist=gilbreth-g[000-011],gilbreth-i[000-004]

# Load modules
module load conda
conda activate llama

# Run script
python ../src/continued_cross_entropy_train.py

if [ -f ../still_training_CE.flag ]; then
    echo "Training not finished, resubmitting..."
    sbatch ../scripts/continued_training_CE.sh
else
    echo "Training finished or error."
fi