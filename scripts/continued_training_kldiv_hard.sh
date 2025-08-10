#!/bin/bash
#SBATCH --job-name=hard_kldiv
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=../logs/hard_kldiv.out
#SBATCH --error=../logs/hard_kldiv.err
#SBATCH --nodelist=gilbreth-g[000-011],gilbreth-i[000-004]

# Load modules
module load conda
conda activate llama

# Run script
python ../src/continued_kldiv_train_hard.py

if [ -f ../still_training_kldiv_hard.flag ]; then
    echo "Training not finished, resubmitting..."
    sbatch ../scripts/continued_training_kldiv.sh
else
    echo "Training finished or error."
fi