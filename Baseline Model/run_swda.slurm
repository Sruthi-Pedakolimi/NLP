#!/bin/bash
#SBATCH --job-name=sst5_lstm
#SBATCH --output=logs/lstm_output_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

module purge
module load python/anaconda3
source ~/miniconda3/bin/activate team_project

python run_swda.py
