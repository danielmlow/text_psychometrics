#!/bin/bash
#SBATCH --job-name=gemma-2b-it
#SBATCH --output=/om2/user/dlow/job_logs/job_%j.out
#SBATCH --output=/om2/user/dlow/slurm/slurm-%A.out
#SBATCH --error=/om2/user/dlow/job_logs/job_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
###SBATCH --gres=gpu:QUADRORTX6000:1 
#SBATCH --constraint=any-gpu
#SBATCH --time=6:30:00
#SBATCH --mail-user=dlow@mit.edu
#SBATCH --mail-type=ALL 

source activate /om2/user/dlow/anaconda/envs/rallypoint_stb_detector

# Define the arguments
TOY=0
MODEL_NAME="google/gemma-2b-it"  
# model_name = "google/gemma-7b-it"
WITH_INTERACTION=1

srun python /home/dlow/datum/zero_shot/open_source_genai.py "$TOY" "$MODEL_NAME" "$WITH_INTERACTION"

echo >> "Finished."
