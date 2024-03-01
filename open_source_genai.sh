#!/bin/bash
#SBATCH --job-name="genai"
#SBATCH --nodes=1
#SBATCH --gres=gpu:QUADRORTX6000:1
#SBATCH --time=00:15:00     # walltime
#SBATCH --output=/om2/user/dlow/slurm/slurm-%A.out
#SBATCH --error=/om2/user/dlow/job_logs/job_%j.err
#SBATCH --open-mode=append
#SBATCH --mail-user=dlow@mit.edu
#SBATCH --mail-type=BEGIN

source activate /om2/user/dlow/anaconda/envs/rallypoint_stb_detector
srun python /home/dlow/datum/zero_shot/open_source_genai.py

echo "Finished"