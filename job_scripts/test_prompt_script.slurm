#!/bin/bash
#SBATCH --job-name=test_prompts     # Job name
# 							#SBATCH --output=job_%j_%Y-%m-%d_%H-%M-%S.out       # Standard output file (%j expands to jobId)
# 							#SBATCH --error=job_%j_%Y-%m-%d_%H-%M-%S.err        # Standard error file (%j expands to jobId)
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --nodes=1                 # Number of nodes
# 							#SBATCH --gres=gpu:1              # Number of GPUs (4 in this case)
#	 						#SBATCH --gres=gpu:a100-40G:1     # Request 4 A100-40G GPUs
#SBATCH --gres=gpu:a100:4
# 							#SBATCH --constraint=a100-40G
#SBATCH --mem=32G                 # Memory per node
#SBATCH --time=4:00:00            # Time limit hrs:min:sec
#SBATCH --partition=gpu           # Partition/queue name
# 							#SBATCH --constraint=gpu_mem:32GB



# Create datetime variable
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# Redirect output and error to files with datetime
exec 1>job_${DATETIME}_${SLURM_JOB_ID}.out
exec 2>job_${DATETIME}_${SLURM_JOB_ID}.err

# Print job info
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Load necessary modules
module purge
module load cuda/11.7
module load anaconda

# Activate conda environment
source activate ddpo

# Navigate to code directory
cd /cluster/home/mprete01/rl/text2svg-ddpo

# Add these environment variables to control device placement
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Make sure all models are initialized properly on their respective devices
accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --mixed_precision=fp16 \
  scripts/train.py --config config/test_prompts.py
 # scripts/train.py --config config/test_prompts.py resume_from=/cluster/home/mprete01/rl/text2svg-ddpo/logs/2025.04.15_19.27.05/checkpoints/checkpoint_9
