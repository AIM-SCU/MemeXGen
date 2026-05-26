#!/bin/bash
# Example SLURM submission script for the Qwen2.5-VL-72B-AWQ judge over the
# LLaVA+FLUX (our) pipeline output. Adapt partition / module / conda activation
# to your cluster.

#SBATCH --job-name=qwen72b-llava
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llava-%j.out
#SBATCH --error=logs/llava-%j.err

# Cluster-specific setup (replace with your environment).
# module load CUDA/12.6.2
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate qwen72b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$(dirname "$0")/.."
python run_generator.py --generator llava
