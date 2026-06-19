#!/bin/bash
#SBATCH --job-name=moral-llama70b
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/llama70b_%j.out
#SBATCH --error=slurm/logs/llama70b_%j.err

# Inference: Llama-3-70B-Instruct on one A100 80GB.
# Submit from the project root:  sbatch slurm/llama70b_a100.sbatch
# Adjust --account/--partition to your DelftBlue allocation if needed.

set -euo pipefail

module load 2023r1
module load cuda/12.1        # match the torch build (torch 2.3.0+cu121)
module load miniconda3

# Activate the project conda env (created per the README).
conda activate moral_env

# Keep all HF/torch caches on scratch (home quota is small).
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_OFFLINE=1   # weights are pre-downloaded; don't hit the network
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

srun python prompting.py \
  --config-file configs/llama70b_a100.yaml \
  --hf-model-folder /scratch/$USER/huggingface-models \
  --prompt-dir results/test_prompts \
  --no-testing
