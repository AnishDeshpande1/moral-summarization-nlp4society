#!/bin/bash
#SBATCH --job-name=moral-deepseek32b
#SBATCH --partition=gpu-v100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/deepseek32b_%j.out
#SBATCH --error=slurm/logs/deepseek32b_%j.err

# Inference: DeepSeek-R1-Distill-Qwen-32B on one V100S 32GB.
# Submit from the project root:  sbatch slurm/deepseek32b_v100.sbatch
# Runs in parallel with the Llama-70B A100 job; writes a different model_tag so
# their outputs never collide in results/test_prompts/.

set -euo pipefail

module load 2023r1
module load cuda/12.1
module load miniconda3

conda activate moral_env

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

srun python prompting.py \
  --config-file configs/deepseek32b_v100.yaml \
  --hf-model-folder /scratch/$USER/huggingface-models \
  --prompt-dir results/test_prompts \
  --no-testing
