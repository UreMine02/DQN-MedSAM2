#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=2:00:00 # time
#SBATCH -J msd02 # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/DynaFold/Generative-Dyna/msd10-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/DynaFold/Generative-Dyna
# conda init
# conda activate rlsam2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$REPO_ROOT" || exit 1

DATA_PATH_MSD="${DATA_PATH_MSD:-$REPO_ROOT/../datasets/Combined_Dataset/MSD}"
RL_ABLATION_METHOD="${RL_ABLATION_METHOD:-default}"

EXP="${EXP:-msd_task10+rlsam2+grpo+inter+cw_soft_gating+obj_ptr_gating+highres_gating_by_lowres_and_ptr}"
SAM_CKPT="${SAM_CKPT:-$REPO_ROOT/checkpoint/sam2_hiera_tiny.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python train_3d.py \
    -exp_name "$EXP" \
    -sam_config sam2_hiera_t \
    -sam_ckpt "$SAM_CKPT" \
    -rl_config rl_modules/config/grpo_po_agent.yaml \
    -rl_ablation_method "$RL_ABLATION_METHOD" \
    -checkpoint_path "./output/$EXP" \
    -dataset msd \
    -task Task10_Colon \
    -memory_bank_size 16 \
    -data_path "$DATA_PATH_MSD" \
    -video_length 16 \
    -num_workers 4 \
    -lr 1e-4 \
    -val_freq 1 \
    -ep 500 \
    -q_updates_per_step 1 \
    -lazy_penalty 0.0 \
    -invalid_penalty -0.01 \
    -num_support 3 \
    -gating_dimension "cw" \
    -gating_softness "soft" \
    -auxiliary_loss "no" \
    -wandb_enabled \
    -obj_ptr_gating \
    -highres_gating "by_lowres" \
