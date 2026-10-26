#!/bin/bash -l
#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 32 # num cpus
#SBATCH --gres=gpu:4 # num gpus
#SBATCH --mem=200GB # ram
#SBATCH --time=12:00:00 # time
#SBATCH -J msd03 # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/msd03-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2
# conda init
# conda activate rlsam2

# One rendezvous port per job: submit_jobs.sh launches several -distributed jobs at
# once, and two of them on one node would otherwise both claim the default 12348.
MASTER_PORT=$((20000 + ${SLURM_JOB_ID:-0} % 20000))

# If a collective stalls, the NCCL watchdog fails the run after the 30-min timeout in
# train_3d.setup() and dumps each rank's last collectives (op, size, sequence number) to
# these files; diffing them shows which rank stopped where.
mkdir -p ./output/nccl_trace
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=./output/nccl_trace/${SLURM_JOB_ID:-local}_rank_
export NCCL_DEBUG=WARN
echo "node $(hostname), job ${SLURM_JOB_ID:-local}, port $MASTER_PORT"

for SEED in 0;
do
    for FOLD in 1 2 3 4;
    do
    EXP=msd_task03+icl+global_pool8_diverse+group_size12_group_std+rw_thr1e-3/seed${SEED}/fold${FOLD}
    python train_3d.py \
        -exp_name $EXP \
        -sam_config sam2_hiera_t \
        -sam_ckpt /data/rlsam2/checkpoints/sam2_hiera_tiny.pt \
        -rl_config rl_modules/config/grpo_po_agent.yaml \
        -checkpoint_path ./output/$EXP \
        -dataset msd \
        -task Task03 \
        -data_path /data/rlsam2/datasets/nii/ \
        -lr 2e-4 -val_freq 50 -ep 50 -warmup_ep 0 -stop_sam2_ep -1 \
        -num_support 5 -memory_bank_size 6 \
        -pool_size 8 -pool_stride 1 -pool_policy diverse -pool_novelty_iou 0.05 -recall_every 1 \
        -agent_act_every 1 -rl_group_size 12 -agent_lr_T_max 100 -q_updates_per_step 4 \
        -rl_reward_threshold 0.001 \
        -fold $FOLD -n_folds 5 \
        -seed 0 \
        -wandb_enabled \
        -distributed -master_port $MASTER_PORT
    done
done
    