#!/bin/bash -l

#SBATCH -p a100 # keep as is
#SBATCH -N 1 # keep as is
#SBATCH -n 8 # num cpus
#SBATCH --gres=gpu:1 # num gpus
#SBATCH --mem=100GB # ram
#SBATCH --time=24:00:00 # time
#SBATCH -J eval # job name
#SBATCH -A strategic
#SBATCH -o "/hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/eval-btcv-%j.out"

# conda activate rlsam2
# cd /hpcfs/users/a1232079/duyanh/MedSAM2/code/DQN-MedSAM2/
# conda init
# conda activate rlsam2

declare -a ckpts=(
    # Sarcoma
    # output/sarcoma+ppo+sam2rl/2026-05-06-23-41-48/best.pth

    # MSD-Heart
    # output/msd_task02+ppo+sam2rl/2026-05-07-05-29-23/best.pth

    # MSD-Prostate
    # output/msd_task05+ppo+sam2rl/2026-05-06-23-41-28/best.pth

    # MSD-Lung
    # output/msd_task06+ppo+sam2rl/2026-05-06-19-23-41/best.pth

    # MSD-Spleen
    # output/msd_task09+ppo+sam2rl/2026-05-07-08-04-02/best.pth
    output/msd_task09+ppo+sam2rl/2026-05-07-13-48-40/best.pth

    # MSD-Colon
    # output/msd_task10+ppo+sam2rl/2026-05-06-18-53-29/best.pth
)

declare -a datasets=(
    # msd
    msd
)

declare -a tasks=(
    # "Task02"
    "Task09"
)

declare -a no_agents=()

for i in ${!ckpts[@]};
do
    # echo $i
    # datasets+=("msd")
    # tasks+=("Task10")
    no_agents+=("agent")
done

# echo $datasets

export CUDA_VISIBLE_DEVICES=1

for idx in ${!ckpts[@]}
do
    # echo ${ckpts[idx]}
    # echo ${tasks[idx]}
    # echo ${no_agents[idx]}
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain "${ckpts[idx]}" \
            -rl_config rl_modules/config/ppo_po_agent.yaml \
            -dataset "${datasets[idx]}" \
            -task "${tasks[idx]}" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -ablation \
            -vis \
            $( [ ${no_agents[idx]} = "no_agent" ] && echo "-no_agent" )
    done
done
