declare -a DATA=(
    Task07
    output/msd_task07+no_agent+icl+long_horizon+no_augment/2026-04-28-20-07-36/best.pth
    no_agent
    Task08
    output/msd_task08+no_agent+icl+long_horizon+no_augment/2026-04-28-20-20-31/best.pth
    no_agent
)

for ((i=0; i<${#DATA[@]}; i+=3)); 
do
    for shot in 5;
    do
        python eval_3d.py \
            -pretrain "${DATA[i+1]}" \
            -rl_config rl_modules/config/grpo_po_agent.yaml \
            -dataset msd \
            -task "${DATA[i]}" \
            -data_path /data/datasets/nii/ \
            -num_support $shot \
            -memory_bank_size 6 \
            -ablation \
            -vis \
            $( [ "${DATA[i+2]}" = "no_agent" ] && echo "-no_agent" )
    done
done
