declare -a DATASET=(
    msd05
    msd06
    msd07
    msd08
    btcv
    msd01
    msd03
    msd04
)

for dataset in ${DATASET[@]};
do
    sbatch scripts/train_${dataset}.sh
done 