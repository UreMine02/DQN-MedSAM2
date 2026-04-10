declare -a DATASET=(
    msd03
    msd04
    msd05
    msd06
    msd07
    msd08
)

for dataset in ${DATASET[@]};
do
    sbatch scripts/train_${dataset}.sh
done 