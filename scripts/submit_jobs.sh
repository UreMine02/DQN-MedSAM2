declare -a DATASET=(
    msd05
    msd07
    msd08
    msd01
    msd03
    msd04
)

for dataset in ${DATASET[@]};
do
    sbatch scripts/train_${dataset}.sh
done 