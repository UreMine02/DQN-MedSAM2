declare -a DATASET=(
    btcv
    # sarcoma
    # msd01
    msd02
    msd03
    # msd04
    # msd05
    # msd06
    # msd07
    # msd08
    # msd09
    # msd10
    # msd
)

for dataset in ${DATASET[@]};
do
    sbatch scripts/train_${dataset}.sh
done