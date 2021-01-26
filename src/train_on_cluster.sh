#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/versa/src/"
AT="@"

# Test the job before actually submitting
# SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

RUNS_PATH="${SOURCE_PATH}/slurm_logs/"
echo $RUNS_PATH
mkdir -p $RUNS_PATH


"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}%J_slurm.out"
#SBATCH --error="${RUNS_PATH}%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="poklukar${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor|balrog|shelob|smaug"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate base
nvidia-smi

python3 run_classifier.py --mode "train_test" --dataset "celebA" \
                          --d_theta 64 \
                          --way 2 --shot 5 --test_shot 5 --test_way 2 \
                          --tasks_per_batch 6 --samples 10 \
                          --learning_rate 1e-4 --iterations 10000 \
                          --dropout 0.9 \
                          --checkpoint_dir "models/way2shot5" \
                          --print_freq 1000
                          

HERE

# python3 run_classifier.py --mode "train_test" --dataset "celebA" --d_theta 64 --way -2 --shot 5 --tasks_per_batch 6 --samples 10 --learning_rate 1e-4 --iterations 10 --dropout 0.9 --checkpoint_dir "models/way2shot5" --print_freq 1