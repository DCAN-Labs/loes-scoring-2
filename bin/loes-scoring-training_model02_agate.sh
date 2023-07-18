#!/bin/sh

#SBATCH --job-name=loes-scoring-512 # job name

#SBATCH --mem=180g        # memory per cpu-core (what is the default?)
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-alex-net-%j.err
#SBATCH -o loes-scoring-alex-net-%j.out

module load gcc cuda/11.2
source /panfs/roc/msisoft/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh
conda activate /home/support/public/torch_cudnn8.2

cd /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/loes-scoring-2/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training/training.py \
  --csv-data-file "/home/miran045/reine097/projects/loes-scoring-2/data/all_loes_score_data.csv" \
  --anatomical-region all --batch-size 5 --num-workers 1 --epochs 128 \
  --model-save-location "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_02_512_2.pt" \
  --plot-location "/home/miran045/reine097/projects/loes-scoring-2/doc/img/model02_512_2.png"
