#!/bin/sh

#SBATCH --job-name=loes-scoring-512 # job name

#SBATCH --mem=180g        # memory per cpu-core (what is the default?)
#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p v100
#SBATCH --gres=gpu:v100:2
#SBATCH --ntasks=6               # total number of tasks across all nodes

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-alex-net-%j.err
#SBATCH -o loes-scoring-alex-net-%j.out

#SBATCH -A miran045

cd /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/loes-scoring-2/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training/training.py \
  --csv-data-file "/home/miran045/reine097/projects/loes-scoring-2/data/all_loes_score_data_qc_1_2.csv" \
  --anatomical-region all --batch-size 5 --num-workers 1 --epochs 256 \
  --model-save-location "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_04.pt" \
  --plot-location "/home/miran045/reine097/projects/loes-scoring-2/doc/img/loes_scoring_04.png" \
  --file-path-column-index 0 --loes-score-column-index 3
