#!/bin/sh

#SBATCH --job-name=loes-scoring-alex-net # job name

#SBATCH --time=16:00:00          # total run time limit (HH:MM:SS)

#SBATCH --mail-type=begin        # send 7mail when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-alex-net-%j.err
#SBATCH -o loes-scoring-alex-net-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src:/users/9/reine097/projects/AlexNet_Abrol2021/src/"
/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/dcan/training/training.py \
  --csv-data-file /users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv \
  --batch-size 1 \
  --num-workers 1 \
  --epochs 128 \
  --model-save-location /home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_13.pt \
  --plot-location /home/miran045/reine097/projects/loes-scoring-2/doc/img/model13.png \
  --output-csv-file /home/miran045/reine097/projects/loes-scoring-2/data/filtered/model13_out.csv \
  --gd 0
