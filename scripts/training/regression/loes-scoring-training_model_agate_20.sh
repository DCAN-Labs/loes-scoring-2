#!/bin/sh

#SBATCH --job-name=loes-scoring-alex-net # job name

#SBATCH --mem=180g        
#SBATCH --time=1:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-%j.err
#SBATCH -o loes-scoring-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src:/users/9/reine097/projects/AlexNet_Abrol2021/src/"
/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/dcan/regression/training.py \
  --csv-input-file "/users/9/reine097/projects/loes-scoring-2/data/regression_training_validation.csv" \
                        --batch-size 1 --num-workers 1 --epochs 64 \
                        --model-save-location "/home/feczk001/shared/data/LoesScoring/loes_scoring_20.pt" \
                        --gd  0 \
                        --folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
                        --csv-output-file "/users/9/reine097/projects/loes-scoring-2/doc/models/model20/model20.csv" \
                        --plot-location "/users/9/reine097/projects/loes-scoring-2/doc/models/model20/model20.png"  \
                        --model "ResNet" \
                        --lr 0.0001 \
                        --scheduler 'onecycle' \
                        --use-train-validation-cols
                         