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
export CUDA_VISIBLE_DEVICES=0
/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/dcan/regression/training.py \
  --csv-input-file "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv" \
                        --batch-size 2 --num-workers 1 \
                        --model-save-location "/home/feczk001/shared/data/LoesScoring/loes_scoring_25.pt" \
                        --gd  0 \
                        --folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
                        --csv-output-file "/users/9/reine097/projects/loes-scoring-2/doc/models/model25/model25.csv" \
                        --plot-location "/users/9/reine097/projects/loes-scoring-2/doc/models/model25/model25.png"  \
                        --model "resnet" \
                        --epochs 200
