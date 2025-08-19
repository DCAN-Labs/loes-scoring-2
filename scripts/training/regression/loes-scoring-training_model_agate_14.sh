#!/bin/sh

#SBATCH --job-name=loes-scoring # job name

#SBATCH --mem=180g        
#SBATCH --time=2:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-%j.err
#SBATCH -o loes-scoring-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/regression || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src"
/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/dcan/regression/training.py \
                        --batch-size 1 --num-workers 1 --epochs 256 \
                        --model-save-location "/home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_14_512.pt" \
                        --csv-output-file  "/users/9/reine097/projects/loes-scoring-2/data/filtered/model14_out_512.csv" \
                        --gd  0 \
                        --folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
                        --csv-input-file "/users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv" \
                        --plot-location "/users/9/reine097/projects/loes-scoring-2/doc/models/model14/model14_512.png" \
                        --model "resnet" \
                        --lr 0.0001
