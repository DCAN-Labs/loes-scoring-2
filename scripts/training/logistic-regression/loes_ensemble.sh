#!/bin/sh

#SBATCH --job-name=loes-ensemble
#SBATCH --mem=180g        
#SBATCH --time=4:00:00          
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=4      
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-ensemble-%j.err
#SBATCH -o loes-ensemble-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src:/users/9/reine097/projects/AlexNet_Abrol2021/src/"

/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/dcan/training/ensemble_predict.py \
    --csv-input-file "/users/9/reine097/projects/loes-scoring-2/data/logistic_regression_data.csv" \
    --csv-output-file "ensemble_predictions.csv" \
    --folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
    --model1-path "/users/9/reine097/projects/loes-scoring-2/src/dcan/training/model_aug18.pt" \
    --model2-path "/users/9/reine097/projects/loes-scoring-2/src/dcan/training/model_aug22.pt" \
    --ensemble-weights "0.6,0.4" \
    --threshold "0.6"