#!/bin/sh

#SBATCH --job-name=loes-scoring-logistic-regression
#SBATCH --mem=180g        
#SBATCH --time=12:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      
#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-logistic-regression-%j.err
#SBATCH -o loes-scoring-logistic-regression-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src:/users/9/reine097/projects/AlexNet_Abrol2021/src/"

/users/9/reine097/projects/loes-scoring-2/.venv/bin/python /users/9/reine097/projects/loes-scoring-2/src/dcan/training/logistic_regression.py \
    --csv-input-file  "/users/9/reine097/projects/loes-scoring-2/data/logistic_regression_data.csv" \
    --csv-output-file "/users/9/reine097/projects/loes-scoring-2/doc/models/logistic_regression/model01/predictions.csv" \
    --model-save-location "/home/feczk001/shared/data/LoesScoring/logistic_regression/model01.pt" \
    --batch-size "4" \
    --epochs "50" \
    --features "scan" \
    --target "cald_develops" \
     --folder "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/" \
  --use_train_validation_cols \
    --model-type "resnet3d" \
    --lr "0.001" \
    --weight-decay "0.001" \
    --threshold "0.795" \
    --augment-minority \
    --num-augmentations 3 \
    --plot-location "/users/9/reine097/projects/loes-scoring-2/doc/models/logistic_regression/model01/roc"
