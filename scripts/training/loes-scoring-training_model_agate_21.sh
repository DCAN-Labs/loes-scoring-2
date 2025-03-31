#!/bin/sh

#SBATCH --job-name=loes-scoring-alex-net # job name

#SBATCH --mem=180g        
#SBATCH --time=16:00:00          
#SBATCH -p a100-4,a100-8
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=6      

#SBATCH --mail-user=reine097@umn.edu
#SBATCH -e loes-scoring-alex-net-%j.err
#SBATCH -o loes-scoring-alex-net-%j.out

cd /users/9/reine097/projects/loes-scoring-2/src/dcan/training || exit
export PYTHONPATH=PYTHONPATH:"/users/9/reine097/projects/loes-scoring-2/src"
/users/9/reine097/projects/loes-scoring-2/.venv/bin/python \
  /users/9/reine097/projects/loes-scoring-2/src/tuning/combined_script.py \
    --use-ray-tuning --num-samples 10 --max-num-epochs 5 \
                    --csv-input-file /users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv \
                    --model-save-location ./results/best_model.pt \
                    --plot-location best_plot.png \
                    --csv-output-file final_predictions.csv \
                    --gd 0 \
                    --folder /home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/ \
                --use-train-validation-cols
