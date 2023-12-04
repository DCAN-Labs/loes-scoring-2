#!/bin/bash
sbatch <<EOT
#!/bin/sh

#SBATCH --job-name=loes-scoring-alex-net # job name

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
export PYTHONPATH=PYTHONPATH:"/home/miran045/reine097/projects/loes-scoring-2/src:/home/miran045/reine097/projects/AlexNet_Abrol2021/src"
/home/miran045/reine097/projects/AlexNet_Abrol2021/venv/bin/python \
  /home/miran045/reine097/projects/loes-scoring-2/src/dcan/training/training.py \
  --csv-data-file "/home/miran045/reine097/projects/loes-scoring-2/data/filtered/ashish_all.csv" \
  --batch-size 1 --num-workers 1 --epochs 256 \
  --model-save-location "/home/feczk001/shared/data/AlexNet/LoesScoring/five_fold/loes_scoring_${1}.pt" \
  --output-csv-file "/home/miran045/reine097/projects/loes-scoring-2/doc/models/five_fold_validation/output${1}.csv" --gd 2 \
  --plot-location "/home/miran045/reine097/projects/loes-scoring-2/doc/models/five_fold_validation/correlation${1}.png" -k $1
EOT
