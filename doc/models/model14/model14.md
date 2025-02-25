# Model 14
## 256 epochs

Use a weighted loss function.

* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv*
* Gd: Unenhanced scans.
* Standardized RMSE: 0.6285900047316173
![Model 14](model14.png "Model 14")
* correlation:    0.8321506161210428
* SLURM script: [*loes-scoring-training_model_agate_14.sh*](../../../bin/training/loes-scoring-training_model_agate_14.sh)
* Epochs: 256
* lr: 0.0001
* output_csv: [*model14.csv*](model14.csv)
* model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_14_512.pt*
* Pearson correlation p-value: 5.805923285303704e-10
* Spearman correlation p-value: 7.034041732339813e-06
