# Model 13
## 512 epochs

ResNet (instead of AlexNet) with a learning rate of 0.0001.

* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv*
* Gd: Unenhanced scans.
* Standardized RMSE: 0.6572768934698772
![Model 13 (513 epochs)](model12_13.png "Model 1")
* correlation:    0.8379666294613598
* SLURM script: [*loes-scoring-training_model_agate_13.sh*](../../../../bin/training/loes-scoring-training_model_agate_13_512.sh)
* Epochs: 512
* lr: 0.001
* output_csv: [*model12_512.csv*](../512_epochs/model12_512.csv)
* model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_12_512.pt*
* Pearson correlation p-value: 3.396958748233429e-10
* Spearman correlation p-value: 1.9223338978277814e-06
