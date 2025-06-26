# Model 19

Used cosine scheduler.

            scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=0)
---
* Model type: ResNet
* Scheduler: CosineAnnealingLR
* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv*
* Gd: Unenhanced scans.
* Standardized RMSE: 0.509832374933453
* correlation:    0.8736231057896536
* SLURM script: [*loes-scoring-training_model_agate_19.sh*](../../../bin/training/loes-scoring-training_model_agate_19.sh)
* Epochs: 256
* lr: 0.0001
* output_csv: [*model19.csv*](model19.csv)
* model: */home/feczk001/shared/data/LoesScoring/loes_scoring_19.pt*
* Pearson correlation p-value: 7.431391320831846e-12
* Spearman correlation p-value: 2.5486362799149536e-06
