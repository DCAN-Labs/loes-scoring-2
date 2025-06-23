# Model 20

Used one cycle scheduler.

            scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=0.01,
                total_steps=len(train_dl) * self.config.epochs,
                pct_start=0.3
            )

---
* standardized_rmse: 0.7510329801792275
* correlation_coefficient: 0.7395470599233053
* Pearson correlation p-value: 4.2111563055899623e-16
* Spearman correlation p-value: 5.516317385333862e-14
---
* Model type: ResNet
* Scheduler: OneCycleLR
* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv*
* Gd: Unenhanced scans.
]![Model 20](model20.png "Model 20")
* SLURM script: [*loes-scoring-training_model_agate_20.sh*](../../../bin/training/loes-scoring-training_model_agate_20.sh)
* Epochs: 256
* lr: 0.0001
* output_csv: [*model20.csv*](model20.csv)
* model: */home/feczk001/shared/data/LoesScoring/loes_scoring_20.pt*
