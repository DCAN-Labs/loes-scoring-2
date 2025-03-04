# Model 20

Used one cycle scheduler.

            scheduler = OneCycleLR(
                self.optimizer, 
                max_lr=0.01,
                total_steps=len(train_dl) * self.config.epochs,
                pct_start=0.3
            )

---
* Model type: ResNet
* Scheduler: OneCycleLR
* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes.csv*
* Gd: Unenhanced scans.
* Standardized RMSE: 0.5091772048887064
![Model 20](model20.png "Model 20")
* correlation:    0.8565618361480323
* SLURM script: [*loes-scoring-training_model_agate_20.sh*](../../../bin/training/loes-scoring-training_model_agate_20.sh)
* Epochs: 256
* lr: 0.0001
* output_csv: [*model20.csv*](model20.csv)
* model: */home/feczk001/shared/data/LoesScoring/loes_scoring_20.pt*
* Pearson correlation p-value: 5.259068091329828e-11
* Spearman correlation p-value: 2.8141077950001913e-06
