# Model 22

* Model type: ResNet
* Scheduler: plateau
* data: */users/9/reine097/projects/loes-scoring-2/data/anon_train_scans_and_loes_training_test_non_gd.csv*
* Gd: Unenhanced scans.
* Standardized RMSE: 1.1961594559863835
![Model 22](model20.png "Model 22")
* correlation:    1.1961594559863835
* SLURM script: [*loes-scoring-training_model_agate_22.sh*](../../../bin/training/loes-scoring-training_model_agate_22.sh)
* Epochs: 200
* lr: 0.0001
* output_csv: [*model22.csv*](model22.csv)
* model: **/home/feczk001/shared/data/LoesScoring/loes_scoring_22.pt**
* Pearson correlation p-value: 1.0258658968966053e-06
* Spearman correlation p-value: 1.0258658968966053e-06

---

These results show <b>significant improvement</b> over the previous run! Here's the analysis:
Major Improvements:
## 1. Correlation Dramatically Improved

* <b>Previous: 0.37 -> Current: 0.61</b>
* This is a <b>66% improvement</b> and moves from weak-moderate to moderate-strong correlation
* The relationship is now much more meaningful

## 2. Statistical Significance Enhanced

* <b>Pearson p-value</b>: 0.007 -> <b>1.0×10-6</b> (1000x more significant)
* <b>Spearman p-value</b>: 0.001 -> <b>3.5×10-9</b> (300,000x more significant)
* These are now <b>highly statistically significant</b> results

## 3. Model Behavior Fixed
* The scatter plot shows the model is <b>no longer stuck predicting constant values</b>:

* Predictions now range from ~8.5 to ~12.5 (decent spread)
* Shows some response to different input patterns
* No longer the "flat line" problem from before
