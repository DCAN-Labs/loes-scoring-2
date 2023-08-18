# Model 5
* data: all of Ashish's Gd-enhanced data
* Standardized RMSE: 1.3465736121901255
![Model 5](./img/model05.png "Model 5")
* correlation:    0.6639005760048992
* p-value:        0.0014129271260232773
* standard error: 0.09497052216668729
* SLURM script: *../bin/loes-scoring-training_model05_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_05.pt*
* Epochs: 512

# Model 3
* data: all of Ashish's data
* Standardized RMSE: 0.8681073956588595
![Model 3](./img/model03.png "Model 3")
* correlation:    0.843721682323223
* p-value:        2.9023678644837652e-11
* standard error: 0.05826465079588512
* SLURM script: *../src/dcan/training/loes-scoring-training_model03_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_03.pt*
* Epochs: 512

# Model 2
* data: all of Ashish's data
* Standardized RMSE: 1.4974884284668133
![Model 2](./img/model02.png "Model 2")
* correlation:    0.861177276333397
* p-value:        6.596685767672446e-08
* standard error: 0.07132685112047961
* SLURM script: *../src/dcan/training/loes-scoring-training_model02_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_02.pt*
* Epochs: 256

# Model 1
* data: all of Ashish's data
* Standardized RMSE: 1.206785682955434
![Model 1](./img/model01.png "Model 1")
* correlation:    0.8062695135102309
* p-value:        6.18675208209489e-07
* standard error: 0.11887297424632068
* SLURM script: *../src/dcan/training/loes-scoring-training_model01_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_02_512.pt*
* Epochs: 128

# Model 6
* data: all of Ashish's non-Gd data
* Standardized RMSE: 2.0317264758139366
![Model 6](./img/model06.png "Model 1")
* correlation:    0.6802879190059432
* p-value:        0.0001827859542662024
* standard error: 0.07425917812043135
* SLURM script: *../src/dcan/training/loes-scoring-training_model06_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_06.pt*
* Epochs: 128

# Model 4
* data: all of Ashish's data and Nascene's session data with QC of 1
  * 205 total cases
    * Ashish: 169
    * David: 36
* Standardized RMSE: 2.0390140967745025
![Model 4](./img/model04.png "Model 4")
* correlation:    0.011185159238607002
* p-value:        0.9366429419513976
* standard error: 0.08188842704604608
* SLURM script: *../src/dcan/training/loes-scoring-training_model04_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_04.pt*
* Epochs: 512
