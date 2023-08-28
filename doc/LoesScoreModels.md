# Model 1
* data: all of Ashish's data
* Standardized RMSE: 0.7221896282431954
![Model 1](./img/model01.png "Model 1")
* correlation:    0.7137920287364632
* p-value:        1.233154773204162e-08
* standard error: 0.07224898056380212
* SLURM script: *../bin/loes-scoring-training_model01_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_02_512.pt*
* Epochs: 128
* lr: 0.0001

# Model 3
* data: all of Ashish's data
* Standardized RMSE: 0.9993048179603309
![Model 3](./img/model03.png "Model 3")
* correlation:    0.5906378280816
* p-value:        9.907172115387848e-06
* standard error: 0.06313512016386402
* SLURM script: *../bin/loes-scoring-training_model03_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_03.pt*
* Epochs: 512
* lr: 0.0001

# Model 2
* data: all of Ashish's data
* Standardized RMSE: 1.3119520507256301
![Model 2](./img/model02.png "Model 2")
* correlation:    0.6531631834747438
* p-value:        4.851578365847081e-07
* standard error: 0.06636777495526482
* SLURM script: *../bin/loes-scoring-training_model02_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_02.pt*
* Epochs: 256
* lr: 0.0001

# Model 6
* data: all of Ashish's non-Gd data
* Standardized RMSE: 1.319839811464443
![Model 6](./img/model06.png "Model 6")
* correlation:    0.8416762616021709
* p-value:        4.819603892634009e-07
* standard error: 0.062030476568710614
* SLURM script: *../bin/loes-scoring-training_model06_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_06.pt*
* Epochs: 128
* lr: 0001

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

# Model 4
* data: all of Ashish's data and Nascene's session data with QC of 1
  * 205 total cases
    * Ashish: 169
    * David: 36
* Standardized RMSE: 1.659556377232974
![Model 4](./img/model04.png "Model 4")
* correlation:    0.8326688004336145
* p-value:        2.172762045838301e-13
* standard error: 0.049556316542118485
* SLURM script: *../bin/loes-scoring-training_model04_mesabi.sh*
* Model: */home/feczk001/shared/data/AlexNet/LoesScoring/loes_scoring_04.pt*
* Epochs: 512
