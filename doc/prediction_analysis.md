Analysis of Loes score predictions
==================================

All files are in:
    
    /home/feczk001/shared/data/loes_scoring/nascene_deid/BIDS/anonymized_names/

Gadolinium enhanced
-------------------

### Best predictions
                                                                               base_name       error
          sub-03_session-06_space-MNI_101_sub-03b_deidentified_SAG_T1_MPRAGE_0919.nii.gz    1.919110
               sub-03_session-09_space-MNI_100_sub-03b_deidentified_SAG_T1_MPRAGE.nii.gz    1.919129
    s ub-03_session-06_space-MNI_102_sub-03b_deidentified_SAG_T1_MPRAGE_Post_0919.nii.gz    1.919129
          sub-03_session-09_space-MNI_103_sub-03b_deidentified_SAG_T1_MPRAGE_Post.nii.gz    1.919129
               sub-03_session-09_space-MNI_101_sub-03b_deidentified_SAG_T1_MPRAGE.nii.gz    1.919129

### Worst predictions
                                                                         base_name         error
         sub-04_session-03_space-MNI_024_sub-04a_deidentified_MPRAGE_SAG_GD.nii.gz    848.416992
          sub-02_session-00_space-MNI_103_sub-02_deidentified_MPRAGE_SAG_GD.nii.gz    830.719238
          sub-02_session-04_space-MNI_022_sub-02_deidentified_MPRAGE_SAG_GD.nii.gz    829.751465
     sub-02_session-05_space-MNI_018_sub-02_deidentified_SAG_T1_MPRAGE_post.nii.gz    825.035522
     sub-05_session-10_space-MNI_024_sub-05_deidentified_SAG_T1_MPRAGE_Post.nii.gz    778.190552

### Best predictions by subject

    sub-01
                                                                                   base_name        error
       sub-01_session-00_space-MNI_002_sub-01_deidentified_18_PEDI_BRAIN_MPRage_SAGIT.nii.gz    60.329544
       sub-01_session-01_space-MNI_002_sub-01_deidentified_16_PEDI_BRAIN_MPRage_SAGIT.nii.gz    86.094208
    
    sub-02
                                                                       base_name         error
        sub-02_session-07_space-MNI_101_sub-02_deidentified_SAG_T1_MPRAGE.nii.gz     14.080871
        sub-02_session-07_space-MNI_100_sub-02_deidentified_SAG_T1_MPRAGE.nii.gz     14.080871
        sub-02_session-06_space-MNI_002_sub-02_deidentified_SAG_T1_MPRAGE.nii.gz    270.513275
        sub-02_session-06_space-MNI_101_sub-02_deidentified_SAG_T1_MPRAGE.nii.gz    270.878662
        sub-02_session-06_space-MNI_100_sub-02_deidentified_SAG_T1_MPRAGE.nii.gz    274.747833
    
    sub-03
                                                                                   base_name       error
              sub-03_session-06_space-MNI_101_sub-03b_deidentified_SAG_T1_MPRAGE_0919.nii.gz    1.919110
              sub-03_session-10_space-MNI_106_sub-03b_deidentified_SAG_T1_MPRAGE_POST.nii.gz    1.919129
                   sub-03_session-09_space-MNI_100_sub-03b_deidentified_SAG_T1_MPRAGE.nii.gz    1.919129
         sub-03_session-06_space-MNI_102_sub-03b_deidentified_SAG_T1_MPRAGE_Post_0919.nii.gz    1.919129
                   sub-03_session-09_space-MNI_101_sub-03b_deidentified_SAG_T1_MPRAGE.nii.gz    1.919129
    
    sub-04
                                                                         base_name         error
            sub-04_session-01_space-MNI_100_sub-04a_deidentified_MPRAGE_SAG.nii.gz    430.925140
            sub-04_session-04_space-MNI_100_sub-04a_deidentified_MPRAGE_SAG.nii.gz    443.140533
            sub-04_session-02_space-MNI_100_sub-04a_deidentified_MPRAGE_SAG.nii.gz    467.221069
         sub-04_session-01_space-MNI_100_sub-04b_deidentified_SAG_T1_MPRAGE.nii.gz    471.476257
         sub-04_session-01_space-MNI_004_sub-04b_deidentified_SAG_T1_MPRAGE.nii.gz    474.170837
    
    sub-05
                                                                             base_name       error
              sub-05_session-07_space-MNI_101_sub-05_deidentified_SAG_T1_MPRAGE.nii.gz    2.919129
              sub-05_session-07_space-MNI_100_sub-05_deidentified_SAG_T1_MPRAGE.nii.gz    2.919129
         sub-05_session-07_space-MNI_103_sub-05_deidentified_SAG_T1_MPRAGE_POST.nii.gz    2.919129
         sub-05_session-07_space-MNI_102_sub-05_deidentified_SAG_T1_MPRAGE_POST.nii.gz    2.919129
         sub-05_session-08_space-MNI_103_sub-05_deidentified_SAG_T1_MPRAGE_Post.nii.gz    2.919129
    
    sub-06
                                                                              base_name       error
              sub-06_session-08_space-MNI_100_sub-06b_deidentified_SAG_T1_MPRAGE.nii.gz    3.919129
              sub-06_session-09_space-MNI_100_sub-06b_deidentified_SAG_T1_MPRAGE.nii.gz    3.919129
              sub-06_session-09_space-MNI_101_sub-06b_deidentified_SAG_T1_MPRAGE.nii.gz    3.919129
         sub-06_session-09_space-MNI_102_sub-06b_deidentified_SAG_T1_MPRAGE_Post.nii.gz    3.919129
         sub-06_session-09_space-MNI_103_sub-06b_deidentified_SAG_T1_MPRAGE_Post.nii.gz    3.919129
    
    sub-07
                                                                             base_name       error
              sub-07_session-03_space-MNI_101_sub-07_deidentified_SAG_T1_MPRAGE.nii.gz    8.080871
              sub-07_session-06_space-MNI_100_sub-07_deidentified_SAG_T1_MPRAGE.nii.gz    9.080871
              sub-07_session-06_space-MNI_101_sub-07_deidentified_SAG_T1_MPRAGE.nii.gz    9.080871
         sub-07_session-06_space-MNI_102_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    9.080871
         sub-07_session-06_space-MNI_103_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    9.080871

### Worst predictions by subject

    sub-01
                                                                                   base_name        error
       sub-01_session-01_space-MNI_002_sub-01_deidentified_16_PEDI_BRAIN_MPRage_SAGIT.nii.gz    86.094208
       sub-01_session-00_space-MNI_002_sub-01_deidentified_18_PEDI_BRAIN_MPRage_SAGIT.nii.gz    60.329544
    
    sub-02
                                                                            base_name         error
             sub-02_session-00_space-MNI_103_sub-02_deidentified_MPRAGE_SAG_GD.nii.gz    830.719238
             sub-02_session-04_space-MNI_022_sub-02_deidentified_MPRAGE_SAG_GD.nii.gz    829.751465
        sub-02_session-05_space-MNI_018_sub-02_deidentified_SAG_T1_MPRAGE_post.nii.gz    825.035522
        sub-02_session-05_space-MNI_103_sub-02_deidentified_SAG_T1_MPRAGE_post.nii.gz    766.786133
             sub-02_session-04_space-MNI_102_sub-02_deidentified_MPRAGE_SAG_GD.nii.gz    703.474365
    
    sub-03
                                                                                  base_name         error
         sub-03_session-01_space-MNI_022_sub-03a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    741.636536
         sub-03_session-03_space-MNI_023_sub-03a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    728.924683
             sub-03_session-05_space-MNI_103_sub-03b_deidentified_SAG_T1_MPRAGE_post.nii.gz    718.844360
         sub-03_session-01_space-MNI_102_sub-03a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    714.656616
             sub-03_session-05_space-MNI_017_sub-03b_deidentified_SAG_T1_MPRAGE_post.nii.gz    711.849609
    
    sub-04
                                                                                  base_name         error
                  sub-04_session-03_space-MNI_024_sub-04a_deidentified_MPRAGE_SAG_GD.nii.gz    848.416992
                  sub-04_session-03_space-MNI_103_sub-04a_deidentified_MPRAGE_SAG_GD.nii.gz    778.080811
         sub-04_session-05_space-MNI_022_sub-04a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    750.710815
         sub-04_session-00_space-MNI_103_sub-04a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    743.387085
                  sub-04_session-04_space-MNI_024_sub-04a_deidentified_MPRAGE_SAG_GD.nii.gz    699.599426
    
    sub-05
                                                                             base_name         error
         sub-05_session-10_space-MNI_024_sub-05_deidentified_SAG_T1_MPRAGE_Post.nii.gz    778.190552
         sub-05_session-10_space-MNI_103_sub-05_deidentified_SAG_T1_MPRAGE_Post.nii.gz    769.078674
         sub-05_session-10_space-MNI_102_sub-05_deidentified_SAG_T1_MPRAGE_Post.nii.gz    736.371521
         sub-05_session-03_space-MNI_103_sub-05_deidentified_SAG_T1_MPRAGE_POST.nii.gz    709.248779
              sub-05_session-00_space-MNI_103_sub-05_deidentified_MPRAGE_SAG_GD.nii.gz    707.888184
    
    sub-06
                                                                                  base_name         error
         sub-06_session-06_space-MNI_103_sub-06a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    682.227783
                  sub-06_session-01_space-MNI_104_sub-06a_deidentified_MPRAGE_SAG_GD.nii.gz    676.752197
             sub-06_session-05_space-MNI_103_sub-06a_deidentified_SAG_T1_MPRAGE_POST.nii.gz    674.536133
             sub-06_session-05_space-MNI_025_sub-06a_deidentified_SAG_T1_MPRAGE_POST.nii.gz    673.441528
         sub-06_session-06_space-MNI_023_sub-06a_deidentified_T1_FLASH_MPRAGE_SAG_+C.nii.gz    662.486694
    
    sub-07
                                                                              base_name         error
          sub-07_session-01_space-MNI_106_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    692.586792
          sub-07_session-01_space-MNI_104_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    686.953491
          sub-07_session-01_space-MNI_019_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    686.619751
         sub-07_session-06_space-MNI_103_sub-07_deidentified_SAG_T1_MPRAGE_POST_.nii.gz    626.694519
          sub-07_session-06_space-MNI_019_sub-07_deidentified_SAG_T1_MPRAGE_POST.nii.gz    624.083496
