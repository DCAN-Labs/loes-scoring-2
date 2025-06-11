# Loes score model training and inference

## Data

### Preventing data leakage

You must be certain that all of the sessions for a given subject are in either the *training* and *validation* session.  No subject can have one (or more) *sessions* in the *training* set and one or more *sessions* in the *test* set.  **This would result in data leakage**.

Input data is stored in a CSV file.  Here are the first few lines of such a file

      anonymized_subject_id anonymized_session_id           scan  loes-score  Gd-enhanced
    0            subject-00            session-00  mprage.nii.gz         1.0            0
    1            subject-00            session-01  mprage.nii.gz         0.0            0
    2            subject-00            session-02  mprage.nii.gz         1.0            0
    3            subject-00            session-03  mprage.nii.gz         1.0            0
    4            subject-01            session-00  mprage.nii.gz         0.0            0

You also have to give a folder input to the program, such as

    --folder /home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/

The NIFTI files should have names such as *subject-02_session-02_space-MNI_brain_mprage_RAVEL.nii.gz*.  It is easy to see how the full paths of files can be created from the `folder` input and the subject and session names.

### Creating model

The program you need to run to create a model is

    src/dcan/training/regression.py

Here are the command-line options:

    usage: training.py [-h] [--tb-prefix TB_PREFIX] [--csv-input-file CSV_INPUT_FILE] [--num-workers NUM_WORKERS] [--batch-size BATCH_SIZE] [--epochs EPOCHS]
                    [--file-path-column-index FILE_PATH_COLUMN_INDEX] [--loes-score-column-index LOES_SCORE_COLUMN_INDEX]
                    [--model-save-location MODEL_SAVE_LOCATION] [--plot-location PLOT_LOCATION] [--optimizer OPTIMIZER] [--lr LR] [--gd GD]
                    [--use-train-validation-cols] [-k K] [--folder FOLDER] [--csv-output-file CSV_OUTPUT_FILE] [--use-weighted-loss]
                    [--scheduler {plateau,step,cosine,onecycle}] [--model {resnet,alexnet}]
                    [comment]

    positional arguments:
    comment               Comment for Tensorboard run

    options:
    -h, --help            show this help message and exit
    --tb-prefix TB_PREFIX
                            Tensorboard data prefix.
    --csv-input-file CSV_INPUT_FILE
                            CSV data file.
    --num-workers NUM_WORKERS
                            Number of worker processes
    --batch-size BATCH_SIZE
                            Batch size for training
    --epochs EPOCHS       Number of epochs to train
    --file-path-column-index FILE_PATH_COLUMN_INDEX
                            Index of the file path in CSV file
    --loes-score-column-index LOES_SCORE_COLUMN_INDEX
                            Index of the Loes score in CSV file
    --model-save-location MODEL_SAVE_LOCATION
    --plot-location PLOT_LOCATION
                            Location to save plot
    --optimizer OPTIMIZER
                            Optimizer type.
    --lr LR               Learning rate
    --gd GD               Use Gd-enhanced scans.
    --use-train-validation-cols
    -k K                  Index for 5-fold validation
    --folder FOLDER       Folder where MRIs are stored
    --csv-output-file CSV_OUTPUT_FILE
                            CSV output file.
    --use-weighted-loss
    --scheduler {plateau,step,cosine,onecycle}
                            Learning rate scheduler
    --model {resnet,alexnet}
                            Model architecture

Here is a sample VS Code run target:

        {
            "name": "training",
            "program": "src/dcan/regression/training.py",
            "type": "debugpy",
            "request": "launch",
            "console": "integratedTerminal",
            "args": ["--csv-input-file", "data/anon_train_scans_and_loes_training_test_non_gd.csv",
                     "--folder",  "/home/feczk001/shared/projects/S1067_Loes/data/Fairview-ag/05-training_ready/",
                     "--DEBUG"]
        }

The output will consist of the following

#### CSV output

A CSV output file will be the same as the input CSV file except for a *predicted_loes_score* for the validation rows.  Here are the first few lines of a sample output:

    anonymized_subject_id,anonymized_session_id,scan,loes-score,Gd-enhanced,predicted_loes_score
    subject-00,session-00,mprage.nii.gz,1.0,0,
    subject-00,session-01,mprage.nii.gz,0.0,0,
    subject-10,session-00,mprage.nii.gz,6.5,0,11.458658218383789
    subject-10,session-06,mprage.nii.gz,15.0,0,11.725173950195312

#### Plot output

Also a plot will be produced showing the actual versus predicted Loes scores.

![Loes score prediction](doc/img/sample_output/sample_loes_score_predictions.png)

#### Statistics

The following statistics will also be in the log file after training completes:

* correlation_coefficient
* Pearson correlation p-value
* Spearman correlation p-value
