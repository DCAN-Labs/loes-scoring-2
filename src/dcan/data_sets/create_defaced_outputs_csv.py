import pandas as pd
from os import listdir
from os.path import isfile, join

# Define paths
my_path = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/raw/'
input_csv = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/loes_scores_no_gd_model.csv'

# List all files in the directory
all_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]

# Load input CSV
input_df = pd.read_csv(input_csv)

loes_scores = []
subjects = []
sessions = []

for f in all_files:
    # Filter rows where 'file-path' contains the file name
    location = input_df[input_df['file-path'].str.contains(f, na=False)]
    
    if not location.empty:
        # Get the first match
        first_match = location.iloc[0]
        loes_score = first_match['loes_score']
        loes_scores.append(loes_score)

        # Extract subject and session information from file name
        subject = f[:6]  # TODO Adjust this according to your file naming convention
        session = f[7:17]  # TODO Adjust this based on actual filename patterns
        
        subjects.append(subject)
        sessions.append(session)

# Prepare DataFrame for output
data = {
    "subject": subjects,
    "session": sessions,
    "file_path": all_files,
    "loes_score": loes_scores
}

output_df = pd.DataFrame(data)

# Save the DataFrame to CSV
output_csv = '/users/9/reine097/loes_scoring/s1067-loes-score/Nascene_data/defacing/defaced_outputs/loes_scores.csv'
output_df.to_csv(output_csv, index=False)

print(f"CSV saved to {output_csv}")
