import csv

file1_name = '../../../data/ALD-google_sheet-Jul272022-Loes_scores-Igor_updated.csv'
def get_data(file_name, rows_to_skip, subject_index, session_index, score_index):
    with open(file_name) as f:
        reader = csv.reader(f)
        data1 = dict()
        rows_skipped = 0
        for row in reader:
            if rows_skipped < rows_to_skip:
                rows_skipped += 1

                continue
            subject = row[subject_index]
            session = row[session_index]
            score_str = row[score_index]
            if score_str != '':
                score = float(score_str)
                if subject not in data1:
                    data1[subject] = dict()
                data1[subject][session] = score

    return data1


file2_name = '../../../data/9_7 MRI sessions Igor Loes score updated.csv'
data2 = get_data(file2_name, 2, 0, 1, 35)
data1 = get_data(file1_name, 1, 0, 1, 2)

for subject in data2:
    if subject in data1:
        for session in data2[subject]:
            if session in data1[subject]:
                diff = data2[subject][session] - data1[subject][session]
                abs_diff = abs(diff)
                if abs_diff > 0.1:
                    print(f'Discrepancy on {subject}/{session}: {abs_diff}')
