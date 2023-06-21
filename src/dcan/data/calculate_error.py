import sys
import pandas as pd


def calculate_error(csv_file_in, csv_file_out):
    df_in = pd.read_csv(csv_file_in)
    df_out = df_in.copy()
    df_out['error'] = df_out.apply(lambda df_out: abs(df_out['prediction'] - df_out['loes_score']), axis=1)
    df_out.to_csv(csv_file_out)
    error_mean = df_out["error"].mean()
    print(f'error mean: {error_mean}')

if __name__ == "__main__":
    calculate_error(sys.argv[1], sys.argv[2])
