import glob
import logging

import pandas as pd

from dcan.metrics import get_standardized_rmse

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

folder = '/home/miran045/reine097/projects/loes-scoring-2/doc/models/five_fold_validation/'
path = rf'{folder}*.csv'
files = glob.glob(path)
dfs = map(lambda f: pd.read_csv(f), files)
result_df = pd.concat(dfs, ignore_index=True, axis=0)
rslt_df = result_df[result_df['validation'] == 1]
rslt_df = rslt_df.sort_values(['subject', 'session'])
rslt_df.to_csv(f'{folder}output_all.csv', index=False)
actuals = list(rslt_df['loes-score'])
predictions = list(rslt_df['prediction'])
standardized_rmse = get_standardized_rmse(actuals, predictions)
log.info(f'standardized_rmse: {standardized_rmse}')
