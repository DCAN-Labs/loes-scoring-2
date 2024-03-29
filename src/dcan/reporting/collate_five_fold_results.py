import glob

import scipy

from dcan.plot.create_scatterplot import create_scatterplot
from util.logconf import logging

import pandas as pd

from dcan.metrics import get_standardized_rmse

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

folder = '/home/miran045/reine097/projects/loes-scoring-2/doc/models/model09/five_fold_validation'
path = rf'{folder}*.csv'
files = glob.glob(path)
dfs = map(lambda f: pd.read_csv(f), files)
result_df = pd.concat(dfs, ignore_index=True, axis=0)
rslt_df = result_df[result_df['validation'] == 1]
rslt_df = rslt_df.sort_values(['subject', 'session'])
rslt_df.to_csv(f'{folder}five_fold_cross_validation.csv', index=False)
actuals = list(rslt_df['loes-score'])
predictions = list(rslt_df['prediction'])
standardized_rmse = get_standardized_rmse(actuals, predictions)
log.info(f'standardized_rmse: {standardized_rmse}')

# noinspection PyUnresolvedReferences
result = scipy.stats.linregress(actuals, predictions)

# noinspection PyUnresolvedReferences
log.info(f"correlation:    {result.rvalue}")
# noinspection PyUnresolvedReferences
log.info(f"p-value:        {result.pvalue}")
# noinspection PyUnresolvedReferences
log.info(f"standard error: {result.stderr}")

create_scatterplot(
    rslt_df[['loes-score', 'prediction', 'subject', 'session']], f'{folder}five_fold_cross_validation.png')
