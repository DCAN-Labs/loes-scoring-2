import math
import statistics

from sklearn.metrics import mean_squared_error


def get_standardized_rmse(target, preds):
    mse = mean_squared_error(target, preds)
    rmse = math.sqrt(mse)
    sigma = statistics.stdev(preds)

    return rmse / sigma
