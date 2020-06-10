import ray
import statsmodels.api as sm
from FFD import fracDiff_FFD
import numpy as np
import yaml
import pandas as pd
from functools import reduce
pd.set_option('display.max_columns', None)  # or 1000
with open("configs.yml") as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)


ret_span = 20
vol_span = 20

adj_close = np.e**pd.read_pickle(configs["log_prices"]["log_adj_close"])

# print(adj_close[:30])
vol = adj_close.pct_change(ret_span).rolling(vol_span).std()
ret = adj_close.pct_change(ret_span)

risk_adj_ret = (ret/vol).stack().shift(-ret_span).reset_index().set_index('Date', drop=True).rename({0:"fwd_ret"}, axis=1)
raw_ret = ret.shift(-ret_span).stack().reset_index().set_index('Date', drop=True).rename({0:"fwd_ret"}, axis=1)

risk_adj_ret.to_pickle(f"{configs['risk_adj']}{ret_span}d.pkl")
raw_ret.to_pickle(f"{configs['raw_ret']}{ret_span}d.pkl")
print(risk_adj_ret)
print(raw_ret)
