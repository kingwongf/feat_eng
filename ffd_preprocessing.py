import ray
import statsmodels.api as sm
from FFD import fracDiff_FFD
import numpy as np
import yaml
import time
from functools import reduce
import pandas as pd
with open("configs.yml") as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

with open("opt_d_ffd.yml") as file:
    opt_d = yaml.load(file, Loader=yaml.FullLoader)

log_prices = pd.read_pickle(configs["log_adj_close"])

print(len(log_prices.columns))
start_time = time.time()


li_dfs = [fracDiff_FFD(log_prices[[list(opt_d.keys())[i]]],list(opt_d.values())[i],thres=1e-3) for i in range(0,10)]
print(f"time elapse: {(time.time() - start_time) / 60}mins")
li_dfs = reduce(
            lambda X, x: pd.merge(X, x, how='outer', left_index=True, right_index=True), li_dfs)
print(li_dfs)