import ray
import statsmodels.api as sm
from FFD import fracDiff_FFD
import numpy as np
import yaml
import pandas as pd

with open("configs.yml") as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)


@ray.remote
def test_get_optimal_ffd(x, ds, t=1e-3):
    cost_funcs =[]
    for i,d in enumerate(ds):
        try:
            dfx = fracDiff_FFD(x.to_frame(), d, thres=t)
            dfx = sm.tsa.stattools.adfuller(dfx[dfx.columns[0]], maxlag=1, regression='c', autolag=None)
            cost = dfx[0] - dfx[4]['5%']
            cost_funcs.append(cost)
            if cost_funcs[i-1] < 0 and cost_funcs[i] < cost_funcs[i-1]:
                opt_d = ds[i-1]
                return float(opt_d)
        except Exception as e:
            print(f'{d} error: {e}')


ray.init()

log_adj_close = pd.read_pickle(configs["log_adj_close"])
ds = opt_ds = [test_get_optimal_ffd.remote(log_adj_close[col], np.arange(0.01,1.01, 0.01))
                    for col in log_adj_close.columns.tolist()]

print(ray.get(ds))

