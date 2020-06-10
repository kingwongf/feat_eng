import ray
import statsmodels.api as sm
from FFD import getWeights_FFD
import numpy as np
import yaml
import time
from functools import reduce
import pandas as pd
with open("configs.yml") as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

with open("opt_d_ffd.yml") as file:
    opt_d = yaml.load(file, Loader=yaml.FullLoader)



@ray.remote
def fracDiff_FFD(series,d,thres=1e-3):
    # Constant width window (new solution)
    w = getWeights_FFD(d,thres)
    width = len(w)-1
    df={}
    for name in series.columns:
        seriesF, df_=series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width], seriesF.index[iloc1]
            test_val = series.loc[loc1,name] # must resample if duplicate index
            if isinstance(test_val, (pd.Series, pd.DataFrame)):
                test_val = test_val.resample('1m').mean()
            if not np.isfinite(test_val).any(): continue # exclude NAs
            #print(f'd: {d}, iloc1:{iloc1} shapes: w:{w.T.shape}, series: {seriesF.loc[loc0:loc1].notnull().shape}')
            try:
                df_.loc[loc1]=np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]
            except:
                continue
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df



ray.init()

df_name = "log_adj_volume"
log_adj_close = pd.read_pickle(configs[df_name])
li_dfs = [fracDiff_FFD.remote(log_adj_close[[list(opt_d.keys())[i]]],list(opt_d.values())[i],thres=1e-3)
          for i in range(len(log_adj_close.columns))
          ]

ray_li_dfs = ray.get(li_dfs)


df = reduce(lambda X, x: pd.merge(X, x, how='outer', left_index=True, right_index=True), ray_li_dfs)

df.to_pickle(f"data/ffd_{df_name}.pkl")
