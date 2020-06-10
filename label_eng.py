import pandas as pd
import numpy as np
import statsmodels.api as sm
# from mlfinlab.labeling import trend_scanning_labels

pd.set_option('display.max_rows', None)

def tValOLS(y):
    x = np.ones((len(y), 2))
    x[:,1] = np.arange(len(y))
    mols = sm.OLS(y,x).fit()
    return mols.tvalues[1]


def trend_labelling(ser, span, min_obs=3):
    assert span > min_obs
    name = ser.name
    ser = ser.to_frame()
    ser['t_val'] = np.nan
    ser['ret'] = np.nan
    for curr_t in ser.iloc[:-span].index:
        lookForward = ser[name].loc[curr_t:].iloc[:span]
        max_abs_t_val = -np.inf
        max_t_val = None
        max_t_index = None
        for fwd in range(min_obs, len(lookForward)):
            y = lookForward.iloc[:fwd]
            tVal = tValOLS(y)
            if abs(tVal) > max_abs_t_val:
                max_abs_t_val = abs(tVal)
                max_t_val = tVal
                max_t_index = y.index[-1]

        ser.loc[curr_t, "t_val"] = max_t_val
        ser.loc[curr_t, "st"] = max_t_index
        # print(f"curr_t: {curr_t}, max_abs_t_val: {max_abs_t_val}, max_t_index: {max_t_index}")
        # print(ser[name].loc[max_t_index])
        # print(ser[name].loc[curr_t])
        ser.loc[curr_t, "ret"] = ser[name].loc[max_t_index]/ ser[name].loc[curr_t] -1
    return ser


df = pd.DataFrame({"close": 100*np.sin(range(0,90))})

print(trend_labelling(df["close"], span=5))

