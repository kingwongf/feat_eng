import numpy as np
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

def exponential_momentum(ts, min_nobs, window):
    '''
    Andrew Clenow's Method
    1. ln(ts) = m*ln(t) + c
    2. annualised momentum = ((e^(m))^(252) -1 ) * 100
    :return:
        annualised momentum score
    '''
    exog = sm.add_constant(np.arange(0, len(ts)))
    rolling_param = RollingOLS(np.log(ts), exog, min_nobs=min_nobs, window=window).fit()
    return (np.power(np.exp(rolling_param.params['x1']), 252)-1)*100 * rolling_param.rsquared

def mom(df, n=1):
    return np.diff(df.values, n=n, axis=0, prepend=np.nan) / df.shift(n).values

def chmom(df, n=1, m=1):
    return np.diff(mom(df,m), n=n, axis=1, prepend=np.nan)