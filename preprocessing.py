import pandas as pd
import numpy as np
import yaml
import swifter
import time

class preProcess:
    def __init__(self, configs_loc):
        with open(configs_loc) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)
        us_eq = pd.read_csv(self.configs['us_eq'], names=["ticker", "Date", "open", "high","low","close","volume",
                                                           "dividend", "split", "adj_open", "adj_high", "adj_low",
                                                           "adj_close", "adj_volume"],
                            index_col="Date")

        self.sp500_sector = pd.read_csv(self.configs['sp500']).rename({'Symbol':'ticker', 'Sector':'sector'}, axis=1)

        li_sp500_ticker = self.sp500_sector['ticker'].tolist()

        us_eq = us_eq[us_eq['ticker'].isin(li_sp500_ticker)]


        # print(us_eq.head(10))

        us_eq.index = pd.to_datetime(us_eq.index)
        # us_eq = us_eq['':]

        self.adj_close = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_close"]).sort_index().ffill().bfill().dropna(axis=0)
        self.adj_volume = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_volume"]).sort_index().ffill().bfill().dropna(axis=0)
        self.dy = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["dividend"]).sort_index().ffill().bfill().dropna(axis=0)
        # self.us_eq = us_eq

        # print(self.adj_close)
        np.log(self.adj_close).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_close"])



    def featGen(self):

        start_time = time.time()

        feats_df = self.adj_close.stack().reset_index().set_index('Date',drop=True)
        feats_df['log_adj_price'] = np.log(feats_df['adj_close'])

        feats_df['ret1d'] = self.adj_close.pct_change(1).values.flatten()
        feats_df['retvol12m'] = np.sqrt(252)*self.adj_close.pct_change(1).rolling(252).std().values.flatten()
        feats_df['retvol1m'] = np.sqrt(21)*self.adj_close.pct_change(1).rolling(21).std().values.flatten()
        feats_df['mom1m'] = self.adj_close.pct_change(21).values.flatten()
        feats_df['mom6m'] = self.adj_close.pct_change(126).values.flatten()
        feats_df['mom12m'] = self.adj_close.pct_change(252).values.flatten()
        feats_df['mom36m'] = self.adj_close.pct_change(756).values.flatten()
        feats_df['mom36m'] = self.adj_close.pct_change(756).values.flatten()
        feats_df['maxret1d'] = self.adj_close.pct_change(1).rolling(21).max().values.flatten()
        feats_df['maxret1m'] = self.adj_close.pct_change(21).rolling(21).max().values.flatten()

        ## industry momentum

        ## price weighted industry momentum in abscent of market cap weighting
        ## w_stock_i = price_stock_i / (sum price_stock)
        feats_df = feats_df.reset_index().merge(self.sp500_sector[['ticker', 'sector']],
                                                how='left', left_on='ticker', right_on='ticker').set_index("Date")

        feats_df.merge(feats_df.groupby(['Date', 'sector']).apply(lambda x: np.average(x['mom1m'],
                                                                                       weights=x['adj_close']
                                                                                       )
                                                                  ).rename('indmom1m', axis=1),
                       left_on=['Date', 'sector'], right_on=['Date', 'sector'])

        feats_df.merge(feats_df.groupby(['Date', 'sector']).apply(lambda x: np.average(x['mom6m'],
                                                                                       weights=x['adj_close']
                                                                                       )
                                                                  ).rename('indmom6m', axis=1),
                       left_on=['Date', 'sector'], right_on=['Date', 'sector'])


        feats_df['dollar_volume'] = (self.adj_volume.droplevel(level=0, axis=1)*self.adj_close.droplevel(level=0, axis=1)).values.flatten()
        feats_df['turnover'] = self.adj_volume.values.flatten()
        feats_df['turnover_vol1m'] = self.adj_volume.rolling(21).std().values.flatten()

        feats_df['dy'] = self.dy.values.flatten()

        feats_df.drop('adj_close', axis=1, inplace=True)

        feats_df.to_pickle(self.configs['feats_df'])

        print(f"time elapse: {(time.time() - start_time)/60}mins")
        print(feats_df)






# pp = preProcess("configs.yml")
pp = preProcess("configs.yml").featGen()