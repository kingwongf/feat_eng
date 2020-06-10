import pandas as pd
import numpy as np
import yaml
import time
# from label_eng import trend_labelling
import swifter

class preProcess:
    def __init__(self, configs_loc):
        with open(configs_loc) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)



    def adj_df(self):

        us_eq = pd.read_csv(self.configs['us_eq'], names=["ticker", "Date", "open", "high", "low", "close", "volume",
                                                          "dividend", "split", "adj_open", "adj_high", "adj_low",
                                                          "adj_close", "adj_volume"],
                            index_col="Date")

        self.sp500_sector = pd.read_csv(self.configs['sp500']).rename({'Symbol': 'ticker', 'Sector': 'sector'}, axis=1)

        li_sp500_ticker = self.sp500_sector['ticker'].tolist()

        us_eq = us_eq[us_eq['ticker'].isin(li_sp500_ticker)]

        # print(us_eq.head(10))

        us_eq.index = pd.to_datetime(us_eq.index)

        no_bfill_adj_close = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"],
                                            values=["adj_close"]).sort_index().ffill()\
            .stack().reset_index().set_index('Date',drop=True).rename({'adj_close':'bfill'}, axis=1)

        no_bfill_adj_close.to_pickle(self.configs["no_bfill_close"])


        ## TODO maybe try without bfill because we would be labelling 0 when trend is flatline
        self.adj_close = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_close"]).sort_index().ffill().bfill().dropna(axis=0)
        self.adj_high = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_high"]).sort_index().ffill().bfill().dropna(axis=0)
        self.adj_open = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_open"]).sort_index().ffill().bfill().dropna(axis=0)
        self.adj_low = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_low"]).sort_index().ffill().bfill().dropna(axis=0)
        self.adj_volume = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["adj_volume"]).sort_index().ffill().bfill().dropna(axis=0)
        self.dy = pd.pivot_table(us_eq, index=us_eq.index, columns=["ticker"], values=["dividend"]).sort_index().ffill().bfill().dropna(axis=0)
        self.us_eq = us_eq



        # np.log(self.adj_close).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_close"])
        # np.log(self.adj_high).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_high"])
        # np.log(self.adj_open).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_open"])
        # np.log(self.adj_low).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_low"])
        # np.log(self.adj_volume).droplevel(level=0, axis=1).to_pickle(self.configs["log_adj_volume"])



    def featGen(self):

        self.adj_df()

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

    def combined_ffd_log_with_feats(self):
        ffd_df = pd.DataFrame()
        ffd_loc = self.configs["ffd"]
        log_loc = self.configs["log_prices"]

        for price in ['open', 'close', 'high', 'low', 'volume']:
            if len(ffd_df.index) <=0:
                ffd_df = pd.read_pickle(f"{ffd_loc}{price}.pkl")
                ffd_df = ffd_df.stack().reset_index()\
                    .rename({'level_0':'Date', 'level_1':'ticker', 0:f"ffd_adj_{price}"}, axis=1)\
                    .set_index('Date', drop=True)
            else:
                ffd_part = pd.read_pickle(f"{ffd_loc}{price}.pkl").stack().reset_index() \
                    .rename({'level_0': 'Date', 'level_1': 'ticker', 0: f"ffd_adj_{price}"}, axis=1)\
                    .set_index('Date', drop=True)\


                # print(ffd_df)
                # print(ffd_part)
                ffd_df = ffd_df.merge(ffd_part, how='left', left_on=['Date', 'ticker'], right_on=['Date', 'ticker'])

        feats_df = pd.read_pickle(self.configs["feats_df"])
        feats_df = feats_df.merge(ffd_df, how='left', left_on=['Date', 'ticker'], right_on=['Date', 'ticker'])


        for name, loc in log_loc.items():
            ## we already have log adj close on feats_df
            if 'close' not in name:
                print(name)
                log_part = pd.read_pickle(loc).stack().reset_index()\
                    .set_index('Date').rename({0:name}, axis=1)


                feats_df = feats_df.merge(log_part, how='left',
                                          left_on=['Date', 'ticker'],
                                          right_on=['Date', 'ticker']
                                          )

        # print(feats_df.columns)
        feats_df.to_pickle(self.configs["ffd_feats"])

    def labelling(self):

        adj_close = np.e**pd.read_pickle(self.configs["log_prices"]["log_adj_close"])


        ## TODO WOn't work because you have 3 columns returning for each ticker column
        # label_trend_20d = trend_labelling(adj_close['AAL'], 20)
        label_trend_20d = adj_close.swifter.apply(trend_labelling, args=(20,))

        print(label_trend_20d)
        # print(label_trend_20d)













# pp = preProcess("configs.yml")
pp = preProcess("configs.yml").adj_df()
# pp = preProcess("configs.yml").featGen()

