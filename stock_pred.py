import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import classification_report, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
import yaml
from tabulate import tabulate

pd.set_option('display.max_columns', None)  # or 1000
# pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


class process:
    def __init__(self, configs_loc):
        with open(configs_loc) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)

            '''
            1. form df by combine label df (i. trend labelling ii. volatility adjusted returns iii. raw returns
                                            and selected features df (i. log prices ii. ffd log prices iii. prices plus technicals for all) 
            1.1 would be 9 *9 combinations, ray?
            1.2 drop other label columns to prevent leakage                                 
                                            
            2. drop na row if label is nan, train test split and standard scalar, prob should only one mean and one std, not dynamically changing
            3. fit XGBClassifier or XGBRegressor, then predict test sets, calculate loss
            4. train again with 60% from start, save model for backtesting last 40 % test period
            '''

    def preprocessing(self, ret_span, label_type, feats_type):

        feats_df = pd.read_pickle(self.configs['ffd_feats'])
        feats_df.sort_index(inplace=True)
        # print(feats_df.describe())

        feats_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feats_df.ffd_adj_volume = feats_df.groupby(['ticker'])['ffd_adj_volume'].ffill()
        feats_df.log_adj_volume = feats_df.groupby(['ticker'])['log_adj_volume'].ffill()

        log_cols = [col for col in feats_df.columns.tolist() if 'log_' in col]
        ffd_cols = [col for col in feats_df.columns.tolist() if 'ffd_' in col]
        price_cols = [col.lstrip('log_') for col in log_cols]

        feats_df[price_cols] = np.e**feats_df[log_cols]
        feats_cols = [col for col in feats_df.columns.tolist() if col not in log_cols+ ffd_cols+price_cols]

        d_feats_type = {"log": feats_cols+log_cols,
                        "ffd": feats_cols+ffd_cols,
                        "raw_price": feats_cols+price_cols}

        cols2use = feats_df[d_feats_type[feats_type]]

        # print(f"log cols: {log_cols}")
        # print(f"ffd cols: {ffd_cols}")
        # print(f"price cols: {price_cols}")
        # print(f"feats cols: {feats_cols}")


        label_df = pd.read_pickle(f"{self.configs[label_type]}{ret_span}d.pkl").replace([np.inf, -np.inf], np.nan).rename({"t_val":"fwd_ret"}, axis=1)

        if label_type=='raw_ret':
            label_df = np.tanh(label_df)


        # print(label_df.head(10))


        ## Realised we should not use ticker which does not exsist in the beginning, should revere backfill
        ## should get adj_close without backfill, join on Xy df, dropna for adj_close are nan

        no_bfill_adj_close = pd.read_pickle(self.configs["no_bfill_close"])
        # print(no_bfill_adj_close.head(10))
        Xy = reduce(lambda X, x: X.sort_index().merge(x.sort_index(), how='left',
                    left_on=["Date", "ticker"], right_on=["Date", "ticker"]),
                    [no_bfill_adj_close, cols2use,
                     label_df[["fwd_ret", "ticker"]]])

        ## drop rows with blank backfill, no label and longest features
        Xy = Xy.dropna(axis=0, how='any', subset=['bfill','fwd_ret','mom36m']).drop(['bfill'], axis=1)

        return Xy
    def train_test_split(self,df, split_date):
        return df[:split_date], df[split_date:]

    def forward_chaining(self, df, n=1):
        ## Retrain model every nth years

        years = df.index.year.unique().tolist()
        ## Start with 10 years of data to train and 1 year to test
        years = years[10+n:]

        dfs =[]
        for year in years:
            train, test = df[:str(year-n)], df[str(year)]
            # print(f"train years: {train.index.min()} to {train.index.max()}, test years: {test.index.min()} to {test.index.max()}")
            dfs.append([(train.index.min().year, train.index.max().year, test.index.min().year, test.index.max().year), train, test])
        return dfs

    def splitXY(self,df, target):
        y = df.pop(target)
        return df, y

    def reg2clf(self, Xy, target_col):
        Xy.loc[Xy[target_col] > 0, target_col] = 1
        Xy.loc[Xy[target_col] < 0, target_col] = -1
        Xy[target_col] = Xy[target_col].astype('category')
        return Xy


    def ml(self, train_Xy, test_Xy, target_col, feats2remove, reg_clf='reg'):
        # print(train_Xy)
        train_X, train_y = self.splitXY(train_Xy, target_col)
        test_X, test_y = self.splitXY(test_Xy, target_col)

        categorical_features = train_X.columns[(train_X.dtypes.values != np.dtype('float64'))].tolist()
        numeric_features = train_X.columns[(train_X.dtypes.values == np.dtype('float64'))].tolist()

        categorical_features.remove(feats2remove)

        numeric_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='median')), ('scalar', StandardScaler())])

        categorical_transfomer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                                 ('onehot_sparse', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transfomer, categorical_features)
            ]
        )

        clf_params = {'num_class': 3, 'objective': 'multi:softprob'}
        reg_params = {'max_depth': 8, 'learning_rate': 0.01, 'n_estimators':300, 'objective':'reg:squarederror'}

        model = XGBRegressor(**reg_params) if reg_clf=='reg' else XGBClassifier(**clf_params)

        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
        print("Preparing to train...")
        # print(train_X.shape)
        clf.fit(train_X, train_y)

        # pickle.dumps(clf, open('clf.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        pred_y = clf.predict(test_X)
        # np.savetxt(file, pred_y)

        if reg_clf=='clf':
            print(classification_report(test_y, pred_y))
        elif reg_clf=='reg':
            r2 = r2_score(test_y, pred_y)
            # print(r2)

        onehot_columns = clf.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
            'onehot_sparse'].get_feature_names(input_features=categorical_features)

        feature_importance = pd.Series(data=clf.named_steps['model'].feature_importances_,
                                       index=np.append(numeric_features, onehot_columns))

        feature_importance = feature_importance.sort_values(ascending=False)

        # print(feature_importance)
        if reg_clf=='reg':
            return r2, feature_importance

    def main(self, ret_span, label_type, feats_type, reg_clf='reg'):
        Xy = self.preprocessing(ret_span=ret_span, label_type=label_type, feats_type=feats_type)
        if reg_clf=='clf':
            Xy = self.reg2clf(Xy, 'fwd_ret')

        fwd_chained_dfs = self.forward_chaining(Xy, n=1)
        r2_scores = {}
        feat_imps = {}
        for train_test_indices, train, test in fwd_chained_dfs: ## TODO remove later
            train_min, train_max, test_min, test_max = train_test_indices
            # print(f"train years: {train_min} to {train_max}, test years: {test_min}")
            r2, feature_importance = self.ml(train_Xy=train, test_Xy=test, target_col="fwd_ret",feats2remove="ticker", reg_clf=reg_clf)
            r2_scores[test_max] = r2
            feat_imps[test_max] = feature_importance
        r2_scores = pd.DataFrame.from_dict(r2_scores, orient='index')
        feat_imps = pd.DataFrame.from_dict(feat_imps, orient='index')

        print(tabulate(r2_scores.round(3), tablefmt="github", headers=['R2']))
        print(tabulate(feat_imps.round(3), tablefmt="github", headers=feat_imps.columns))







pp = process("configs.yml").main(ret_span=5, label_type='risk_adj', feats_type="log")
