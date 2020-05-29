import pandas as pd
import numpy as np
import yaml
import swifter
import time
from FFD import test_get_optimal_ffd, fracDiff_FFD



class ffdProcess:
    def __init__(self, configs_loc):
        with open(configs_loc) as file:
            self.configs = yaml.load(file, Loader=yaml.FullLoader)


        self.log_adj_close = pd.read_pickle(self.configs["log_adj_close"])

    def ffd(self):

        start_time = time.time()
        # print(test_get_optimal_ffd(self.log_adj_close["AAPL"], np.arange(0.01,1.01, 0.01))) ## 0.34 for AAPL, 7mins

        opt_ds = [test_get_optimal_ffd(self.log_adj_close[col], np.arange(0.01,1.01, 0.01)) for col in self.log_adj_close.columns.tolist()]
        d_dict = dict(zip(self.log_adj_close.columns.tolist(), opt_ds))

        with open('opt_d_ffd.yml', 'w') as outfile:
            yaml.dump(d_dict, outfile, default_flow_style=False)

        # print(fracDiff_FFD(self.log_adj_close, 0.34))

        print(f"time elapsed: {(time.time()-start_time)/60} mins")



ffd = ffdProcess("configs.yml").ffd()