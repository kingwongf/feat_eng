import ray
from stock_pred import process



li_label_types =['trend', 'raw_ret', 'risk_adj']
li_feature_types = ['ffd', 'log', 'raw_price']
li_ret_spans = [5,20]
pp = process("configs.yml").main(ret_span=5, label_type='risk_adj', feats_type="log")