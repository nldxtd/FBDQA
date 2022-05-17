import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def data_process():

    file_dir = "./data"
    syms = list(range(10))
    dates = list(range(79))
    times = ['am', 'pm']
    df = pd.DataFrame()

    for sym in syms:
        for date in dates:
            for time in times:
                file_name = f"snapshot_sym{sym}_date{date}_{time}.csv"
                if not os.path.isfile(os.path.join(file_dir,file_name)):
                    continue
                new_df = pd.read_csv(os.path.join(file_dir,file_name))

                # 价格+1（从涨跌幅还原到对前收盘价的比例）
                new_df['bid1'] = new_df['n_bid1']+1
                new_df['bid2'] = new_df['n_bid2']+1
                new_df['bid3'] = new_df['n_bid3']+1
                new_df['bid4'] = new_df['n_bid4']+1
                new_df['bid5'] = new_df['n_bid5']+1
                new_df['ask1'] = new_df['n_ask1']+1
                new_df['ask2'] = new_df['n_ask2']+1
                new_df['ask3'] = new_df['n_ask3']+1
                new_df['ask4'] = new_df['n_ask4']+1
                new_df['ask5'] = new_df['n_ask5']+1
                
                # 量价组合
                new_df['spread1'] =  new_df['ask1'] - new_df['bid1']
                new_df['spread2'] =  new_df['ask2'] - new_df['bid2']
                new_df['spread3'] =  new_df['ask3'] - new_df['bid3']
                new_df['mid_price1'] =  new_df['ask1'] + new_df['bid1']
                new_df['mid_price2'] =  new_df['ask2'] + new_df['bid2']
                new_df['mid_price3'] =  new_df['ask3'] + new_df['bid3']
                new_df['weighted_ab1'] = (new_df['ask1'] * new_df['n_bsize1'] + new_df['bid1'] * new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['weighted_ab2'] = (new_df['ask2'] * new_df['n_bsize2'] + new_df['bid2'] * new_df['n_asize2']) / (new_df['n_bsize2'] + new_df['n_asize2'])
                new_df['weighted_ab3'] = (new_df['ask3'] * new_df['n_bsize3'] + new_df['bid3'] * new_df['n_asize3']) / (new_df['n_bsize3'] + new_df['n_asize3'])
                
                new_df['vol1_rel_diff']   = (new_df['n_bsize1'] - new_df['n_asize1']) / (new_df['n_bsize1'] + new_df['n_asize1'])
                new_df['volall_rel_diff'] = (new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                - new_df['n_asize1'] - new_df['n_asize2'] - new_df['n_asize3'] - new_df['n_asize4'] - new_df['n_asize5'] ) / \
                                ( new_df['n_bsize1'] + new_df['n_bsize2'] + new_df['n_bsize3'] + new_df['n_bsize4'] + new_df['n_bsize5'] \
                                + new_df['n_asize1'] + new_df['n_asize2'] + new_df['n_asize3'] + new_df['n_asize4'] + new_df['n_asize5'] )
                                        
                new_df['amount'] = new_df['amount_delta'].map(np.log1p)
                df = df.append(new_df)

def main():
    data_process()

if __name__ == "main":
    main()