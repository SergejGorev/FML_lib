import numpy as np
import pandas as pd
import datetime as dt
import warnings
import pickle

class DataImportClass:
    data_pickle_path_for_dump = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.pickle"

    history_ask_file = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/EURUSD_5 Mins_Ask_2005.01.01_2018.08.23.csv"  # r'd:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\01-Data\EURUSD_5 Mins_Ask_2005.01.01_2018.08.23.csv'
    history_bid_file = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/EURUSD_5 Mins_Bid_2005.01.01_2018.08.23.csv"  # r'd:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\01-Data\EURUSD_5 Mins_Bid_2005.01.01_2018.08.23.csv'
    # ---


    def execute(self):
        # считываем M5-историю, создаём датафреймы и делаем дампы
        symbol = pd.read_csv(history_ask_file, sep=';')
        symbol.columns = ['Datetime', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask']
        symbol['Datetime'] = symbol['Datetime'].apply(
                    lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M:%S"))
        symbol.index = symbol['Datetime']
        symbol['open_ask'] = symbol['open_ask'].apply(lambda x: float(x.replace(',', '.')))
        symbol['high_ask'] = symbol['high_ask'].apply(lambda x: float(x.replace(',', '.')))
        symbol['low_ask'] = symbol['low_ask'].apply(lambda x: float(x.replace(',', '.')))
        symbol['close_ask'] = symbol['close_ask'].apply(lambda x: float(x.replace(',', '.')))
        symbol['volume_ask'] = symbol['volume_ask'].apply(lambda x: float(x.replace(',', '.')))
        symbol.drop(columns=['Datetime'], inplace=True)
        symbol['abs_change_ask'] = symbol['close_ask'] - symbol['open_ask']
        symbol['abs_change_h_ask'] = symbol['high_ask'] - symbol['open_ask']
        symbol['abs_change_l_ask'] = symbol['low_ask'] - symbol['open_ask']
        #---
        symbol_bid = pd.read_csv(history_bid_file, sep=';')
        symbol_bid.columns = ['Datetime', 'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid']
        symbol_bid['Datetime'] = symbol_bid['Datetime'].apply(
                    lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M:%S"))
        symbol_bid.index = symbol_bid['Datetime']
        symbol_bid['open_bid'] = symbol_bid['open_bid'].apply(lambda x: float(x.replace(',', '.')))
        symbol_bid['high_bid'] = symbol_bid['high_bid'].apply(lambda x: float(x.replace(',', '.')))
        symbol_bid['low_bid'] = symbol_bid['low_bid'].apply(lambda x: float(x.replace(',', '.')))
        symbol_bid['close_bid'] = symbol_bid['close_bid'].apply(lambda x: float(x.replace(',', '.')))
        symbol_bid['volume_bid'] = symbol_bid['volume_bid'].apply(lambda x: float(x.replace(',', '.')))
        symbol_bid.drop(columns=['Datetime'], inplace=True)
        symbol_bid['abs_change_bid'] = symbol_bid['close_bid'] - symbol_bid['open_bid']
        symbol_bid['abs_change_h_bid'] = symbol_bid['high_bid'] - symbol_bid['open_bid']
        symbol_bid['abs_change_l_bid'] = symbol_bid['low_bid'] - symbol_bid['open_bid']
        #---
        symbol['open_bid'] = symbol_bid['open_bid']
        symbol['high_bid'] = symbol_bid['high_bid']
        symbol['low_bid'] = symbol_bid['low_bid']
        symbol['close_bid'] = symbol_bid['close_bid']
        symbol['volume_bid'] = symbol_bid['volume_bid']
        symbol['abs_change_bid'] = symbol_bid['abs_change_bid']
        symbol['abs_change_h_bid'] = symbol_bid['abs_change_h_bid']
        symbol['abs_change_l_bid'] = symbol_bid['abs_change_l_bid']
        #---
        symbol['spread'] = symbol['open_ask'] - symbol['open_bid']
        #---
        symbol['open'] = symbol['open_ask']
        symbol['high'] = symbol['high_ask']
        symbol['low'] = symbol['low_ask']
        symbol['close'] = symbol['close_ask']
        #---
        symbol['abs_change'] = symbol['abs_change_ask']
        symbol['abs_change_h'] = symbol['abs_change_h_ask']
        symbol['abs_change_l'] = symbol['abs_change_l_ask']
        #---

        pckl = open(self.data_pickle_path_for_dump, "wb")
        pickle.dump(symbol, pckl)
        pckl.close()


if __name__ == '__main__':
    req = DataImportClass()
    req.execute()