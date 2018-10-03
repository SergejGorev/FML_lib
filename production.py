import features_creating
import numpy as np
import pandas as pd
import warnings
import talib as tl
import datetime as dt
import time
import xgboost as xgb
import pickle
import os
from random import choice
# import copy

# !!! IT'S IN DEVELOPMENT !!!

class Production:
    #---
    log_save = True
    prefix = 'EURUSD_1.0'
    root_folder_path = r'd:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD'
    log_file_path = root_folder_path+os.sep+'04-Production'+os.sep+'EURUSD_ML_1.0_log.log'
    price_ask_file_path = root_folder_path+os.sep+'04-Production'+os.sep+'jforex_history_ask.csv'
    price_bid_file_path = root_folder_path+os.sep+'04-Production'+os.sep+'jforex_history_bid.csv'
    df_pickle_path = root_folder_path
    pred_file_JF_path = root_folder_path+os.sep+'04-Production'+os.sep+'ml_prediction_eurusd.txt'
    #---
    # *** SETTINGS: ***
    test_mode = False
    sleep_time = 1
    dump_updated_df_pickle = True
    data_back_step = 1  # на сколько индексов отступаем назад для обновления данных

    #---
    ml_models = ['eurusd_01']
    ml_models_files = ['model_eurusd_01.xgb']
    df_pickles = [root_folder_path+os.sep+'04-Production'+os.sep+'eurusd_5_v1.4_short.pickle']
    df_upd_pickles = [root_folder_path+os.sep+'04-Production'+os.sep+'eurusd_5_v1.4_upd.pickle']

    # thresholds = [(.0, .0)]

    #fair_constants = [2.0, 2.0, 2.0]

    ml_features = [
            ['lr_duo_1440_5i0',
             'lr_duo_1152_5i0',
             'ema_720',
             'lr_duo_288_5i0',
             'adx_576',
             'lr_duo_576_5i0',
             'lr_duo_1152_2i5',
             'lr_uno_1440_1i5',
             'sr_1440',
             'lr_uno_1440_5i0',
             'lr_uno_1440_2i5',
             'lr_duo_576_2i5',
             'lr_duo_864_5i0',
             'lr_duo_1440_2i5',
             'lr_uno_1152_2i5',
             'ema_576',
             'lr_duo_720_5i0',
             'lr_uno_1152_5i0',
             'lr_uno_1152_1i5',
             'ema_432',
             'lr_duo_864_1i5',
             'adx_432',
             'tema_288',
             'lr_duo_720_1i5',
             'tema_12',
             'tema_720',
             'adx_720',
             'dema_6',
             'lr_duo_864_2i5',
             'lr_duo_1440_1i5',
             'lr_duo_288_1i5',
             'ema_288',
             'adi_6',
             'ema_60',
             'tema_6',
             'tema_190',
             'lr_cmpr_1152_5i0',
             'ema_18',
             'open',
             'tema_432',
             'hurst_1440_10',
             'tema_576',
             'rtrn_864',
             'dema_576',
             'tema_24',
             'lr_cmpr_1440_5i0',
             'sr_432',
             'ema_6',
             'adx_144',
             'dema_72',
             'sr_720',
             'tema_18',
             'dema_12',
             'adx_288',
             'adi_720',
             'rtrn_1440',
             'ema_12',
             'adi_1440',
             'dema_108',
             'dema_18',
             'dema_432',
             'lr_duo_108_5i0',
             'ema_24',
             'dema_144',
             'lr_duo_72_5i0',
             'adi_60',
             'ema_48',
             'bb_rp_1440_1i0',
             'adi_12',
             'hurst_720_50',
             'adi_24',
             'ema_72',
             'lr_duo_1152_1i5',
             'dema_288',
             'adi_48',
             'lr_cmpr_720_5i0',
             'lr_duo_190_5i0',
             'adi_18',
             'ema_144',
             'hurst_1440_25',
             'ema_36',
             'sr_576',
             'adi_576',
             'bb_rp_1440_3i0',
             'hurst_1440_50',
             'lr_cmpr_1440_2i5',
             'adx_190',
             'adi_36',
             'tema_144',
             'dema_720',
             'ema_108',
             'bb_rp_1440_2i0',
             'tema_108',
             'dema_190',
             'lr_duo_190_2i5',
             'hurst_576_50',
             'cci_1440',
             'hurst_576_25',
             'adi_72',
             'hurst_864_50',
             'rsi_720',
             'macd_s_117_234_78',
             'dema_36',
             'adx_108',
             'dema_24',
             'lr_cmpr_864_5i0',
             'hurst_720_25',
             'hurst_576_10',
             'tema_72',
             'adi_432',
             'rtrn_1152',
             'rtrn_720',
             'hurst_864_10',
             'lr_duo_720_2i5',
             'rtrn_576',
             'hurst_1152_50',
             'adi_108',
             'macd_117_234_78',
             'sr_288',
             'adi_144',
             'ema_open_720',
             'hurst_720_10',
             'ema_cmpr_6_720',
             'hurst_288_25',
             'hurst_1152_10',
             'lr_uno_576_1i5',
             'lr_uno_288_1i5',
             'lr_uno_576_2i5',
             'cci_1152',
             'rsi_576',
             'hurst_288_50',
             'lr_uno_864_5i0',
             'lr_cmpr_576_2i5',
             'cci_720',
             'hurst_432_50',
             'lr_cmpr_1152_1i5',
             'hurst_432_10',
             'lr_uno_864_2i5',
             'lr_uno_576_5i0',
             'cci_864',
             'lr_duo_576_1i5',
             'lr_uno_864_1i5',
             'mfi_720',
             'mfi_288',
             'lr_duo_108_2i5',
             'lr_uno_288_5i0',
             'adi_288',
             'hurst_864_25',
             'mfi_576',
             'lr_cmpr_1152_2i5',
             'lr_uno_720_1i5',
             'lr_cmpr_1440_1i5',
             'ema_open_576',
             'tema_open_720',
             'hurst_1152_25',
             'lr_cmpr_720_2i5',
             'lr_uno_720_5i0',
             'lr_duo_190_1i5',
             'ema_open_288',
             'lr_uno_288_2i5',
             'tema_36',
             'hurst_288_10',
             'cci_576',
             'tema_cmpr_6_720',
             'cci_288',
             'lr_cmpr_576_5i0',
             'dema_open_720',
             'cci_432',
             'ema_open_432',
             'mfi_432',
             'lr_cmpr_864_2i5',
             'ema_cmpr_6_288',
             'dema_cmpr_6_720',
             'dema_open_576',
             'lr_uno_720_2i5',
             'tema_open_576',
             'hurst_432_25',
             'lr_duo_288_2i5',
             'bb_rp_576_1i0',
             'lr_cmpr_864_1i5',
             'dema_cmpr_6_432',
             'bb_rp_576_2i0',
             'bb_rp_720_1i0',
             'lr_cmpr_576_1i5',
             'dema_open_432',
             'lr_uno_190_5i0',
             'bb_rp_720_3i0',
             'rtrn_190',
             'so_k_234_2',
             'bb_rp_288_3i0',
             'lr_duo_36_5i0',
             'lr_uno_190_1i5',
             'rsi_144',
             'so_d_234_2',
             'bb_rp_576_3i0',
             'bb_rp_720_2i0',
             'adx_72',
             'rtrn_144',
             'lr_duo_108_1i5',
             'lr_uno_190_2i5']
                  ]

    # # Расчёт BBANDS - Bollinger Bands
    # timeperiod_bb = [int(4*24*0.75), int(4*24*0.75), int(4*24*0.75)]
    # nbdevup, nbdevdn = [1.5, 1.5, 1.5], [1.5, 1.5, 1.5]
    #
    # # Расчёт "Аллигатора" (3 скользящие средние)
    # timeperiod_al = [3, 3, 3]
    # mult_al = [1.5, 1.5, 1.5]
    #
    # # Расчёт экспонента Хёрста (hurst_ernie_chan(p))
    # max_lag = [25, 25, 25]
    # timeperiod_h = [int(4*24*0.75), int(4*24*0.75), int(4*24*0.75)]
    #
    # # Расчёт наклона линий линейной регрессии для двух периодов (короткий и длинный)
    # timeperiod_lr = [4*12, 4*12, 4*12]
    # mult_lr = [5.0, 5.0, 5.0]

    #---
    log_file = None
    #---
    ml_prediction_arr = []
    #---

    def log_and_print(self, my_str):
        """
        The function for information output, incl. in log file
        :param my_str: string.
        :return: None.
        """
        print(my_str)
        if self.log_save:
            self.log_file.write(my_str)


    # def data_added_update(df, rows_count):
    #     """
    #     Сalculation features values for the added data
    #     :param df: pandas dataframe.
    #     :param rows_count: integer.
    #         The count of dataframe last rows for features calculation.
    #     :return: None
    #     """
    #     warnings.filterwarnings('ignore')
    #     ind_arr = df.index[-rows_count:]
    #     # df.loc[ind_arr, 'abs_change'] = df.loc[ind_arr, 'close'] - df.loc[ind_arr, 'open']
    #     # df.loc[ind_arr, 'abs_change_h'] = df.loc[ind_arr, 'high'] - df.loc[ind_arr, 'open']
    #     # df.loc[ind_arr, 'abs_change_l'] = df.loc[ind_arr, 'low'] - df.loc[ind_arr, 'open']
    #
    #     #--- BB
    #     bb_indent = timeperiod_bb[indx] - 1 + rows_count
    #     ubb_arr, mbb_arr, lbb_arr = \
    #     tl.BBANDS(df.loc[df.index[-bb_indent:], 'open'].values, timeperiod=timeperiod_bb[indx], \
    #               nbdevup=nbdevup[indx], nbdevdn=nbdevdn[indx], matype=0)
    #     #print(ubb_arr[-rows_count:])
    #     df.loc[ind_arr, 'ubb'] = ubb_arr[-rows_count:]
    #     df.loc[ind_arr, 'mbb'] = mbb_arr[-rows_count:]
    #     df.loc[ind_arr, 'lbb'] = lbb_arr[-rows_count:]
    #     df.loc[ind_arr, 'decision_mr'] = df.loc[ind_arr].apply(lambda x: 1 if x.open<x.lbb else (-1 if x.open>x.ubb else 0), axis=1)
    #
    #     #--- Alligator
    #     timeperiod_al_1 = timeperiod_al[indx]
    #     shift_al_1 = int(np.floor(timeperiod_al[indx]/2.5))
    #     timeperiod_al_2 = int(timeperiod_al_1*mult_al[indx])
    #     shift_al_2 = timeperiod_al_1 #5
    #     timeperiod_al_3 = int(timeperiod_al_2*mult_al[indx])
    #     shift_al_3 = timeperiod_al_2 #8
    #     al_indent = 1+timeperiod_al_1+shift_al_1+timeperiod_al_2+shift_al_2+timeperiod_al_3+shift_al_3+ rows_count
    #     al_1_arr = tl.SMA(df.loc[df.index[-al_indent:], 'open'].shift(shift_al_1).values, \
    #                       timeperiod=timeperiod_al_1)
    #     al_2_arr = tl.SMA(df.loc[df.index[-al_indent:], 'open'].shift(shift_al_2).values, \
    #                         timeperiod=timeperiod_al_2)
    #     al_3_arr = tl.SMA(df.loc[df.index[-al_indent:], 'open'].shift(shift_al_3).values, \
    #                         timeperiod=timeperiod_al_3)
    #
    #     alligator_answer = lambda x: 1 if ((x.al_1>x.al_2) & (x.al_2>x.al_3)) else \
    #                                 (-1 if ((x.al_1<x.al_2) & (x.al_2<x.al_3)) else 0)
    #     df.loc[ind_arr, 'al_1'] = al_1_arr[-rows_count:]
    #     df.loc[ind_arr, 'al_2'] = al_2_arr[-rows_count:]
    #     df.loc[ind_arr, 'al_3'] = al_3_arr[-rows_count:]
    #     df.loc[ind_arr, 'decision_tf'] = df.loc[ind_arr].apply(alligator_answer, axis=1)
    #
    #     #--- Hurst
    #     h_indent = timeperiod_h[indx] - 1 + rows_count
    #     lags = range(2,max_lag[indx])
    #     hurst_arr = df.loc[df.index[-h_indent:], 'open'].rolling(window=timeperiod_h[indx], \
    #                               center=False).apply(lambda x: hurst_ernie_chan(x, lags))
    #     print()
    #     df.loc[ind_arr, 'hurst'] = hurst_arr[-rows_count:]
    #
    #     #--- LR
    #     timeperiod_lr_1 = timeperiod_lr[indx]
    #     timeperiod_lr_2 = int(timeperiod_lr_1*mult_lr[indx])
    #     lr_indent = 1+timeperiod_lr_1+timeperiod_lr_2+ rows_count
    #
    #     lr_1_arr = tl.LINEARREG_SLOPE(df.loc[df.index[-lr_indent:], 'open'].values, \
    #                                   timeperiod=timeperiod_lr_1)
    #     df.loc[ind_arr, 'lr_1'] = lr_1_arr[-rows_count:]
    #     lr_2_arr = tl.LINEARREG_SLOPE(df.loc[df.index[-lr_indent:], 'open'].values, \
    #                                   timeperiod=timeperiod_lr_2)
    #     df.loc[ind_arr, 'lr_2'] = lr_2_arr[-rows_count:]
    #
    #     # расширение признаков за счёт добавления разности ценовых значений индикаторов и open
    #     # 'ubb_open', 'mbb_open', 'lbb_open', 'al_1_open', 'al_2_open', 'al_3_open'
    #     df.loc[ind_arr, 'ubb_open'] = df.loc[ind_arr, 'ubb'] - df.loc[ind_arr, 'open']
    #     df.loc[ind_arr, 'mbb_open'] = df.loc[ind_arr, 'mbb'] - df.loc[ind_arr, 'open']
    #     df.loc[ind_arr, 'lbb_open'] = df.loc[ind_arr, 'lbb'] - df.loc[ind_arr, 'open']
    #     df.loc[ind_arr, 'al_1_open'] = df.loc[ind_arr, 'al_1'] - df.loc[ind_arr, 'open']
    #     df.loc[ind_arr, 'al_2_open'] = df.loc[ind_arr, 'al_2'] - df.loc[ind_arr, 'open']
    #     df.loc[ind_arr, 'al_3_open'] = df.loc[ind_arr, 'al_3'] - df.loc[ind_arr, 'open']
    #
    #     warnings.filterwarnings('default')


    def main(self):
        modif_date_prev = dt.datetime(1970,1,1)
        if self.log_save:
            self.log_file = open(self.log_file_path, "a", encoding='utf-16')
            self.log_and_print('\nThe log file is opened.\n')

        while True:
            start_time = dt.datetime.now()
            try:
                modif_date = dt.datetime.fromtimestamp(os.path.getmtime(self.price_ask_file_path))
                print("{0}: modif_date= {1}, modif_date_prev= {2}, modif_date_equal= {3}".
                      format(self.prefix, modif_date, modif_date_prev, modif_date == modif_date_prev))
            except OSError:
                print('cannot open', self.price_ask_file_path)

            if modif_date > modif_date_prev:
                modif_date_prev = modif_date

                ml_prediction_arr = []
                for model_ix, model_name in enumerate(self.ml_models):
                    my_str = "model_ix= {0}, model_name= {1}".format(model_ix, model_name)
                    self.log_and_print(my_str)
                    #--- загружаем основной датафрейм
                    try:
                        df_pickle_path_ = self.df_pickles[model_ix]
                        with open(df_pickle_path_, "rb") as pckl:
                            data_df = pickle.load(pckl)
                    except OSError:
                        print('cannot open', df_pickle_path_)
                        break
                    #---
                    #---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    my_str = '\n'+str(start_time)+":\ndata_df.head(3):\n"+str(data_df.head(3))
                    self.log_and_print(my_str)
                    my_str = "\ndata_df.tail(3):\n"+str(data_df.tail(3))
                    self.log_and_print(my_str)
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    #--- читаем данные из файла
                    try:
                        price_df = pd.read_csv(self.price_ask_file_path, sep=';')
                        price_df.columns = ['Datetime', 'open_ask', 'high_ask', 'low_ask', 'close_ask', 'volume_ask']
                        price_df['Datetime'] = price_df['Datetime'].apply(
                            lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M"))
                        price_df.index = price_df['Datetime']
                        price_df['open_ask'] = price_df['open_ask'].apply(lambda x: float(x.replace(',', '.')))
                        price_df['high_ask'] = price_df['high_ask'].apply(lambda x: float(x.replace(',', '.')))
                        price_df['low_ask'] = price_df['low_ask'].apply(lambda x: float(x.replace(',', '.')))
                        price_df['close_ask'] = price_df['close_ask'].apply(lambda x: float(x.replace(',', '.')))
                        price_df['volume_ask'] = price_df['volume_ask'].apply(lambda x: float(x.replace(',', '.')))
                        price_df.drop(columns=['Datetime'], inplace=True)
                        # ---
                        price_df_bid = pd.read_csv(self.price_bid_file_path, sep=';')
                        price_df_bid.columns = ['Datetime', 'open_bid', 'high_bid', 'low_bid', 'close_bid', 'volume_bid']
                        price_df_bid['Datetime'] = price_df_bid['Datetime'].apply(
                            lambda x: dt.datetime.strptime(x, "%Y.%m.%d %H:%M"))
                        price_df_bid.index = price_df_bid['Datetime']
                        price_df_bid['open_bid'] = price_df_bid['open_bid'].apply(lambda x: float(x.replace(',', '.')))
                        price_df_bid['high_bid'] = price_df_bid['high_bid'].apply(lambda x: float(x.replace(',', '.')))
                        price_df_bid['low_bid'] = price_df_bid['low_bid'].apply(lambda x: float(x.replace(',', '.')))
                        price_df_bid['close_bid'] = price_df_bid['close_bid'].apply(lambda x: float(x.replace(',', '.')))
                        price_df_bid['volume_bid'] = price_df_bid['volume_bid'].apply(lambda x: float(x.replace(',', '.')))
                        price_df_bid.drop(columns=['Datetime'], inplace=True)
                        price_df_bid['abs_change_bid'] = price_df_bid['close_bid'] - price_df_bid['open_bid']
                        price_df_bid['abs_change_h_bid'] = price_df_bid['high_bid'] - price_df_bid['open_bid']
                        price_df_bid['abs_change_l_bid'] = price_df_bid['low_bid'] - price_df_bid['open_bid']
                        # ---
                        price_df['open_bid'] = price_df_bid['open_bid']
                        price_df['high_bid'] = price_df_bid['high_bid']
                        price_df['low_bid'] = price_df_bid['low_bid']
                        price_df['close_bid'] = price_df_bid['close_bid']
                        price_df['volume_bid'] = price_df_bid['volume_bid']
                        # ---
                        price_df['open'] = price_df['open_ask']
                        price_df['high'] = price_df['high_ask']
                        price_df['low'] = price_df['low_ask']
                        price_df['close'] = price_df['close_ask']
                        # ---
                        price_df['spread'] = price_df['open_ask'] - price_df['open_bid']
                        # ---
                        price_df['abs_change'] = price_df['close_ask'] - price_df['open_ask']
                        price_df['abs_change_h'] = price_df['high_ask'] - price_df['open_ask']
                        price_df['abs_change_l'] = price_df['low_ask'] - price_df['open_ask']
                        #print("price_df:\n"+str(price_df))
                    except BaseException:
                        my_str = '\ncannot read and proceed ' + str(self.price_ask_file_path)
                        self.log_and_print(my_str)
                        print("\n________________________________________________________________________\n")
                        continue

                    #---
                    my_str = "\nprice_df.head(3):\n"+str(price_df.head(3))
                    self.log_and_print(my_str)
                    my_str = "\nprice_df.tail(3):\n"+str(price_df.tail(3))
                    self.log_and_print(my_str)
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    #--- находим в индексах первую общую дату для сравнения индексов
                    date_1 = price_df.index[0]
                    date_2 = data_df.index[0]
                    if (date_1 > date_2):
                        start_date = date_1
                    else:
                        start_date = date_2
                    my_str = "\nstart_date for comparison= {}".format(start_date)
                    self.log_and_print(my_str)

                    #--- сравниваем индексы для находждения пропущенных данных
                    price_df_new = price_df.loc[price_df.index >= start_date, :]
                    data_df_new = data_df.loc[data_df.index >= start_date,
                                                        ['open', 'high', 'low', 'close', 'spread', 'volume_ask']]
                    #---
                    my_str = "price_df_new.shape= {}".format(price_df_new.shape)
                    self.log_and_print(my_str)
                    #---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    my_str = "\nprice_df_new.head():\n" + str(price_df_new.head())
                    self.log_and_print(my_str)
                    #---
                    my_str = "\nprice_df_new.tail():\n"+str(price_df_new.tail())
                    self.log_and_print(my_str)
                    #---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    my_str = "\ndata_df_new.shape= {}".format(data_df_new.shape)
                    self.log_and_print(my_str)
                    #---
                    my_str = "\n\ndata_df_new.head():\n"+str(data_df_new.head())
                    self.log_and_print(my_str)
                    #---
                    my_str = "\n\ndata_df_new.tail():\n"+str(data_df_new.tail())
                    self.log_and_print(my_str)
                    #---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---

                    df_check = pd.DataFrame(list(zip(price_df_new.index, data_df_new.index)), columns=['indx_1', 'indx_2'])
                    my_func = lambda x: True if x['indx_1'] == x['indx_2'] else False
                    df_check['check'] = df_check.apply(my_func, axis=1)
                    # print(df_check)
                    #--- определяем первую не совпадающую дату
                    lost_date = df_check.loc[df_check['check'] == False, 'indx_1']
                    try:
                        first_lost_date = list(lost_date.head(1))[0]
                        my_str = "\ntry: first_lost_date= {}".format(first_lost_date)
                        self.log_and_print(my_str)
                        # ---
                        first_lost_date_idx = lost_date.head(1).index[0]
                        my_str = "\ntry: first_lost_date_idx= {}".format(first_lost_date_idx)
                        self.log_and_print(my_str)
                        # ---
                    except IndexError:
                        # my_str = 'IndexError: new data not exist'
                        # print(my_str)
                        # print("\n________________________________________________________________________\n")
                        # if log_save:
                        #     log_file.write("\n" + my_str)
                        # continue
                        first_lost_date = list(df_check.loc[:, 'indx_1'].tail(1))[0]
                        my_str = "\nexcept: first_lost_date= {}".format(first_lost_date)
                        self.log_and_print(my_str)
                        # ---
                        first_lost_date_idx = df_check.index[-1]
                        my_str = "\nexcept: first_lost_date_idx= {}".format(first_lost_date_idx)
                        self.log_and_print(my_str)
                        # ---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    if (first_lost_date_idx >= self.data_back_step):
                        idx_req = first_lost_date_idx - self.data_back_step
                    prev_date = list(df_check.loc[df_check.index == idx_req, 'indx_1'])[0]
                    my_str = "\ntry: prev_date= {}".format(prev_date)
                    self.log_and_print(my_str)
                    # ---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    #--- объединяем датафреймы
                    data_df_for_concat = data_df.loc[data_df.index < prev_date]
                    # print(data_df_for_concat)
                    price_df_for_concat = price_df.loc[price_df.index >= prev_date]
                    data_df_new = pd.concat([data_df_for_concat, price_df_for_concat], sort=True)

                    features_for_str = ['open', 'high', 'low', 'close', 'spread', 'volume_ask']
                    clmn_arr = list(data_df_new.columns)
                    print('clmn_arr= ', clmn_arr)
                    for i in features_for_str: clmn_arr.remove(i)
                    # clmn_arr.remove('pred')
                    exp_arr = []
                    for i in range(2):
                        val = choice(clmn_arr)
                        exp_arr.append(val)
                        clmn_arr.remove(val)
                    features_for_str.extend(exp_arr)
                    # features_for_str.extend(['pred'])
                    my_str = "\ndata_df_new (head(3)):\n"+str(data_df_new.loc[:, features_for_str].head(3))
                    self.log_and_print(my_str)
                    my_str = "\ndata_df_new (tail(3)):\n"+str(data_df_new.loc[:, features_for_str].tail(3))
                    self.log_and_print(my_str)
                    #---
                    my_str = '-------------------------------------------------------------------------'
                    self.log_and_print(my_str)
                    #---
                    rows_count = price_df_for_concat.shape[0]
                    my_str = "\nrows_count= "+str(rows_count)
                    self.log_and_print(my_str)

                    feat_cr = features_creating.FeaturesCalcClass()
                    feat_cr.data_pickle_path_for_dump = self.df_upd_pickles[model_ix]
                    feat_cr.update_df(data_df_new)
                    print('data_df_new:\n', data_df_new)

            #         data_added_update(data_df_new, rows_count=rows_count, indx=model_ix)
            #         data_df = data_df_new
            #         #---
            #         #--- загружаем модель
            #         if test_mode == False:
            #             try:
            #                 model_path_ = model_path + ml_models_files[model_ix]
            #                 with open(model_path_, "rb") as pckl:
            #                     model = pickle.load(pckl)
            #                     pckl.close()
            #             except BaseException:
            #                 my_str = '\n\ncannot open ml model ' + str(model_path_)
            #                 self.log_and_print(my_str)
            #                 break
            #
            #             #---
            #             ind_arr = data_df.index[-rows_count:]
            #             X_pred = data_df.loc[ind_arr, ml_features[model_ix]]
            #             pred = model.predict(X_pred)
            #             #print('pred.shape= {}'.format(pred.shape))
            #
            #             data_df.loc[ind_arr, 'pred'] = pred
            #             # на основании предсказания рассчитываем финансовые показатели
            #
            #             pred_dir = list(map(lambda x: pred_transform_adapt(x, thresholds[model_ix][1],
            #                                                                thresholds[model_ix][0]), pred))
            #             data_df.loc[ind_arr, 'pred_dir'] = pred_dir
            #             #---
            #             pred_value = str(int(data_df.loc[data_df.index[-1:], 'pred_dir'].values[0]))
            #             my_str = '\n\npred_value= {}\n'.format(pred_value)
            #             self.log_and_print(my_str)
            #             # добавляем предсказание в массив
            #             ml_prediction_arr.append((str(data_df.index[-1]), str(pred_value), str(np.round(data_df.open[-1],5))))
            #
            #             # df pickle dump
            #             if(dump_updated_df_pickle):
            #                 with open(df_pickle_path_, "wb") as pckl:
            #                     pickle.dump(data_df, pckl)
            #                     pckl.close()
            #             #---
            #         else:  # test_mode == True
            #             rand_val = np.random.randint(-1, 1, 1)[0]
            #             pred_value = -1 if rand_val == -1 else 1
            #             my_str = '\n\ntest_mode: pred_value= {}\n'.format(pred_value)
            #             self.log_and_print(my_str)
            #             # добавляем предсказание в массив
            #             ml_prediction_arr.append((str(data_df.index[-1]), str(pred_value), str(np.round(data_df.open[-1],5))))
            #             #---
            #         #---
            #         clmns_ = ['open', 'abs_change', 'al_1', 'al_1_open', 'pred', 'pred_dir', 'target']
            #         my_str = '\n\n'+str(data_df.loc[data_df.index[-5:], clmns_])
            #         self.log_and_print(my_str)
            #         #---
            #         my_str = '\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
            #         self.log_and_print(my_str)
            #         #---
            #
            #     # формируем итоговое предсказание
            #     prediction_sum = 0
            #     prediction_dt = ''
            #     prediction_open = ''
            #     for prediction_ix, prediction in enumerate(ml_prediction_arr):
            #         my_str = '\nprediction_ix: {0}, prediction= {1}'.format(prediction_ix, prediction)
            #         self.log_and_print(my_str)
            #         prediction_sum += int(prediction[1])
            #         prediction_dt = prediction[0]
            #         prediction_open = prediction[2]
            #
            #     prediction_final = 0
            #     if prediction_sum > 0:
            #         prediction_final = 1
            #     elif prediction_sum < 0:
            #         prediction_final = -1
            #
            #     my_str = '\nprediction_dt= {0}, prediction_final= {1}, prediction_open= {2}'.format(prediction_dt,
            #                                                                 prediction_final, prediction_open)
            #     self.log_and_print(my_str)
            #
            #     if test_mode == False:
            #         # # формируем записи в файлах для торговых советников MT5 и JForex
            #         # try:
            #         #     with open(pred_file_MT5_path, "w", encoding='utf-16') as f:
            #         #         f.write(prediction_dt)
            #         #         f.write("\n")
            #         #         f.write(str(prediction_final))
            #         #         f.write("\n")
            #         #         f.write(prediction_open)
            #         #         f.close()
            #         # except BaseException:
            #         #     my_str = '\n\ncannot write MT5 pred_value '+str(pred_file_MT5_path)
            #         #     self.log_and_print(my_str)
            #
            #         try:
            #             with open(pred_file_JF_path, "w", encoding='utf-8') as f:
            #                 f.write(prediction_dt)
            #                 f.write("\n")
            #                 f.write(str(prediction_final))
            #                 f.write("\n")
            #                 f.write(prediction_open)
            #                 f.close()
            #         except BaseException:
            #             my_str = '\n\ncannot write JF pred_value '+str(pred_file_JF_path)
            #             self.log_and_print(my_str)
            #
            #         # #--- запись сигнального файла
            #         # try:
            #         #     signal_ = open(signal_file_MT5_path, "wt")
            #         #     signal_.write(str(1))
            #         #     signal_.close()
            #         # except BaseException:
            #         #     my_str = '\n\ncannot write signal'+str(signal_file_MT5_path)
            #         #     self.log_and_print(my_str)
            #         # #---
            #     else:  # test_mode == True
            #         # try:
            #         #     with open(pred_file_MT5_path, "w", encoding='utf-16') as f:
            #         #         f.write(prediction_dt)
            #         #         f.write("\n")
            #         #         f.write(str(prediction_final))
            #         #         f.write("\n")
            #         #         f.write(prediction_open)
            #         #         f.close()
            #         # except BaseException:
            #         #     my_str = '\n\ntest_mode: cannot write MT5 pred_value '+str(pred_file_MT5_path)
            #         #     self.log_and_print(my_str)
            #
            #         try:
            #             with open(pred_file_JF_path, "w", encoding='utf-8') as f:
            #                 f.write(prediction_dt)
            #                 f.write("\n")
            #                 f.write(str(prediction_final))
            #                 f.write("\n")
            #                 f.write(prediction_open)
            #                 f.close()
            #         except BaseException:
            #             my_str = '\n\ntest_mode: cannot write JF pred_value '+str(pred_file_JF_path)
            #             self.log_and_print(my_str)
            #
            #         # #--- запись сигнального файла
            #         # try:
            #         #     signal_ = open(signal_file_MT5_path, "wt")
            #         #     signal_.write(str(1))
            #         #     signal_.close()
            #         # except BaseException:
            #         #     my_str = '\n\ntest_mode: cannot write signal '+str(signal_file_MT5_path)
            #         #     self.log_and_print(my_str)
            #         # #---
            #
            #     finish_time = dt.datetime.now()
            #     duration = finish_time-start_time
            #     my_str = u'\n\ncalculation duration= {0}, finish_time= {1}'.format(duration, finish_time)
            #     self.log_and_print(my_str)
            #     my_str = "\n________________________________________________________________________\n"
            #     self.log_and_print(my_str)
            #
            #     if log_save:
            #         log_file.flush()
            else:
                time.sleep(self.sleep_time)

        if self.log_save:
            self.log_file.close()
        #---

if __name__ == "__main__":
    req = Production()
    req.main()