import processing
import finfunctions
import multiprocessing as mp
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import talib as tl
import datetime as dt
import time
import random
import pickle
#import sys


class FeaturesCalcClass:
    #---
    num_threads = 8
    data_pickle_path = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.pickle"
    dump_pickle = True # dump data pickle
    data_pickle_path_for_dump = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.2.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.2.pickle"

    #---
    # 1. Hurst exponent
    # !!! Расчёт осуществляется очень долго!!!
    # Для нескольких проходов настоятельно рекомендуется использовать мультипоточность вне jupyter'а.
    hurst_tmprd_arr = [72, 288, 432, 576, 720, 864, 1152, 1440]
    hurst_max_lag_arr = [10, 25, 50]
    hurst_base_clmn_name = 'hurst'

    # 2. Exponential Moving Averages
    ema_tmprd_arr = [6, 108, 288, 432, 576, 720]
    ema_base_clmn_name = 'ema'
    ema_fin_func = tl.EMA

    # 3. Sharpe Ratio
    sr_tmprd_arr = [108, 288, 576, 720, 1440]
    sr_base_clmn_name = 'sr'
    #sr_fin_func = finfunctions.sharpe_ratio_feature

    # 4. Accumulation/Distribution Index (ADI)
    adi_tmprd_arr = [108, 288, 576, 720, 1440]
    adi_base_clmn_name = 'adi'
    #adi_fin_func = finfunctions.adi

    # 5. Bollinger Bands (BB)
    bb_tmprd_arr = [72, 108, 190, 288, 576, 720]
    bb_d_arr = [0.5, 1., 1.5, 2., 2.5, 3.]
    bb_base_clmn_names = ['ubb', 'mbb', 'lbb']
    bb_fin_func = tl.BBANDS

    # 6. Stochastic Oscillator
    so_param_arr = [(5, 2), (6, 3), (12, 3), (36, 2), (78, 2), (234, 2)]
    so_k_slowk_period = 0.5
    so_base_clmn_names = ['so_k', 'so_d']
    so_fin_func = tl.STOCH
    # STOCH(high, low, close[, fastk_period=?, slowk_period=?, slowk_matype=?, slowd_period=?, slowd_matype=?])

    # 7. Relative Strength Index (RSI)
    rsi_tmprd_arr = [6, 12, 36, 72, 108, 190, 288, 576, 720]
    rsi_base_clmn_name = 'rsi'
    rsi_fin_func = tl.RSI

    # 8. Commodity Channel Index (CCI)
    cci_tmprd_arr = [3, 6, 9, 12, 36, 72, 108, 190, 288, 576, 720, 864, 1152, 1440]
    cci_base_clmn_name = 'cci'
    cci_fin_func = tl.CCI
    # CCI(high, low, close[, timeperiod=?])

    # 9. Average Directional Moving Index (ADX)
    adx_tmprd_arr = [36, 72, 108, 190, 288, 576, 720, 864, 1152, 1440]
    adx_base_clmn_name = 'adx'
    adx_fin_func = tl.ADX
    # ADX(high, low, close[, timeperiod=?])

    # 10.1. Double Exponentially Smoothed Returns
    dema_tmprd_arr = [108, 190, 288, 576, 720, 864, 1152, 1440]
    dema_base_clmn_name = 'dema'
    dema_fin_func = tl.DEMA

    # 10.2. Triple Exponentially Smoothed Returns
    tema_tmprd_arr = [6, 12, 36, 72, 108, 190, 288, 576, 720, 864, 1152, 1440]
    tema_base_clmn_name = 'tema'
    tema_fin_func = tl.TEMA

    # 11. Moving Average Convergence-Divergence (MACD)
    macd_params_arr = [(3, 6, 2), (6, 12, 4), (18, 36, 12), (36, 78, 26), (117, 234, 78)]
    macd_base_clmn_names = ['macd', 'macd_s', 'macd_h']
    macd_fin_func = tl.MACD
    # MACD(real[, fastperiod=?, slowperiod=?, signalperiod=?])
    # Outputs: macd, macdsignal, macdhist

    # 12. Money Flow Index (MFI)
    mfi_tmprd_arr = [36, 72, 108, 190, 288, 576, 720, 864, 1152, 1440]
    mfi_base_clmn_name = 'mfi'
    mfi_fin_func = tl.MFI
    # MFI(high, low, close, volume[, timeperiod=?])
    # Outputs: real

    # 13. Linear regression slope
    lr_tmprd_arr = [36, 72, 108, 190, 288, 576, 720, 864, 1152, 1440]
    lr_mult_arr = [1.5, 2.5, 5.]
    lr_base_clmn_names = ['lr_uno', 'lr_duo']
    lr_fin_func = tl.LINEARREG_SLOPE

    def sharpe_ratio_feature(self, return_values):
        """
        Calculate Sharpe Ratio for creating feature.

        Returns Sharpe Ratio value.

        Parameters
        ----------
        return_values :  float array, float ndarray
            Array with return values.

        Returns
        -------
        sharpe_ratio : float
            Sharpe ratio value.

        """
        mean_ = np.mean(return_values)
        std_ = np.std(return_values)
        # print("mean_= {1:.4f}, std_= {2:.4f}".format(mean_, std_))
        return_values_len = len(return_values)
        if (std_ != 0.) & return_values_len != 0.:
            res = mean_ / (std_ * return_values_len**0.5)
        else:
            res = 0.
        return res


    def adi(self, df, window, price_open_clmn, price_high_clmn, price_low_clmn, price_close_clmn, vol_clmn):
        """
        Calculate Accumulation/Distribution Index (ADI).

        Returns ADI value.

        Parameters
        ----------
        df : dataframe.
            Dataframe with price (close, high, low) and volume values.
        window :  integer
            Size of rolling window.
        price_open_clmn :  string
            Open price column name.
        price_high_clmn :  string
            High price column name.
        price_low_clmn :  string
            Low price column name.
        price_close_clmn :  string
            Close price column name.
        vol_clmn :  string
            Volume column name.

        Returns
        -------
        adi : float
            ADI value.
        """
        df_calc = df.loc[:, [price_high_clmn, price_low_clmn, price_close_clmn]]
        roll_res = df_calc.rolling(window=window, center=False).apply(np.max).values
        df_calc['high_roll'] = [x[0] for x in roll_res]
        df_calc['low_roll'] = [x[1] for x in roll_res]
        df_calc['close_roll'] = [x[2] for x in roll_res]
        df_calc['open'] = df[price_open_clmn]
        df_calc['volume'] = df[vol_clmn]
        df_calc['vol_roll'] = df_calc['volume'].rolling(window=window, center=False).apply(np.sum)
        df_calc['high_roll'] = df_calc['high_roll'].shift(1)
        df_calc['low_roll'] = df_calc['low_roll'].shift(1)
        df_calc['close_roll'] = df_calc['close_roll'].shift(1)
        df_calc['vol_roll'] = df_calc['vol_roll'].shift(1)
        func_clv = lambda x: ((x['open'] - x['low']) - (x['high'] - x['open'])) / (x['high'] - x['low']) \
            if (x['high'] - x['low']) != 0. else 0.
        df_calc['clv'] = df_calc.apply(func_clv, axis=1)

        df_calc['res'] = df_calc['clv'] * df_calc['vol_roll']
        return df_calc['res'].values


    def hurst_calc(self, *args):
        df = args[0]
        new_clmn_names = args[1]
        tmprd_hurst = args[2]
        lags = args[3]

        result = (new_clmn_names[0], df['open'].rolling(window=tmprd_hurst, center=False).apply(
                                                                        lambda x: finfunctions.hurst_ernie_chan(x, lags)))
        print('%s says result for args %s = %s' % (mp.current_process().name, args[1], result))
        return args[1], result


    def run_hurst_calc(self, df, dump_pickle=True):
        time_start = dt.datetime.now()
        print('hurst: time_start= {}'.format(time_start))
        print('cpu_count() = %d\n' % mp.cpu_count())

        #--- Create pool
        print('Creating pool with %d processes\n' % self.num_threads)
        pool = mp.Pool(self.num_threads)
        print('pool = %s' % pool)
        print()

        TASKS = []

        for tmprd_hurst in self.hurst_tmprd_arr:
            for max_lag in self.hurst_max_lag_arr:
                postfix = '_' + processing.digit_to_text(tmprd_hurst) + '_' + processing.digit_to_text(max_lag)
                new_clmn_names = [self.hurst_base_clmn_name + postfix]
                lags = range(2, max_lag)
                TASKS.append((df, new_clmn_names, tmprd_hurst, lags))

        #print('TASKS:\n', TASKS)
        results = [pool.apply_async(self.hurst_calc, t) for t in TASKS]

        for r in results:
            res = r.get()
            print('\t', res)
            print('type(res)= {0}, res= \n{1}'.format(type(res), res))
            clmn_name = res[0][0]
            print('res[1][1]: \n', res[1][1])
            clmn_data = res[1][1].values
            print('clmn_name= {0}, clmn_data:\n {1}'.format(clmn_name, clmn_data))
            df[clmn_name] = clmn_data

        if dump_pickle:
            pckl = open(self.data_pickle_path_for_dump, "wb")
            pickle.dump(df, pckl)
            pckl.close()

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('hurst: time_finish= {0}, duration= {1}'.format(time_start, time_duration))

        return df


    def ema_calc(self, df):
        for tmprd_ema in self.ema_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_ema)
            new_clmn_names = [self.ema_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.ema_fin_func, 'add_columns': new_clmn_names, 'shift': 0,
                   'real': df['open'].values, 'timeperiod': tmprd_ema}
            df = processing.features_add(**inp)

            inp = (df, 'open', new_clmn_names[0], 'ema_open' + postfix)
            df = processing.clmn_compare(*inp)

        return df


    def sharpe_calc(self, *args):
        df = args[0]
        new_clmn_names = args[1]
        tmprd_sr = args[2]

        result = (new_clmn_names[0], df['open'].rolling(window=tmprd_sr, center=False).apply(func=self.sharpe_ratio_feature))
        print('%s says result for args %s = %s' % (mp.current_process().name, args[1], result))
        return args[1], result


    def run_sharpe_calc(self, df, dump_pickle=True):
        time_start = dt.datetime.now()
        print('sharpe: time_start= {}'.format(time_start))
        print('cpu_count() = %d\n' % mp.cpu_count())

        df['return'] = df['open'].pct_change()

        #--- Create pool
        print('Creating pool with %d processes\n' % self.num_threads)
        pool = mp.Pool(self.num_threads)
        print('pool = %s' % pool)
        print()

        TASKS = []

        for tmprd_sr in self.sr_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_sr)
            new_clmn_names = [self.sr_base_clmn_name + postfix]
            TASKS.append((df, new_clmn_names, tmprd_sr))

        #print('TASKS:\n', TASKS)
        results = [pool.apply_async(self.sharpe_calc, t) for t in TASKS]

        for r in results:
            res = r.get()
            print('\t', res)
            print('type(res)= {0}, res= \n{1}'.format(type(res), res))
            clmn_name = res[0][0]
            print('res[1][1]: \n', res[1][1])
            clmn_data = res[1][1].values
            print('clmn_name= {0}, clmn_data:\n {1}'.format(clmn_name, clmn_data))
            df[clmn_name] = clmn_data

        if dump_pickle:
            pckl = open(self.data_pickle_path_for_dump, "wb")
            pickle.dump(df, pckl)
            pckl.close()

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('sharpe: time_finish= {0}, duration= {1}'.format(time_start, time_duration))

        return df


    def adi_calc(self, *args):
        df = args[0]
        new_clmn_names = args[1]
        tmprd_adi = args[2]

        result = (new_clmn_names[0], self.adi(df, tmprd_adi, 'open', 'high', 'low', 'close', 'volume_aver'))
        print('%s says result for args %s = %s' % (mp.current_process().name, args[1], result))
        return args[1], result


    def run_adi_calc(self, df, dump_pickle=True):
        time_start = dt.datetime.now()
        print('adi: time_start= {}'.format(time_start))
        print('cpu_count() = %d\n' % mp.cpu_count())

        df['volume_aver'] = (df['volume_ask'] + df['volume_bid']) / 2

        #--- Create pool
        print('Creating pool with %d processes\n' % self.num_threads)
        pool = mp.Pool(self.num_threads)
        print('pool = %s' % pool)
        print()

        TASKS = []

        for tmprd_adi in self.adi_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_adi)
            new_clmn_names = [self.adi_base_clmn_name + postfix]
            TASKS.append((df, new_clmn_names, tmprd_adi))

        #print('TASKS:\n', TASKS)
        results = [pool.apply_async(self.sharpe_calc, t) for t in TASKS]

        for r in results:
            res = r.get()
            print('\t', res)
            print('type(res)= {0}, res= \n{1}'.format(type(res), res))
            clmn_name = res[0][0]
            print('res[1][1]: \n', res[1][1])
            clmn_data = res[1][1].values
            print('clmn_name= {0}, clmn_data:\n {1}'.format(clmn_name, clmn_data))
            df[clmn_name] = clmn_data

        if dump_pickle:
            pckl = open(self.data_pickle_path_for_dump, "wb")
            pickle.dump(df, pckl)
            pckl.close()

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('adi: time_finish= {0}, duration= {1}'.format(time_start, time_duration))

        return df


    def bb_calc(self, df):
        for tmprd_bb in self.bb_tmprd_arr:
            for d in self.bb_d_arr:
                nbdevup, nbdevdn = d, d
                postfix = '_' + processing.digit_to_text(tmprd_bb) + '_' + processing.digit_to_text(d)
                new_clmn_names = [name + postfix for name in self.bb_base_clmn_names]
                print('new_clmn_names: ', new_clmn_names)
                inp = {'df': df, 'function': self.bb_fin_func, 'add_columns': new_clmn_names,
                       'real': df['open'].values, 'timeperiod': tmprd_bb, 'nbdevup': nbdevup, 'nbdevdn': nbdevdn,
                       'matype': 0}
                df = processing.features_add(**inp)

                inp = (df, 'open', new_clmn_names[0], 'ubb_open' + postfix)
                df = processing.clmn_compare(*inp)
                inp = (df, 'open', new_clmn_names[2], 'lbb_open' + postfix)
                df = processing.clmn_compare(*inp)

        return df


    def so_calc(self, df):
        for param_so in self.so_param_arr:
            postfix = '_' + processing.digit_to_text(param_so[0]) + '_' + processing.digit_to_text(param_so[1])
            new_clmn_names = [name + postfix for name in self.so_base_clmn_names]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.so_fin_func, 'add_columns': new_clmn_names, 'shift': 1,
                   'high': df['high'].values, 'low': df['low'].values, 'close': df['close'].values,
                   'fastk_period': param_so[0], 'slowk_period': int(param_so[0] * self.so_k_slowk_period),
                   'slowd_period': param_so[1]}
            df = processing.features_add(**inp)

            inp = (df, new_clmn_names[0], new_clmn_names[1], 'so_k_d' + postfix)
            df = processing.clmn_compare(*inp)

        return df


    def rsi_calc(self, df):
        for tmprd_rsi in self.rsi_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_rsi)
            new_clmn_names = [self.rsi_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.rsi_fin_func, 'add_columns': new_clmn_names, 'shift': 0,
                   'real': df['open'].values, 'timeperiod': tmprd_rsi}
            df = processing.features_add(**inp)

        return df


    def cci_calc(self, df):
        for tmprd_cci in self.cci_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_cci)
            new_clmn_names = [self.cci_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.cci_fin_func, 'add_columns': new_clmn_names, 'shift': 1,
                   'high': df['high'].values, 'low': df['low'].values, 'close': df['close'].values,
                   'timeperiod': tmprd_cci}
            df = processing.features_add(**inp)

        return df


    def adx_calc(self, df):
        for tmprd_adx in self.adx_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_adx)
            new_clmn_names = [self.adx_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.adx_fin_func, 'add_columns': new_clmn_names, 'shift': 1,
                   'high': df['high'].values, 'low': df['low'].values, 'close': df['close'].values,
                   'timeperiod': tmprd_adx}
            df = processing.features_add(**inp)

        return df


    def dema_calc(self, df):
        for tmprd_dema in self.dema_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_dema)
            new_clmn_names = [self.dema_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.dema_fin_func, 'add_columns': new_clmn_names, 'shift': 0,
                   'real': df['open'].values, 'timeperiod': tmprd_dema}
            df = processing.features_add(**inp)

            inp = (df, 'open', new_clmn_names[0], 'dema_open' + postfix)
            df = processing.clmn_compare(*inp)

        return df


    def tema_calc(self, df):
        for tmprd_tema in self.tema_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_tema)
            new_clmn_names = [self.tema_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.tema_fin_func, 'add_columns': new_clmn_names, 'shift': 0,
                   'real': df['open'].values, 'timeperiod': tmprd_tema}
            df = processing.features_add(**inp)

            inp = (df, 'open', new_clmn_names[0], 'tema_open' + postfix)
            df = processing.clmn_compare(*inp)

        return df


    def macd_calc(self, df):
        for params_macd in self.macd_params_arr:
            postfix = '_' + processing.digit_to_text(params_macd[0]) \
                      + '_' + processing.digit_to_text(params_macd[1]) + '_' + processing.digit_to_text(params_macd[2])
            new_clmn_names = [name + postfix for name in self.macd_base_clmn_names]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.macd_fin_func, 'add_columns': new_clmn_names, 'shift': 0,
                   'real': df['open'].values, 'fastperiod': params_macd[0], 'slowperiod': params_macd[1],
                   'signalperiod': params_macd[2]}
            df = processing.features_add(**inp)

        return df


    def mfi_calc(self, df):
        df['volume_aver'] = (df['volume_ask'] + df['volume_bid']) / 2
        for tmprd_mfi in self.mfi_tmprd_arr:
            postfix = '_' + processing.digit_to_text(tmprd_mfi)
            new_clmn_names = [self.mfi_base_clmn_name + postfix]
            print('new_clmn_names: ', new_clmn_names)
            inp = {'df': df, 'function': self.mfi_fin_func, 'add_columns': new_clmn_names, 'shift': 1,
                   'high': df['high'].values, 'low': df['low'].values, 'close': df['close'].values,
                   'volume': df['volume_aver'].values, 'timeperiod': tmprd_mfi}
            df = processing.features_add(**inp)

        return df


    def lr_calc(self, df):
        for tmprd_lr in self.lr_tmprd_arr:
            for mult_lr in self.lr_mult_arr:
                postfix = '_' + processing.digit_to_text(tmprd_lr) + '_' + processing.digit_to_text(mult_lr)
                new_clmn_names = [name + postfix for name in self.lr_base_clmn_names]
                print('new_clmn_names: ', new_clmn_names)
                tmprd_lr_1 = tmprd_lr
                df[new_clmn_names[0]] = self.lr_fin_func(df['open'].values, timeperiod=tmprd_lr_1)

                tmprd_lr_2 = int(tmprd_lr_1 * mult_lr)
                df[new_clmn_names[1]] = self.lr_fin_func(df['open'].values, timeperiod=tmprd_lr_2)

                inp = (df, new_clmn_names[0], new_clmn_names[1], 'lr_cmpr' + postfix)
                df = processing.clmn_compare(*inp)

        return df

    def execute(self):
        #--- dataframe load
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        pckl = open(self.data_pickle_path, "rb")
        data = pickle.load(pckl)
        pckl.close()
        #---

        #---
        print("Hurst calculation...")
        data = self.run_hurst_calc(data, True)
        #---
        data = self.ema_calc(data)
        data = self.run_sharpe_calc(data)
        data = self.run_adi_calc(data)
        data = self.bb_calc(data)
        data = self.so_calc(data)
        data = self.rsi_calc(data)
        data = self.cci_calc(data)
        data = self.adx_calc(data)
        data = self.dema_calc(data)
        data = self.tema_calc(data)
        data = self.macd_calc(data)
        data = self.mfi_calc(data)
        data = self.lr_calc(data)
        # #---

        if self.dump_pickle:
            pckl = open(self.data_pickle_path_for_dump, "wb")
            pickle.dump(data, pckl)
            pckl.close()

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_start, time_duration))


if __name__ == '__main__':
    req = FeaturesCalcClass()
    req.execute()