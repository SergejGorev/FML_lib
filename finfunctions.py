import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle


def test_func(*args):
    print('type(args)= ', type(args))
    print('type(args[0])= ', type(args[0]))
    print('np.mean(args[0])= ', np.mean(args[0]))
    print(args)
    return 1.

# Hurst exponention function
def hurst_ernie_chan(p, lags):
    variancetau = [];
    tau = []

    for lag in lags:
        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    # print tau
    # print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau), np.log10(variancetau), 1)

    hurst = m[0] / 2

    return hurst


def check_pred_r(x):
    """
    Функция для вычисления фин. результата по бару на основании предсказан. направления и
    результатов исторического тестирования в торговом терминале
    """
    pred_dir_val, pl_buy, pl_sell = x[0], x[1], x[2]
    res = 0.0
    if (pred_dir_val == 1):
        res = pl_buy
    elif (pred_dir_val == -1):
        res = pl_sell
    return res


def sharpe_ratio(datetime_values, return_values):
    """
    Calculate Sharpe Ratio. Classical algorythm with everyday return.

    Returns Sharpe Ratio value.

    Parameters
    ----------
    datetime_values : datetime array, datetime ndarray
        Array with datetime values.
    return_values :  float array, float ndarray
        Array with return values.

    Returns
    -------
    sharpe_ratio : float
        Sharpe ratio value.

    """
    return_df_ = pd.DataFrame(list(zip(datetime_values, return_values)), columns=['date', 'return'])
    return_df_['dt_date'] = return_df_['date'].dt.date
    # print(return_df_)
    day_returns = return_df_.loc[:, ['dt_date', 'return']].groupby('dt_date').sum()
    # print(day_returns)
    day_returns_values = day_returns.values
    # print(day_returns_values)
    sqrt_ = np.sqrt(len(day_returns_values))
    mean_ = np.mean(day_returns_values)
    std_ = np.std(day_returns_values)  # np.max(np.std(day_returns_values), 0.01)
    # print("sqrt_= {0:.4f}, mean_= {1:.4f}, std_= {2:.4f}".format(sqrt_, mean_, std_))
    res = sqrt_ * mean_ / std_
    return res


def pred_fin_res(y_pred, label_buy, label_sell, profit_value, loss_value):
    """
    Calculate return and Sharpe Ratio (classical algorythm with everyday return).

    Returns Return and Sharpe Ratio values.

    Parameters
    ----------
    y_pred : pandas.Series
        Array with predicted labels {-1, 0, 1}. The index of the series is the date.
    label_buy : pandas.Series
        Array with buy labels {-1, 1}. -1 means that stop loss is triggered. +1 means that take profit is triggered.
    label_sell : pandas.Series
        Array with buy labels {-1, 1}. -1 means that stop loss is triggered. +1 means that take profit is triggered.
    profit_value : float value.
        The size of the profit in the case of triggering the take profit.
    loss_value : float value.
        The size of the loss in the case of triggering the stop loss.

    Returns
    -------
    sharpe_ratio : float
        Sharpe ratio value.

    """
    return_df_ = pd.DataFrame(y_pred, columns=['pred'])
    return_df_['date'] = return_df_.index
    return_df_['dt_date'] = return_df_['date'].dt.date
    return_df_['label_buy'] = label_buy
    return_df_['label_sell'] = label_sell

    def return_func(row):
        res = 0.
        if row['pred'] == 1:
            if row['label_buy'] == 1:
                res = profit_value
            else:
                res = loss_value
        elif row['pred'] == -1:
            if row['label_sell'] == 1:
                res = profit_value
            else:
                res = loss_value
        return res

    return_df_['return'] = return_df_.apply(func=return_func, axis=1)
    # print(return_df_)
    #---
    # сохранение дампа датафрейма в тестовых целях
    pckl = open("return_df_.pickle", "wb")
    pickle.dump(return_df_, pckl)
    pckl.close()
    #---
    return_res = return_df_['return'].sum()
    day_returns = return_df_.loc[:, ['dt_date', 'return']].groupby('dt_date').sum()
    # print(day_returns)
    day_returns_values = day_returns.values
    # print(day_returns_values)
    sqrt_ = np.sqrt(len(day_returns_values))
    mean_ = np.mean(day_returns_values)
    std_ = np.std(day_returns_values)  # np.max(np.std(day_returns_values), 0.01)
    # print("sqrt_= {0:.4f}, mean_= {1:.4f}, std_= {2:.4f}".format(sqrt_, mean_, std_))
    sr_res = sqrt_ * mean_ / std_
    return return_res, sr_res


def sharpe_ratio_feature(*return_values):
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
    #print('return_values: ', return_values)
    if str(type(return_values))=="<class 'tuple'>":
        values = return_values[1]
    else:
        values = return_values
    # print('values: ', values)

    mean_ = np.mean(values)
    std_ = np.std(values)
    # print("mean_= {1:.4f}, std_= {2:.4f}".format(mean_, std_))
    if std_ != 0.:
        res = mean_ / std_
    else:
        res = 0.
    return res


def adi(df, window, price_open_clmn, price_high_clmn, price_low_clmn, price_close_clmn, vol_clmn):
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
    func_clv = lambda x: ((x['open'] - x['low']) - (x['high'] - x['open']))/(x['high'] - x['low']) \
        if (x['high'] - x['low'])!=0. else 0.
    df_calc['clv'] = df_calc.apply(func_clv, axis=1)

    df_calc['res'] = df_calc['clv']*df_calc['vol_roll']
    return df_calc['res'].values


def getTEvents(gRaw, h):
    """
    THE SYMMETRIC CUSUM FILTER
    :param gRaw:
    :param h:
    :return:
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i in diff.index[1:]:
        sPos, sNeg = max(0, sPos + diff.loc[i]), min(0, sNeg + diff.loc[i])
        if sNeg < -h:
            sNeg = 0;
            tEvents.append(i)
        elif sPos > h:
            sPos = 0;
            tEvents.append(i)
    return pd.DatetimeIndex(tEvents)


def getTEvents_num(gRaw, h):
    """
    THE SYMMETRIC CUSUM FILTER (numeric)
    :param gRaw:
    :param h:
    :return:
    """
    tEvents, sPos, sNeg = [], 0, 0
    diff = gRaw.diff()
    for i, x in enumerate(diff.index[1:]):
        sPos, sNeg = max(0, sPos + diff.loc[x]), min(0, sNeg + diff.loc[x])
        # print("i= {0}, x= {1}, sPos= {2}, sNeg= {3}".format(i, x, sPos, sNeg))
        if sNeg < -h:
            sNeg = 0;
            tEvents.append(i + 1)
        elif sPos > h:
            sPos = 0;
            tEvents.append(i + 1)
    return tEvents


def getWeights(d, size):
    """
    WEIGHTING FUNCTION
    # thres>0 drops insignificant weights
    :param d:
    :param size:
    :return:
    """
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def plotWeights(dRange, nPlots, size):
    w = pd.DataFrame()
    for d in np.linspace(dRange[0], dRange[1], nPlots):
        w_ = getWeights(d, size=size)
        w_ = pd.DataFrame(w_, index=range(w_.shape[0])[::-1], columns=[d])
        w = w.join(w_, how='outer')
    ax = w.plot()
    ax.legend(loc='upper left');
    plt.show()
    return


# THE NEW FIXED-WIDTH WINDOW FRACDIFF METHOD
def fracDiff_FFD(series, d, part=0.05):  # , thres=1e-5
    """
    Constant width window (new solution)
    Note 1: thres determines the cut-off weight for the window
    Note 2: d can be any positive fractional, not necessarily bounded [0,1].
    """
    # 1) Compute weights for the longest series
    w = getWeights(d, int(series.shape[0] * part))  # w = getWeights_FFD(d, thres)
    # print('w:')
    # print(w)
    width = len(w) - 1
    # print("\nwidth= {}".format(width))
    # 2) Apply weights to values
    df = {}
    for name in series.columns:
        seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
        for iloc1 in range(width, seriesF.shape[0]):
            loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1, name]): continue  # exclude NAs
            df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]
        df[name] = df_.copy(deep=True)
    df = pd.concat(df, axis=1)
    return df