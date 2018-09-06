import numpy as np
import pandas as pd

def features_add(df, function, add_columns, shift=0, **params):
    """
    Функция для добавления признаков к датафрейму.

    df - датафрейм,
    function - функция для формирования признаков,
    add_columns - наименования добавляемых столбцов,

    Остальные параметры передаются в функцию именованными в соответствие с требованиями даной функции.
    Например, для tl.BBANDS: real=data['open'].values, timeperiod=int(4*24*0.75), nbdevup=1.5, nbdevdn=1.5, matype=0
    """
    # print('function= ', function)
    # print('\nparams: ', params)
    res = function(**params)
    # print('res:\n', res)

    if shift>0:
        if len(add_columns) == 1:
            df[add_columns[0]] = res
            df[add_columns[0]] = df[add_columns[0]].shift(shift)
        else:
            #print('\nres: ', res)
            #print('len(res): ', len(res))
            for i, clmn in enumerate(add_columns):
                df[clmn] = res[i]
                df[clmn] = df[clmn].shift(shift)
    else:
        if len(add_columns) == 1:
            df[add_columns[0]] = res
        else:
            #print('\nres: ', res)
            #print('len(res): ', len(res))
            for i, clmn in enumerate(add_columns):
                df[clmn] = res[i]

    return df


def compare_safe(value_1, value_2):
    """
    Функция для безопасного сравнения значений (обрабатывается случай делителя равного нулю).
    :param value_1: первое значение
    :param value_2: второе значение
    :return: результат сравнения
    """
    if value_1==0.:
        res = 1e15
    else:
        res = (value_2 - value_1)/value_1
    return res


def clmn_compare(df, clmn_1, clmn_2, new_clmn_name):
    """
    :param df: датафрейм
    :param clmn_1: наименование столбца, по отношению к которому осуществляется сравнение
    :param clmn_2: наименование столбца, который сравнивается
    :param new_clmn_name: наименование нового столбца с результатом сравения
    :return: датафрейм
    """
    #compare_func = lambda x: (x[clmn_2] - x[clmn_1]) / x[clmn_1]
    #df[new_clmn_name] = df.apply(compare_func, axis=1)
    print('new_clmn_name= ', new_clmn_name)
    # df['diff_clmn'] = df[clmn_2] - df[clmn_1]
    # cmpr_func = lambda x: x['diff_clmn']/x[clmn_1] if (x[clmn_1] != 0.) else np.nan
    # df[new_clmn_name] = df.apply(cmpr_func, axis=1)
    # df.drop(columns='diff_clmn', inplace=True)
    df[new_clmn_name] = (df[clmn_2] - df[clmn_1])/df[clmn_1]
    df[new_clmn_name].replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def digit_to_text(digit):
    """
    Функция для преобразования чисел в текст.
    Заменяет точку "i".
    :param digit: число
    :return: текст
    """
    digit_str = str(digit)
    digit_str = digit_str.replace('.', 'i')
    return digit_str

