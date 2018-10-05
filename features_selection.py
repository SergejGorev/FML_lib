import finfunctions, processing
import pandas as pd
import numpy as np
import datetime as dt
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from scipy import stats
import copy
import warnings
import os
from itertools import combinations
from matplotlib import pyplot as plt


class FeaturesSelectionClass:
    n_loops = 2500  # количество циклов
    features_part = 0.06  # доля признаков, участвующих в тестировании в каждом проходе
    folder_name = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD"
    data_pickle_file_name = "eurusd_5_v1.4.pickle"
    label_pickle_file_name = "eurusd_5_v1_lbl_0i003_1i0_0i5.pickle"

    postfix = '_0i003_1i0_0i5'
    version = 'v1.0'
    target_clmn_prefix = 'target_label'
    profit_value = 28
    loss_value = -15 #-25
    dump_pickle = True # dump data pickle
    purged_period = 3  #(days)


    f_i_pickle_prefix = "feat_imp"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\feat_imp_20180905_3.pickle"
    ftrs_major_arr_pickle_prefix = "ftrs_major_arr"
    ftrs_minor_arr_pickle_prefix = "ftrs_minor_arr"

    feat_imp_filenames_arr = ['feat_imp_0i002_1i0_1i0_v1.0.pickle', 'feat_imp_0i002_1i0_2i0_v1.0.pickle',
                              'feat_imp_0i003_1i0_1i0_v1.0.pickle', 'feat_imp_0i0015_1i0_1i0_v1.0.pickle',
                              'feat_imp_0i0025_1i0_1i0_v1.0.pickle']  # , 'feat_imp_0i0035_1i0_1i0_v1.0.pickle'
    clmn_names_arr = ['0i002_1i0_1i0_v1.0', '0i002_1i0_2i0_v1.0',
                      '0i003_1i0_1i0_v1.0', '0i0015_1i0_1i0_v1.0',
                      '0i0025_1i0_1i0_v1.0']  # , '0i0035_1i0_1i0_v1.0'

    price_step = 0.001
    start_date = dt.datetime(2008, 1, 1, 0, 0)
    finish_date = dt.datetime(2018, 6, 15, 23, 59)
    dt0 = [dt.datetime(2009, 7, 1), dt.datetime(2011, 7, 1), dt.datetime(2013, 7, 1), dt.datetime(2015, 7, 1),
           dt.datetime(2017, 7, 1)]
    dt1 = [dt.datetime(2010, 6, 15), dt.datetime(2012, 6, 15), dt.datetime(2014, 6, 15), dt.datetime(2016, 6, 15),
           dt.datetime(2018, 6, 15)]

    data_pickle_path = ""
    label_pickle_path = ""
    f_i_pickle_path = ""
    ftrs_major_arr_pickle_path = ""
    ftrs_minor_arr_pickle_path = ""
    target_clmn = ""
    testTimes = None
    data_for_ml = None
    features_arr = []
    n_features = 0
    last_clmn = 0
    # ---


    def __init__(self):
        self.pickle_postfix = self.postfix + '_' + self.version + '.pickle'
        self.data_pickle_path = self.folder_name + os.sep + self.data_pickle_file_name  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.4.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.4.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.4.pickle"
        self.label_pickle_path = self.folder_name + os.sep + self.label_pickle_file_name  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"

        self.target_clmn =  self.target_clmn_prefix + self.postfix  # 'target_label_0i0025_1i0_1i0'
        self.f_i_pickle_path = self.folder_name + os.sep + self.f_i_pickle_prefix + self.pickle_postfix
        self.ftrs_major_arr_pickle_path = self.folder_name + os.sep + self.ftrs_major_arr_pickle_prefix + self.pickle_postfix
        self.ftrs_minor_arr_pickle_path = self.folder_name + os.sep + self.ftrs_minor_arr_pickle_prefix + self.pickle_postfix
        # print('data_pickle_path= '+self.data_pickle_path)
        # print('label_pickle_path= ' + self.label_pickle_path)
        # print('target_clmn= ' + self.target_clmn)
        # print('f_i_pickle_path= ' + self.f_i_pickle_path)
        # print('ftrs_major_arr_pickle_path= ' + self.ftrs_major_arr_pickle_path)
        # print('ftrs_minor_arr_pickle_path= ' + self.ftrs_minor_arr_pickle_path)


    @staticmethod
    def select_data_for_ml(data_lbl, price_step, target_clmn):
        """
        The function selects data from the whole dataset with defined price step.
        :param data_lbl: pandas dataframe.
        :param price_step: float.
        :param target_clmn: string.
        :return: pandas dataframe with selected data.
        """
        data_sel_idx = finfunctions.getTEvents(data_lbl.open_ask, price_step)
        print('len(data_sel_idx)= {}'.format(len(data_sel_idx)))
        data_for_ml = data_lbl.loc[data_sel_idx, :]
        # !!! убираем из обучения метки с нулями !!!
        data_for_ml = data_for_ml.loc[data_for_ml[target_clmn] != 0, :]
        data_for_ml_group = data_for_ml.groupby(by=target_clmn)[target_clmn].count()
        print('\ndata_samples_group:\n', data_for_ml_group)
        return data_for_ml

    @staticmethod
    def set_testTimes(dt0, dt1):
        """
        The function creates array with the boundaries of the time periods.
        :param dt0: datatime array.
        :param dt1: datatime array.
        :return: pandas series with the boundaries of the time periods.
        """
        # формирование тестовых периодов
        testTimes = pd.Series(dt1, index=dt0)
        print('\ntestTimes:\n{}'.format(testTimes))
        return testTimes

    @staticmethod
    def setting_features_array(data_lbl, last_clmn):
        """
        The function selects features from dataset columns.
        :param data_lbl: pandas dataframe.
        :param last_clmn: int.
        :return: array with features names.
        """
        features_arr = []
        features_arr.append(data_lbl.columns[17])
        features_arr.append(data_lbl.columns[16])
        features_arr.extend(data_lbl.columns[24:last_clmn])
        # features_arr.remove('return')
        print('features_arr:\n', features_arr)
        return features_arr


    @staticmethod
    def cross_val_xgb(df_train_, df_test_, testTimes_, features_for_ml_, target_clmn_, \
                      max_depth_=3, n_estimators_=100, n_jobs_=-1, calc_fin_stats=True, df_lbl=None,
                      profit_value=0., loss_value=0., print_log=True):
        """
        The function implements ML-model cross validation and returns evaluation statistics.
        :param df_train_: pandas dataframe.
        :param df_test_: pandas dataframe.
        :param testTimes_: pandas series.
        :param features_for_ml_: string array.
        :param target_clmn_: string.
        :param max_depth_: integer.
        :param n_estimators_: integer.
        :param n_jobs_: integer.
        :param calc_fin_stats: boolean.
        :param df_lbl: pandas dataframe.
        :param profit_value: float.
        :param loss_value: float.
        :param print_log: boolean.
        :return: dictionary with evaluation statistics.
        """
        res_dict = {}
        acc_arr = []
        f1_arr = []
        conf_matrix_arr = []
        ftrs_imp_arr = []
        if calc_fin_stats:
            rtrn_arr = []
            sr_arr = []
        test_periods_count = len(testTimes_)
        if print_log: print('test_periods_count= ', test_periods_count)

        warnings.filterwarnings(action='ignore')
        for test in zip(testTimes_.index, testTimes_):
            if print_log: print('\ntest= {0}'.format(test))
            clf = XGBClassifier(max_depth=max_depth_, n_estimators=n_estimators_, n_jobs=n_jobs_)

            df_test_iter = df_test_
            if ((calc_fin_stats==True) & (df_lbl is not None)):
                df_test_iter['label_buy'] = df_lbl['label_buy']
                df_test_iter['label_sell'] = df_lbl['label_sell']
            df_test_iter = df_test_iter.loc[(test[0] <= df_test_.index) & (df_test_.index <= test[1]), :]
            df_train_iter = df_train_.loc[df_train_.index.difference(df_test_iter.index)]

            X_train_iter = df_train_iter.loc[:, features_for_ml_]
            y_train_iter = df_train_iter.loc[:, target_clmn_]
            X_test_iter = df_test_iter.loc[:, features_for_ml_]
            y_test_iter = df_test_iter.loc[:, target_clmn_]

            clf.fit(X_train_iter, y_train_iter)
            y_pred_iter = clf.predict(X_test_iter)

            acc = accuracy_score(y_test_iter, y_pred_iter)
            if print_log: print('accuracy= {0:.5f}'.format(acc))
            acc_arr.append(acc)
            f1_scr = f1_score(y_test_iter, y_pred_iter, average='weighted')
            if print_log: print('f1_score= {0:.5f}'.format(f1_scr))
            f1_arr.append(f1_scr)
            conf_matrix = confusion_matrix(y_test_iter, y_pred_iter)
            if print_log: print('\nconf_matrix:\n{}'.format(conf_matrix))
            conf_matrix_arr.append(conf_matrix)

            if print_log: print("\nfeature_importances:")
            f_i = list(zip(features_for_ml_, clf.feature_importances_))
            dtype = [('feature', 'S30'), ('importance', float)]
            f_i_nd = np.array(f_i, dtype=dtype)
            f_i_sort = np.sort(f_i_nd, order='feature')  # f_i_sort = np.sort(f_i_nd, order='importance')[::-1]
            f_i_arr = f_i_sort.tolist()
            ftrs_imp_arr.append(f_i_arr)
            if print_log:
                for i, imp in enumerate(f_i_arr, 1):
                    print('{0}. {1:<30} {2:.5f}'.format(i, str(imp[0]).replace("b\'", "").replace("\'", ""), imp[1]))
            #--- financial statistics calculation
            y_pred_series = pd.Series(data=y_pred_iter, index=df_test_iter.index)
            fin_res = finfunctions.pred_fin_res(y_pred=y_pred_series, label_buy=df_test_iter['label_buy'],
                                                label_sell=df_test_iter['label_sell'], profit_value=profit_value,
                                                loss_value=loss_value)
            if calc_fin_stats:
                rtrn_arr.append(fin_res[0])
                sr_arr.append(fin_res[1])
                if print_log:
                    print('return= {0:.2f}, SR= {1:.4f}'.format(fin_res[0], fin_res[1]))

            #---
        if print_log: print('\nacc_arr= ', acc_arr)
        acc_arr_mean = np.mean(acc_arr)
        acc_arr_std = np.std(acc_arr)
        if print_log: print('acc_arr_mean= {0:.5f}, acc_arr_std= {1:.5f}'.format(acc_arr_mean, acc_arr_std))
        res_dict['acc_score_mean'] = acc_arr_mean
        res_dict['acc_score_std'] = acc_arr_std
        res_dict['acc_score_arr'] = str(acc_arr)
        if print_log: print('\nf1_arr= ', f1_arr)
        f1_arr_mean = np.mean(f1_arr)
        f1_arr_std = np.std(f1_arr)
        if print_log: print('f1_arr_mean= {0:.5f}, f1_arr_std= {1:.5f}'.format(f1_arr_mean, f1_arr_std))
        res_dict['f1_score_mean'] = f1_arr_mean
        res_dict['f1_score_std'] = f1_arr_std
        res_dict['f1_score_arr'] = str(f1_arr)
        # ---
        res_dict['conf_matrix_arr'] = str(conf_matrix_arr)
        # ---
        if calc_fin_stats:
            if print_log: print('\nrtrn_arr= ', rtrn_arr)
            rtrn_arr_mean = np.mean(rtrn_arr)
            rtrn_arr_std = np.std(rtrn_arr)
            if print_log: print('rtrn_arr_mean= {0:.2f}, rtrn_arr_std= {1:.2f}'.format(rtrn_arr_mean, rtrn_arr_std))
            res_dict['return_mean'] = rtrn_arr_mean
            res_dict['return_std'] = rtrn_arr_std
            res_dict['return_arr'] = str(rtrn_arr)

            if print_log: print('\nsharpe_arr= ', sr_arr)
            sr_arr_mean = np.mean(sr_arr)
            sr_arr_std = np.std(sr_arr)
            if print_log: print('sharpe_mean= {0:.5f}, sharpe_std= {1:.5f}'.format(sr_arr_mean, sr_arr_std))
            res_dict['sharpe_mean'] = sr_arr_mean
            res_dict['sharpe_std'] = sr_arr_mean
            res_dict['sharpe_arr'] = str(sr_arr)
        #---
        # print('\nftrs_imp_arr:\n', ftrs_imp_arr)
        res_dict['ftrs_imp_arr'] = ftrs_imp_arr

        if print_log: print()
        features_imp_dict = {}
        for i in range(len(ftrs_imp_arr[0])):
            feature_name = ftrs_imp_arr[0][i][0]
            feature_name = str(feature_name).replace("b'", "").replace("'", "")
            feature_arr = [ftrs_imp_arr[my_iter][i][1] for my_iter in range(test_periods_count)]
            feature_arr_mean = np.mean(feature_arr)
            # if print_log: print('feature_name= {0}, feature_arr= {1}, mean= {2:.5f}'.format(feature_name,
            #                                                                   feature_arr, feature_arr_mean))
            features_imp_dict[feature_name] = feature_arr_mean
        print('features_imp_dict:\n', features_imp_dict)
        res_dict['features_imp_dict'] = features_imp_dict
        warnings.filterwarnings(action='default')
        return res_dict


    @staticmethod
    def cpcv_xgb(df_train, df_test, df_lbl, features_for_ml, target_clmn,
                 start_date=None, finish_date=None,
                 purged_period=3, cpcv_n=6, cpcv_k=2, max_depth=3, n_estimators=100, n_jobs=-1,
                 profit_value=0., loss_value=0.,
                 use_pred_proba=False, pred_proba_threshold=.505,
                 save_paths_return=False, pickle_path='path_return_df.pickle',
                 save_picture=False, picture_path='CPCV_testing_return.jpg',
                 pred_values_series_aggregation=False,
                 dump_model=False, print_log=True):
        """
        The function implements ML-model combinatorial purged cross validation and returns evaluation statistics.
        :param df_train_: pandas dataframe.
        :param df_test_: pandas dataframe.
        :param features_for_ml_: string array.
        :param target_clmn_: string.
        :param purged_period: integer.
            Purged period (days).
        :param cpcv_n: integer.
        :param cpcv_k: integer.
        :param max_depth_: integer.
        :param n_estimators_: integer.
        :param n_jobs_: integer.
        :param calc_fin_stats: boolean.
        :param df_lbl: pandas dataframe.
        :param profit_value: float.
        :param loss_value: float.
        :param save_paths_return: boolean.
        :param print_log: boolean.
        :return: dictionary with evaluation statistics.
        """
        res_dict = {}
        rtrn_arr = []
        sr_arr = []
        #--- test parts combinations
        test_periods_arr = list(combinations(range(cpcv_n), cpcv_k))
        test_periods_count = len(test_periods_arr)
        paths_count = int(test_periods_count*cpcv_k/cpcv_n)
        # if print_log: print(test_periods_arr)
        if print_log: print('test_periods_count= {0}, paths_count= {1}'.format(test_periods_count, paths_count))
        calc_arr = [[] for i in range(cpcv_n)]  # array for return series
        if pred_values_series_aggregation:
            pred_arr = [[] for i in range(cpcv_n)]  # array for predicted values series
            if use_pred_proba: pred_proba_arr = [[] for i in range(cpcv_n)]
        #---
        #--- datasets cutting
        if start_date is not None:
            if print_log: print('start_date= {0}'.format(start_date))
            df_train = df_train.loc[start_date<=df_train.index, :]
            df_test = df_test.loc[start_date <= df_test.index, :]
            df_lbl = df_lbl.loc[start_date <= df_lbl.index, :]

        if finish_date is not None:
            if print_log: print('finish_date= {0}'.format(finish_date))
            df_train = df_train.loc[df_train.index<=finish_date, :]
            df_test = df_test.loc[df_test.index <= finish_date, :]
            df_lbl = df_lbl.loc[df_lbl.index <= finish_date, :]
        #---
        df_test['label_buy'] = df_lbl['label_buy']
        df_test['label_sell'] = df_lbl['label_sell']
        #--- periods split
        total_ix = len(df_test.index)
        part_len = int(total_ix / cpcv_n)
        if print_log: print('total_ix= {0}, part_len= {1}'.format(total_ix, part_len))

        curr_first_ix = 0
        curr_last_ix = part_len
        periods = []
        for i in range(cpcv_n):
            # print('i= {0}'.format(i))
            if i < cpcv_n - 1:
                periods.append([df_test.index[curr_first_ix], df_test.index[curr_last_ix]])
                curr_first_ix = curr_last_ix + 1
                curr_last_ix = curr_first_ix + part_len
            else:
                periods.append([df_test.index[curr_first_ix], df_test.index[-1]])
        # print('periods:\n', periods)
        if print_log:
            for i, tm_arr in enumerate(periods, 0):
                print(i, ': ', tm_arr)
        #---
        warnings.filterwarnings(action='ignore')
        time_start = dt.datetime.now()
        if print_log: print('time_start= {}'.format(time_start))
        for num, test_periods in enumerate(test_periods_arr):
            if print_log: print('num= ', num)
            train_periods = list(range(cpcv_n))
            for i in test_periods: train_periods.remove(i)
            if print_log: print('test_periods : ', test_periods)
            if print_log: print('train_periods: ', train_periods)
            #--- new train periods creating
            last_period = cpcv_n - 1
            new_periods = copy.deepcopy(periods)
            if print_log: print('\nTime periods editing (purge and embargo applying):')
            if print_log: print('-------------------------------------------------------------------------')
            for prd in test_periods:
                print('prd= ', prd)
                if prd == 0:
                    purge = False
                    embargo = True
                elif prd == last_period:
                    purge = True
                    embargo = False
                else:
                    purge = True
                    embargo = True

                if purge:
                    if prd - 1 not in test_periods:
                        prev_finish_date = new_periods[prd - 1][1]
                        finish_date = prev_finish_date - np.timedelta64(
                            int(np.around(purged_period + .49, decimals=0)), 'D')
                        if print_log: print('for {0}: prev_finish_date= {1}, finish_date= {2}'.format(prd - 1, prev_finish_date,
                                                                                        finish_date))
                        new_periods[prd - 1][1] = finish_date
                if embargo:
                    if prd + 1 not in test_periods:
                        prev_start_date = new_periods[prd + 1][0]
                        start_date = prev_start_date + np.timedelta64(int(np.around(purged_period + .49, decimals=0)),
                                                                      'D')
                        if print_log: print('for {0}: prev_start_date= {1}, start_date= {2}'.format(prd + 1, prev_start_date,
                                                                                      start_date))
                        new_periods[prd + 1][0] = start_date
                if print_log: print('-------------------------------------------------------------------------')
            if print_log:
                print('periods (embargoed and purged):')
                for i, tm_arr in enumerate(new_periods, 0):
                    print(i, ': ', tm_arr)
            #---
            #--- train dataframe slicing
            if print_log: print('\nTrain dataframe generation:')
            if print_log: print('-------------------------------------------------------------------------')
            df_train_iter = pd.DataFrame()
            for prd in train_periods:
                start_date = new_periods[prd][0]
                finish_date = new_periods[prd][1]
                if print_log: print('prd= {0}: start_date= {1}, finish_date= {2}'.format(prd, start_date, finish_date))
                df_train_part = df_train.loc[(start_date <= df_train.index) & (df_train.index <= finish_date), :]
                # if print_log: print('df_train_part:\n', df_train_part)
                df_train_iter = pd.concat((df_train_iter, df_train_part))
                if print_log: print('-------------------------------------------------------------------------')
            # if print_log: print('df_train_iter:\n', df_train_iter)
            #---
            #---
            clf = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, n_jobs=n_jobs)
            X_train_iter = df_train_iter.loc[:, features_for_ml]
            y_train_iter = df_train_iter.loc[:, target_clmn]

            clf.fit(X_train_iter, y_train_iter)
            #--- dump model (mostly for test purposes)
            if dump_model: clf.get_booster().dump_model('xgb_model_dump'+'_'+str(cpcv_n)+'_'+str(cpcv_k)+'_'+\
                                                        str(max_depth)+'_'+str(n_estimators)+'__'+str(num)+ '.dump',
                                                        with_stats=True)
            #---
            #--- test dataframe slicing
            if print_log: print('\nTesting:')
            if print_log: print('-------------------------------------------------------------------------')
            for prd in test_periods:
                start_date = new_periods[prd][0]
                finish_date = new_periods[prd][1]
                if print_log: print('prd= {0}: start_date= {1}, finish_date= {2}'.format(prd, start_date, finish_date))
                df_test_iter = df_test.loc[(start_date <= df_test.index) & (df_test.index <= finish_date), :]
                # if print_log: print('df_test_iter:\n', df_test_iter)

                X_test_iter = df_test_iter.loc[:, features_for_ml]

                if use_pred_proba:
                    y_pred_proba_iter = clf.predict_proba(X_test_iter)
                    res_func = lambda x: -1 if x[0]>=pred_proba_threshold else (1 if x[1]>=pred_proba_threshold else 0)
                    y_pred_iter = list(map(res_func, y_pred_proba_iter))
                    # if print_log: print('y_pred_iter:\n', y_pred_iter)
                    y_pred_proba_df = pd.DataFrame(data=y_pred_proba_iter, index=df_test_iter.index,
                                                       columns=['proba_m', 'proba_p'])
                else:
                    y_pred_iter = clf.predict(X_test_iter)
                    # if print_log: print('y_pred_proba_df:\n', y_pred_proba_df)
                #--- financial statistics calculation
                y_pred_series = pd.Series(data=y_pred_iter, index=df_test_iter.index)
                fin_res = finfunctions.pred_return(y_pred=y_pred_series, label_buy=df_test_iter['label_buy'],
                                                    label_sell=df_test_iter['label_sell'], profit_value=profit_value,
                                                    loss_value=loss_value)
                if print_log: print('return= {0:.2f}'.format(fin_res.sum()))
                if pred_values_series_aggregation:
                    pred_arr[prd].append(y_pred_series)
                    if use_pred_proba: pred_proba_arr[prd].append(y_pred_proba_df)
                calc_arr[prd].append(fin_res)
                if print_log: print('-------------------------------------------------------------------------')
            if print_log:
                cur_calc_position = (num+1)/test_periods_count
                if cur_calc_position>1: cur_calc_position=1
                time_cur = dt.datetime.now()
                time_left = time_cur - time_start
                time_est =  time_left/(num+1)*(test_periods_count-(num+1))
                print('{0:.1%} is done. ETA= {1}'.format(cur_calc_position, time_est))
            if print_log: print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')

        #--- paths data aggregation
        if pred_values_series_aggregation:
            path_pred_arr = [[] for i in range(paths_count)]
            for i in range(len(pred_arr[0])):
                for j in range(len(pred_arr)):
                    path_pred_arr[i].append(pred_arr[j][i])
            if use_pred_proba:
                path_pred_proba_arr = [[] for i in range(paths_count)]
                for i in range(len(pred_proba_arr[0])):
                    for j in range(len(pred_proba_arr)):
                        path_pred_proba_arr[i].append(pred_proba_arr[j][i])
                # if print_log: print('path_pred_proba_arr:\n', path_pred_proba_arr)
        #---
        path_arr = [[] for i in range(paths_count)]
        for i in range(len(calc_arr[0])):
            for j in range(len(calc_arr)):
                path_arr[i].append(calc_arr[j][i])
        #---
        del calc_arr
        del pred_arr
        #---
        paths_pred_df = None
        paths_pred_proba_df = None
        if pred_values_series_aggregation:
            if print_log: print('path_pred_arr processing...')
            for path, path_res in enumerate(path_pred_arr):
                if print_log: print('path= {0}, len(path_res)= {1}'.format(path, len(path_res)))
                df_test_res = pd.concat(path_res)
                df_test_res = pd.DataFrame(df_test_res, columns=['y_pred'])
                if path == 0:
                    paths_pred_df = pd.DataFrame(index=df_test_res.index)
                paths_pred_df['y_pred_' + str(path)] = df_test_res['y_pred']
            if use_pred_proba:
                if print_log: print('path_pred_proba_arr processing...')
                for path, path_res in enumerate(path_pred_proba_arr):
                    if print_log: print('path= {0}, len(path_res)= {1}'.format(path, len(path_res)))
                    df_test_res = pd.concat(path_res, axis=0)
                    df_test_res = pd.DataFrame(df_test_res, columns=['proba_m', 'proba_p'])
                    # if print_log: print('df_test_res:\n', df_test_res)
                    if path==0:
                        paths_pred_proba_df = pd.DataFrame(index=df_test_res.index)
                    paths_pred_proba_df['proba_m_' + str(path)] = df_test_res['proba_m']
                    paths_pred_proba_df['proba_p_' + str(path)] = df_test_res['proba_p']
        #---
        #--- financial statistics calculation
        paths_return_df = None
        for path, path_res in enumerate(path_arr):
            if print_log: print('financial statistics calculation...')
            if print_log: print('path= {0}, len(path_res)= {1}'.format(path, len(path_res)))
            df_test_res = pd.concat(path_res)
            df_test_res = pd.DataFrame(df_test_res, columns=['return'])
            if path==0:
                paths_return_df = pd.DataFrame(index=df_test_res.index)
            paths_return_df['return_'+str(path)] = df_test_res['return']
            return_res = df_test_res['return'].sum()
            rtrn_arr.append(return_res)
            df_test_res['date'] = df_test_res.index
            df_test_res['dt_date'] = df_test_res['date'].dt.date
            day_returns = df_test_res.loc[:, ['dt_date', 'return']].groupby('dt_date').sum()
            # if print_log: print(day_returns)
            day_returns_values = day_returns.values
            # if print_log: print(day_returns_values)
            sqrt_ = np.sqrt(len(day_returns_values))
            mean_ = np.mean(day_returns_values)
            std_ = np.std(day_returns_values)
            # if print_log: print("sqrt_= {0:.4f}, mean_= {1:.4f}, std_= {2:.4f}".format(sqrt_, mean_, std_))
            sr_res = sqrt_ * mean_ / std_
            sr_arr.append(sr_res)
        #---
        rtrn_mean = np.mean(rtrn_arr)
        rtrn_std = np.std(rtrn_arr)
        sr_mean = np.mean(sr_arr)
        sr_std = np.std(sr_arr)
        res_dict['rtrn_mean'] = rtrn_mean
        res_dict['rtrn_std'] = rtrn_std
        res_dict['rtrn_arr'] = rtrn_arr
        res_dict['sr_mean'] = sr_mean
        res_dict['sr_std'] = sr_std
        res_dict['sr_arr'] = sr_arr
        res_dict['paths_return_df'] = paths_return_df
        if pred_values_series_aggregation:
            res_dict['paths_pred_df'] = paths_pred_df
            if use_pred_proba:
                res_dict['paths_pred_proba_df'] = paths_pred_proba_df
        if save_paths_return:
            with open(pickle_path, 'wb') as pckl:
                pickle.dump(paths_return_df, pckl)
        if print_log: print('res_dict:\n', res_dict)
        if save_picture:
            basic_columns = paths_return_df.columns
            for clmn in basic_columns:
                paths_return_df[clmn + '_cumsum'] = paths_return_df[clmn].cumsum()
            add_columns = list(paths_return_df.columns)
            for clmn in basic_columns:
                add_columns.remove(clmn)
            if print_log: print('\nadd_columns= ', add_columns)

            fig, ax = plt.subplots(figsize=(15, 8))
            paths_return_df[add_columns].plot(ax=ax, colormap=plt.cm.Spectral, legend=False)
            ax.set(title='CPCV testing return')
            ax.grid()
            fig.savefig(picture_path)
        warnings.filterwarnings(action='default')
        return res_dict


    def setting_features_count(self):
        """
        The function sets value of class variable n_features.
        :return: None
        """
        features_count = len(self.features_arr)
        n_features = int(features_count * self.features_part) if self.features_part <= 1. else features_count
        self.n_features = n_features
        print('\nfeatures_count= {0}, features_part= {1}, n_features= {2}'.format(features_count, self.features_part,
                                                                                self.n_features))


    def features_selection(self, data_lbl):
        """
        The function performs cyclic cross-validation testing with randomly selected features,
        determines the relative features importance and forms and save a dataframe with evaluated statistics.
        :param data_lbl: pandas dataframe.
        :return: None.
        """
        self.features_arr = self.setting_features_array(data_lbl, self.last_clmn)
        self.setting_features_count()
        # ---
        df_columns = ['acc_score_mean', 'acc_score_std', 'acc_score_arr', 'f1_score_mean', 'f1_score_std', 'f1_score_arr',
                      'conf_matrix_arr', 'return_mean', 'return_std', 'return_arr',
                      'sharpe_mean', 'sharpe_std', 'sharpe_arr', 'features_imp_dict']
        df_columns.extend(self.features_arr)
        df_st = pd.DataFrame(index=np.arange(0, self.n_loops), columns=df_columns)
        # ---
        df_train = self.data_for_ml  # уменьшенное количество образцов за счёт отбора
        df_test = data_lbl  # все образцы
        label_buy, label_sell = 'label_buy' + self.postfix, 'label_sell' + self.postfix
        df_lbl = data_lbl.loc[:, [label_buy, label_sell]]
        df_lbl.columns = ['label_buy', 'label_sell']
        # ---
        test_periods_count = len(self.testTimes)
        print('\ntest_periods_count= {0}'.format(test_periods_count))
        # ---
        time_start = dt.datetime.now()
        for step in range(self.n_loops):
            print('\nstep= ', step)
            features_for_select = copy.deepcopy(self.features_arr)
            features_for_ml = []
            acc_arr = []
            f1_arr = []
            ftrs_imp_arr = []
            # np.random.seed = 23

            print('self.n_features= {0}'.format(self.n_features))
            for i in range(self.n_features):
                sel_len = len(features_for_select)
                rnd = np.random.randint(0, sel_len)
                feat_to_add = features_for_select[rnd]
                # print('sel_len= {0}, rnd= {1}, feat_to_add= {2}'.format(sel_len, rnd, feat_to_add))
                features_for_ml.append(feat_to_add)
                features_for_select.pop(rnd)

            print('features_for_ml:\n', features_for_ml)

            my_res = self.cross_val_xgb(df_train_=df_train, df_test_=df_test,
                                        testTimes_=self.testTimes,
                                        features_for_ml_=features_for_ml,
                                        target_clmn_=self.target_clmn, max_depth_=3,
                                        n_estimators_=100,
                                        n_jobs_=-1, calc_fin_stats=True,
                                        df_lbl=df_lbl, profit_value=self.profit_value,
                                        loss_value=self.loss_value,
                                        print_log=True)

            # print('my_res:\n', my_res)

            df_st.loc[df_st.index == step, 'acc_score_mean'] = my_res['acc_score_mean']
            df_st.loc[df_st.index == step, 'acc_score_std'] = my_res['acc_score_std']
            df_st.loc[df_st.index == step, 'acc_score_arr'] = my_res['acc_score_arr']
            df_st.loc[df_st.index == step, 'f1_score_mean'] = my_res['f1_score_mean']
            df_st.loc[df_st.index == step, 'f1_score_std'] = my_res['f1_score_std']
            df_st.loc[df_st.index == step, 'f1_score_arr'] = my_res['f1_score_arr']

            df_st.loc[df_st.index == step, 'conf_matrix_arr'] = my_res['conf_matrix_arr']
            df_st.loc[df_st.index == step, 'return_mean'] = my_res['return_mean']
            df_st.loc[df_st.index == step, 'return_std'] = my_res['return_std']
            df_st.loc[df_st.index == step, 'return_arr'] = my_res['return_arr']
            df_st.loc[df_st.index == step, 'sharpe_mean'] = my_res['sharpe_mean']
            df_st.loc[df_st.index == step, 'sharpe_std'] = my_res['sharpe_std']
            df_st.loc[df_st.index == step, 'sharpe_arr'] = my_res['sharpe_arr']

            features_imp_dict = my_res['features_imp_dict']
            df_st.loc[df_st.index == step, 'features_imp_dict'] = str(features_imp_dict)

            #---
            for item in features_imp_dict.items():
                df_st.loc[df_st.index == step, item[0]] = item[1]
            #---
            # сохранение дампа
            with open(self.f_i_pickle_path, "wb") as pckl:
                pickle.dump(df_st, pckl)
            time_cur = dt.datetime.now()
            time_est = time_cur - time_start
            time_eta = (time_est/(step+1))*(self.n_loops-(step+1)) if (self.n_loops-(step+1)) != 0. else 0.
            print('\n{0:.2%} is done. time_eta= {1}'.format((step+1)/self.n_loops, time_eta))
            print('\n----------------------------------------------------------------------------------------------------\n')


    @staticmethod
    def features_arr_analyze(features_imp_arr_path='feat_imp.pickle', first_feature_number=14,
                             best_features_save=False, major_features_count=72, minor_features_count=128,
                             major_features_path='major_ftrs_arr.pickle', minor_features_path='minor_ftrs_arr.pickle',
                             selection_by_return=False, acc_basic_level=0.5, rtrn_basic_level=0.,
                             print_log=True):
        """
        The function analyzes data from features importance dataframe
        and generates aggregated statistics and ranking of features.
        :param features_imp_arr_path: string.
        :param first_feature_number: integer.
        :param best_features_save: boolean.
        :param major_features_count: integer.
        :param minor_features_count: integer.
        :param major_features_path: string.
        :param minor_features_path: string.
        :param selection_by_return: boolean.
        :param selection_by_return: boolean.
        :param acc_basic_level: float.
        :param rtrn_basic_level: float.
        :param prnt_log: boolean.
        :return: tuple of arrays with major and minor features.
        """
        with open(features_imp_arr_path, "rb") as pckl:
            feat_imp_df = pickle.load(pckl)

        feat_imp_df = feat_imp_df[:feat_imp_df['acc_score_mean'].dropna().count()]
        # replace 0. to np.nan. It's important for mean calculation.
        feat_imp_df.replace(to_replace=0., value=np.nan, inplace=True)
        feat_imp_df_shape = feat_imp_df.shape
        if print_log: print('feat_imp_df.shape= ', feat_imp_df_shape)
        #--- short statistics
        if print_log:
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(features_imp_arr_path)
            print('selection_by_return= ', selection_by_return)
            print('\nFeatures short statistics:')
            feat_statistics = feat_imp_df.loc[:,
                                  ['acc_score_mean', 'f1_score_mean', 'return_mean', 'sharpe_mean']].describe()
            print('\n', feat_statistics)

            n = feat_imp_df_shape[0]
            res_conf_int = stats.norm.interval(0.95, loc=feat_statistics.loc['mean', 'acc_score_mean'],
                                               scale=feat_statistics.loc['std', 'acc_score_mean'] / np.sqrt(n - 1))

            print('\naccuracy mean conf. intervals (for 0.95 conf.level)= [{0:.4f}, {1:.4f}]'.format(
                                                                                    res_conf_int[0], res_conf_int[1]))
            res_conf_int = stats.norm.interval(0.95, loc=feat_statistics.loc['mean', 'f1_score_mean'],
                                               scale=feat_statistics.loc['std', 'f1_score_mean'] / np.sqrt(n - 1))

            print('f1 score mean conf. intervals (for 0.95 conf.level)= [{0:.4f}, {1:.4f}]'.format(
                                                                                    res_conf_int[0], res_conf_int[1]))
            res_conf_int = stats.norm.interval(0.95, loc=feat_statistics.loc['mean', 'return_mean'],
                                               scale=feat_statistics.loc['std', 'return_mean'] / np.sqrt(n - 1))

            print('return mean conf. intervals (for 0.95 conf.level)= [{0:.4f}, {1:.4f}]'.format(
                                                                                    res_conf_int[0], res_conf_int[1]))
            res_conf_int = stats.norm.interval(0.95, loc=feat_statistics.loc['mean', 'sharpe_mean'],
                                               scale=feat_statistics.loc['std', 'sharpe_mean'] / np.sqrt(n - 1))

            print('sharpe ratio mean conf. intervals (for 0.95 conf.level)= [{0:.4f}, {1:.4f}]'.format(
                                                                                    res_conf_int[0], res_conf_int[1]))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            n_count = feat_imp_df[['acc_score_mean']].count()[0]
            pos_part_count = feat_imp_df.loc[feat_imp_df.acc_score_mean > acc_basic_level, ['acc_score_mean']].count()[0]
            pos_part = pos_part_count / n_count
            print('\nacc_mean > {0:.2f}: n_count= {1}, pos_part_count= {2}, pos_part= {3:.2%}'.format(acc_basic_level,
                                                                                    n_count, pos_part_count, pos_part))
            n_count = feat_imp_df[['return_mean']].count()[0]
            pos_part_count = feat_imp_df.loc[feat_imp_df.return_mean > rtrn_basic_level, ['return_mean']].count()[0]
            pos_part = pos_part_count / n_count
            print('\nreturn_mean > {0:.2f}: n_count= {1}, pos_part_count= {2}, pos_part= {3:.2%}'.format(rtrn_basic_level,
                                                                             n_count, pos_part_count, pos_part))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #---
        if selection_by_return:
            feat_imp_df_part = feat_imp_df.loc[feat_imp_df.return_mean > rtrn_basic_level, :]
        else:
            feat_imp_df_part = feat_imp_df.loc[feat_imp_df.acc_score_mean > acc_basic_level, :]
        if print_log:
            print('Features (selected part) short statistics:')
            print('\n', feat_imp_df_part.loc[:,
                        ['acc_score_mean', 'f1_score_mean', 'return_mean', 'sharpe_mean']].describe())
            # print('Columns in selected part of data_frame:')
            # for i, clmn in enumerate(feat_imp_df_part.columns):
            #     print('{0:<3}. {1}'.format(i, clmn))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #---
        res = feat_imp_df_part.loc[:, feat_imp_df_part.columns[first_feature_number:]].mean().sort_values(ascending=False)
        if print_log:
            print('5 first columns: ', \
                  feat_imp_df_part.loc[:, feat_imp_df_part.columns[first_feature_number:]].columns[:5])
            print('\nThe major features:\n')
            for i, item in enumerate(zip(res.index[:major_features_count], res[:major_features_count]), 1):
                print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))
            print('\nThe minor features:\n')
            for i, item in enumerate(zip(res.index[major_features_count:(major_features_count+minor_features_count)],
                                         res[major_features_count:(major_features_count+minor_features_count)]), 1):
                print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))
            print('\nThe worst features:\n')
            for i, item in enumerate(zip(res.index[-major_features_count:], res[major_features_count:]),
                                     -major_features_count):
                print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

        major_features_arr = list(res.index[0:major_features_count])
        minor_features_arr = list(res.index[major_features_count:(major_features_count + minor_features_count)])
        if best_features_save:
            with open(major_features_path, "wb") as pckl:
                pickle.dump(major_features_arr, pckl)
            with open(minor_features_path, "wb") as pckl:
                pickle.dump(minor_features_arr, pckl)

        return major_features_arr, minor_features_arr

    @staticmethod
    def features_importance_generalization(feat_imp_filenames_arr, clmn_names_arr, first_feature_number=14,
                                           selection_by_return=True,
                                           acc_basic_level=0.5, rtrn_basic_level=0.,
                                           save_res_df=True, res_df_pickle_path='gen_ftrs_imp.pickle',
                                           best_features_save=False, major_features_count=72, minor_features_count=128,
                                           major_features_path='major_ftrs_arr.pickle',
                                           minor_features_path='minor_ftrs_arr.pickle',
                                           print_log=True):
        """
        The function generalizes features importance from array of files with serialized dataframes
        with features relative importances.
        :param feat_imp_filenames_arr: string array.
            The files pathes from serialized dataframes with features relative importances.
        :param clmn_names_arr: string array.
        :param first_feature_number: integer.
        :param selection_by_return: boolean.
        :param acc_basic_level: float.
        :param rtrn_basic_level: float.
        :param save_res_df: boolean.
        :param res_df_pickle_path: string.
        :param best_features_save: boolean.
        :param major_features_count: integer.
        :param minor_features_count: integer.
        :param major_features_path: string.
        :param minor_features_path: string
        :param print_log: boolean.
        :return: pandas series with generalized features importances.
        """
        gen_imp_df = pd.DataFrame(columns=clmn_names_arr)
        if print_log: print('gen_imp_df:\n', gen_imp_df)

        for step, file_name in enumerate(feat_imp_filenames_arr):
            if print_log:
                print('step= {0}\nfile_name= {1}'.format(step, file_name))
                print('selection_by_return= ', selection_by_return)
            with open(file_name, 'rb') as pckl:
                feat_imp_df = pickle.load(pckl)
            feat_imp_df = feat_imp_df[:feat_imp_df['acc_score_mean'].dropna().count()]
            # replace 0. to np.nan. It's important for mean calculation.
            feat_imp_df.replace(to_replace=0., value=np.nan, inplace=True)
            feat_imp_df_shape = feat_imp_df.shape
            if print_log: print('\nfeat_imp_df.shape= ', feat_imp_df_shape)
            #---
            print('--------------------------------------------------------------------------------------')
            n_count = feat_imp_df[['acc_score_mean']].count()[0]
            pos_part_count = feat_imp_df.loc[feat_imp_df.acc_score_mean > acc_basic_level, ['acc_score_mean']].count()[0]
            pos_part_acc = pos_part_count / n_count
            print('acc_mean > {0:.2f}: n_count= {1}, pos_part_count= {2}, pos_part= {3:.2%}'.format(acc_basic_level,
                                                                                n_count, pos_part_count, pos_part_acc))
            n_count = feat_imp_df[['return_mean']].count()[0]
            pos_part_count = feat_imp_df.loc[feat_imp_df.return_mean > rtrn_basic_level, ['return_mean']].count()[0]
            pos_part_rtrn = pos_part_count / n_count
            print('\nreturn_mean > {0:.2f}: n_count= {1}, pos_part_count= {2}, pos_part= {3:.2%}'.format(rtrn_basic_level,
                                                                             n_count, pos_part_count, pos_part_rtrn))
            print('--------------------------------------------------------------------------------------')
            #---
            if selection_by_return:
                feat_imp_df_part = feat_imp_df.loc[feat_imp_df.return_mean > rtrn_basic_level, :]
                weight = pos_part_rtrn
            else:
                feat_imp_df_part = feat_imp_df.loc[feat_imp_df.acc_score_mean > acc_basic_level, :]
                weight = pos_part_acc
            if print_log:
                print('\nFeatures (selected part) short statistics:')
                print('\n', feat_imp_df_part.loc[:,
                            ['acc_score_mean', 'f1_score_mean', 'return_mean', 'sharpe_mean']].describe())
                # print('Columns in selected part of data_frame:')
                # for i, clmn in enumerate(feat_imp_df_part.columns):
                #     print('{0:<3}. {1}'.format(i, clmn))
                print('--------------------------------------------------------------------------------------')
            # ---
            res = feat_imp_df_part.loc[:, feat_imp_df_part.columns[first_feature_number:]].mean().sort_values(
                ascending=False)
            # print('res:\n', res)
            res = res*weight
            # print('\nres (after weight multiplication):\n', res)
            if print_log:
                # print('5 first columns: ', \
                #       feat_imp_df_part.loc[:, feat_imp_df_part.columns[first_feature_number:]].columns[:5])
                # print('res:\n',res)
                print('\nThe features ranking (after weight multiplication):\n')
                for i, item in enumerate(zip(res.index, res), 1):
                    print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))
            print('******************************************************************************************\n')

            gen_imp_df[clmn_names_arr[step]] = res

        gen_imp_df['total'] = gen_imp_df.sum(axis=1)
        gen_imp_df.sort_values(by='total', ascending=False, inplace=True)

        if print_log: print('gen_imp_df:\n', gen_imp_df)
        if save_res_df:
            with open(res_df_pickle_path, 'wb') as pckl:
                pickle.dump(gen_imp_df, pckl)

        if best_features_save:
            major_features_arr = list(gen_imp_df.index[0:major_features_count])
            minor_features_arr = list(gen_imp_df.index[major_features_count:(major_features_count + minor_features_count)])
            with open(major_features_path, "wb") as pckl:
                pickle.dump(major_features_arr, pckl)
            with open(minor_features_path, "wb") as pckl:
                pickle.dump(minor_features_arr, pckl)
            print('\nThe major features:\n')
            for i, item in enumerate(zip(gen_imp_df.index[:major_features_count],
                                         gen_imp_df[:major_features_count]['total'].values), 1):
                print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))
            print('\nThe minor features:\n')
            for i, item in enumerate(zip(gen_imp_df.index[major_features_count:(major_features_count+minor_features_count)],
                                         gen_imp_df[major_features_count:(major_features_count+minor_features_count)]['total'].values), 1):
                print('{0:<2}. {1:<25}{2:.5f}'.format(i, item[0], item[1]))

        return gen_imp_df


    def execute_cpcv(self):
        """
        The function executes CPCV testing.
        :return: None
        """
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        # --- dataframe load
        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)
        print('\ndata.shape: ', data.shape)

        with open(self.label_pickle_path, "rb") as pckl:
            lbl = pickle.load(pckl)
            label_buy, label_sell = 'label_buy' + self.postfix, 'label_sell' + self.postfix
            lbl['label_buy'] = lbl[label_buy]
            lbl['label_sell'] = lbl[label_sell]
            lbl.drop(columns=[label_buy, label_sell], inplace=True)
        print('lbl.shape: ', lbl.shape)
        # ---
        data_lbl = pd.concat((data, lbl), axis=1)
        print('data_lbl.shape: ', data_lbl.shape)

        # ---
        # del data
        # del lbl

        # data_for_ml = self.select_data_for_ml(data_lbl=data_lbl, price_step=self.price_step,
        #                                            target_clmn=self.target_clmn)
        # with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "wb") as pckl:
        #      pickle.dump(data_for_ml, pckl)

        # ---
        # загрузка датафрейма в тестовых целях
        with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "rb") as pckl:
            data_for_ml = pickle.load(pckl)
        # ---
        del data_lbl
        #---
        features_for_ml = \
            ['adi_12', 'adi_6', 'lr_duo_1440_5i0', 'adi_36', 'adi_432',
             'adi_720', 'adi_144', 'hurst_288_10', 'tema_288', 'adi_1440',
             'sr_576', 'adx_72', 'lr_cmpr_1152_2i5', 'adi_48', 'ema_open_288',
             'adi_288', 'adx_720', 'sr_1440', 'lr_cmpr_1440_1i5', 'adi_18',
             'lr_cmpr_720_5i0', 'dema_open_720', 'lr_cmpr_1152_5i0']
        # ['lr_duo_1440_5i0',
        #  'lr_duo_1152_5i0',
        #  'ema_720',
        #  'lr_duo_288_5i0',
        #  'adx_576',
        #  'lr_duo_576_5i0',
        #  'lr_duo_1152_2i5',
        #  'lr_uno_1440_1i5',
        #  'sr_1440',
        #  'lr_uno_1440_5i0',
        #  'lr_uno_1440_2i5',
        #  'lr_duo_576_2i5',
        #  'lr_duo_864_5i0',
        #  'lr_duo_1440_2i5',
        #  'lr_uno_1152_2i5',
        #  'ema_576',
        #  'lr_duo_720_5i0',
        #  'lr_uno_1152_5i0',
        #  'lr_uno_1152_1i5',
        #  'ema_432',
        #  'lr_duo_864_1i5',
        #  'adx_432',
        #  'tema_288',
        #  'lr_duo_720_1i5',
        #  'tema_12',
        #  'tema_720',
        #  'adx_720',
        #  'dema_6',
        #  'lr_duo_864_2i5',
        #  'lr_duo_1440_1i5',
        #  'lr_duo_288_1i5',
        #  'ema_288',
        #  'adi_6',
        #  'ema_60',
        #  'tema_6',
        #  'tema_190',
        #  'lr_cmpr_1152_5i0',
        #  'ema_18',
        #  'open',
        #  'tema_432',
        #  'hurst_1440_10',
        #  'tema_576',
        #  'rtrn_864',
        #  'dema_576',
        #  'tema_24',
        #  'lr_cmpr_1440_5i0',
        #  'sr_432',
        #  'ema_6',
        #  'adx_144',
        #  'dema_72',
        #  'sr_720',
        #  'tema_18',
        #  'dema_12',
        #  'adx_288',
        #  'adi_720',
        #  'rtrn_1440',
        #  'ema_12',
        #  'adi_1440',
        #  'dema_108',
        #  'dema_18',
        #  'dema_432',
        #  'lr_duo_108_5i0',
        #  'ema_24',
        #  'dema_144',
        #  'lr_duo_72_5i0',
        #  'adi_60',
        #  'ema_48',
        #  'bb_rp_1440_1i0',
        #  'adi_12',
        #  'hurst_720_50',
        #  'adi_24',
        #  'ema_72',
        #  'lr_duo_1152_1i5',
        #  'dema_288',
        #  'adi_48',
        #  'lr_cmpr_720_5i0',
        #  'lr_duo_190_5i0',
        #  'adi_18',
        #  'ema_144',
        #  'hurst_1440_25',
        #  'ema_36',
        #  'sr_576',
        #  'adi_576',
        #  'bb_rp_1440_3i0',
        #  'hurst_1440_50',
        #  'lr_cmpr_1440_2i5',
        #  'adx_190',
        #  'adi_36',
        #  'tema_144',
        #  'dema_720',
        #  'ema_108',
        #  'bb_rp_1440_2i0',
        #  'tema_108',
        #  'dema_190',
        #  'lr_duo_190_2i5',
        #  'hurst_576_50',
        #  'cci_1440',
        #  'hurst_576_25',
        #  'adi_72',
        #  'hurst_864_50',
        #  'rsi_720',
        #  'macd_s_117_234_78',
        #  'dema_36',
        #  'adx_108',
        #  'dema_24',
        #  'lr_cmpr_864_5i0',
        #  'hurst_720_25',
        #  'hurst_576_10',
        #  'tema_72',
        #  'adi_432',
        #  'rtrn_1152',
        #  'rtrn_720',
        #  'hurst_864_10',
        #  'lr_duo_720_2i5',
        #  'rtrn_576',
        #  'hurst_1152_50',
        #  'adi_108',
        #  'macd_117_234_78',
        #  'sr_288',
        #  'adi_144',
        #  'ema_open_720',
        #  'hurst_720_10',
        #  'ema_cmpr_6_720',
        #  'hurst_288_25',
        #  'hurst_1152_10',
        #  'lr_uno_576_1i5',
        #  'lr_uno_288_1i5',
        #  'lr_uno_576_2i5',
        #  'cci_1152',
        #  'rsi_576',
        #  'hurst_288_50',
        #  'lr_uno_864_5i0',
        #  'lr_cmpr_576_2i5',
        #  'cci_720',
        #  'hurst_432_50',
        #  'lr_cmpr_1152_1i5',
        #  'hurst_432_10',
        #  'lr_uno_864_2i5',
        #  'lr_uno_576_5i0',
        #  'cci_864',
        #  'lr_duo_576_1i5',
        #  'lr_uno_864_1i5',
        #  'mfi_720',
        #  'mfi_288',
        #  'lr_duo_108_2i5',
        #  'lr_uno_288_5i0',
        #  'adi_288',
        #  'hurst_864_25',
        #  'mfi_576',
        #  'lr_cmpr_1152_2i5',
        #  'lr_uno_720_1i5',
        #  'lr_cmpr_1440_1i5',
        #  'ema_open_576',
        #  'tema_open_720',
        #  'hurst_1152_25',
        #  'lr_cmpr_720_2i5',
        #  'lr_uno_720_5i0',
        #  'lr_duo_190_1i5',
        #  'ema_open_288',
        #  'lr_uno_288_2i5',
        #  'tema_36',
        #  'hurst_288_10',
        #  'cci_576',
        #  'tema_cmpr_6_720',
        #  'cci_288',
        #  'lr_cmpr_576_5i0',
        #  'dema_open_720',
        #  'cci_432',
        #  'ema_open_432',
        #  'mfi_432',
        #  'lr_cmpr_864_2i5',
        #  'ema_cmpr_6_288',
        #  'dema_cmpr_6_720',
        #  'dema_open_576',
        #  'lr_uno_720_2i5',
        #  'tema_open_576',
        #  'hurst_432_25',
        #  'lr_duo_288_2i5',
        #  'bb_rp_576_1i0',
        #  'lr_cmpr_864_1i5',
        #  'dema_cmpr_6_432',
        #  'bb_rp_576_2i0',
        #  'bb_rp_720_1i0',
        #  'lr_cmpr_576_1i5',
        #  'dema_open_432',
        #  'lr_uno_190_5i0',
        #  'bb_rp_720_3i0',
        #  'rtrn_190',
        #  'so_k_234_2',
        #  'bb_rp_288_3i0',
        #  'lr_duo_36_5i0',
        #  'lr_uno_190_1i5',
        #  'rsi_144',
        #  'so_d_234_2',
        #  'bb_rp_576_3i0',
        #  'bb_rp_720_2i0',
        #  'adx_72',
        #  'rtrn_144',
        #  'lr_duo_108_1i5',
        #  'lr_uno_190_2i5']
        cpcv_n = 5
        cpcv_k = 2
        max_depth = 3
        n_estimators = 5
        use_pred_proba = True
        pred_proba_threshold = .505  #.505
        if use_pred_proba:
            pickle_path = self.folder_name + os.sep + 'paths_return_df' + self.postfix + '_' + self.version +\
                          '_pred_prob_'+processing.digit_to_text(pred_proba_threshold) + '.pickle'
            picture_path = self.folder_name + os.sep + 'CPCV_testing_return' + self.postfix + '_' + \
                           str(cpcv_n) + '_' + str(cpcv_k) + '_' + str(max_depth) + '_' + str(n_estimators)  + '_' +\
                           self.version + '_pred_prob_'+processing.digit_to_text(pred_proba_threshold)+ '.jpg'
        else:
            pickle_path = self.folder_name + os.sep + 'paths_return_df' + self.pickle_postfix
            picture_path = self.folder_name + os.sep + 'CPCV_testing_return' + self.postfix +'_'+ \
                           str(cpcv_n)+'_'+ str(cpcv_k)+'_'+ str(max_depth)+'_'+ str(n_estimators)+'.jpg'
        res = self.cpcv_xgb(data_for_ml, data, lbl, features_for_ml, self.target_clmn,
                 start_date=self.start_date, finish_date=self.finish_date,
                 purged_period=3, cpcv_n=cpcv_n, cpcv_k=cpcv_k, max_depth=max_depth, n_estimators=n_estimators,
                 n_jobs=-1, profit_value=self.profit_value, loss_value=self.loss_value,
                 use_pred_proba=use_pred_proba, pred_proba_threshold=pred_proba_threshold,
                 save_paths_return=True, pickle_path=pickle_path,
                 save_picture=True, picture_path=picture_path,
                 pred_values_series_aggregation=True, dump_model=True, print_log=True)
        #---
        with open(self.folder_name + os.sep + "paths_pred_df.pickle", "wb") as pckl:
            pickle.dump(res['paths_pred_df'], pckl)

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))


    def cpcv_mean_decrease_efficiency(self):
        """
        The function executes CPCV testing.
        :return: None
        """
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        # --- dataframe load
        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)
        print('\ndata.shape: ', data.shape)

        with open(self.label_pickle_path, "rb") as pckl:
            lbl = pickle.load(pckl)
            label_buy, label_sell = 'label_buy' + self.postfix, 'label_sell' + self.postfix
            lbl['label_buy'] = lbl[label_buy]
            lbl['label_sell'] = lbl[label_sell]
            lbl.drop(columns=[label_buy, label_sell], inplace=True)
        print('lbl.shape: ', lbl.shape)
        # ---
        data_lbl = pd.concat((data, lbl), axis=1)
        print('data_lbl.shape: ', data_lbl.shape)

        # ---
        # del data
        # del lbl

        # data_for_ml = self.select_data_for_ml(data_lbl=data_lbl, price_step=self.price_step,
        #                                            target_clmn=self.target_clmn)
        # with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "wb") as pckl:
        #      pickle.dump(data_for_ml, pckl)

        # ---
        # загрузка датафрейма в тестовых целях
        with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "rb") as pckl:
            data_for_ml = pickle.load(pckl)
        # ---
        del data_lbl
        #---
        features_for_ml = \
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
        cpcv_n = 5
        cpcv_k = 2
        max_depth = 3
        n_estimators = 5
        use_pred_proba = True
        pred_proba_threshold = .505  #.505
        res_dic = {}
        res = self.cpcv_xgb(data_for_ml, data, lbl, features_for_ml, self.target_clmn,
                 start_date=self.start_date, finish_date=self.finish_date,
                 purged_period=3, cpcv_n=cpcv_n, cpcv_k=cpcv_k, max_depth=max_depth, n_estimators=n_estimators,
                 n_jobs=-1, profit_value=self.profit_value, loss_value=self.loss_value,
                 use_pred_proba=use_pred_proba, pred_proba_threshold=pred_proba_threshold,
                 save_paths_return=False, pickle_path='',
                 save_picture=False, picture_path='',
                 pred_values_series_aggregation=True, dump_model=False, print_log=True)
        #---
        sr_base = res['sr_mean']
        res_dic['_base_sr'] = (sr_base, 0.)
        print(res_dic)
        #---
        print('***************************************************************************************************')
        time_loop_start = dt.datetime.now()
        arr_len = len(features_for_ml)
        for i, ftr in enumerate(features_for_ml, 1):
            print('i= {0}, removed feature= {1}'.format(i, ftr))
            features_for_ml_copy = copy.deepcopy(features_for_ml)
            features_for_ml_copy.remove(ftr)
            res = self.cpcv_xgb(data_for_ml, data, lbl, features_for_ml_copy, self.target_clmn,
                     start_date=self.start_date, finish_date=self.finish_date,
                     purged_period=3, cpcv_n=cpcv_n, cpcv_k=cpcv_k, max_depth=max_depth, n_estimators=n_estimators,
                     n_jobs=-1, profit_value=self.profit_value, loss_value=self.loss_value,
                     use_pred_proba=use_pred_proba, pred_proba_threshold=pred_proba_threshold,
                     save_paths_return=False, pickle_path='',
                     save_picture=False, picture_path='',
                     pred_values_series_aggregation=True, dump_model=False, print_log=True)
            sr_cur = res['sr_mean']
            delta = sr_cur - sr_base
            res_dic[ftr] = (sr_cur, delta)
            # print(res_dic)
            time_cur = dt.datetime.now()
            time_left = time_cur - time_loop_start
            time_eta = time_left/i*(arr_len-i)
            print('sr_cur= {0:.6f}, delta= {1:.6f}'.format(sr_cur, delta))
            print('time_eta= {0}, time_left= {1}, time_cur={2}'.format(time_eta, time_left, time_cur))
            print('***************************************************************************************************')


        with open(self.folder_name + os.sep + "ftrs_mean_decr_eff_res_dic.pickle", "wb") as pckl:
            pickle.dump(res_dic, pckl)

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))


    def cpcv_mean_increase_efficiency(self):
        """
        The function executes CPCV testing.
        :return: None
        """
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        # --- dataframe load
        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)
        print('\ndata.shape: ', data.shape)

        with open(self.label_pickle_path, "rb") as pckl:
            lbl = pickle.load(pckl)
            label_buy, label_sell = 'label_buy' + self.postfix, 'label_sell' + self.postfix
            lbl['label_buy'] = lbl[label_buy]
            lbl['label_sell'] = lbl[label_sell]
            lbl.drop(columns=[label_buy, label_sell], inplace=True)
        print('lbl.shape: ', lbl.shape)
        # ---
        data_lbl = pd.concat((data, lbl), axis=1)
        print('data_lbl.shape: ', data_lbl.shape)

        # ---
        # del data
        # del lbl

        # data_for_ml = self.select_data_for_ml(data_lbl=data_lbl, price_step=self.price_step,
        #                                            target_clmn=self.target_clmn)
        # with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "wb") as pckl:
        #      pickle.dump(data_for_ml, pckl)

        # ---
        # загрузка датафрейма в тестовых целях
        with open(self.folder_name + os.sep + "data_for_ml_test_1.0.pickle", "rb") as pckl:
            data_for_ml = pickle.load(pckl)
        # ---
        del data_lbl
        #---
        features_for_ml_kernel = \
            ['adi_12', 'adi_6', 'lr_duo_1440_5i0', 'adi_36', 'adi_432',
             'adi_720', 'adi_144', 'hurst_288_10', 'tema_288', 'adi_1440',
             'sr_576', 'adx_72', 'lr_cmpr_1152_2i5', 'adi_48', 'ema_open_288',
             'adi_288', 'adx_720', 'sr_1440', 'lr_cmpr_1440_1i5', 'adi_18',
             'lr_cmpr_720_5i0', 'dema_open_720', 'lr_cmpr_1152_5i0']
        features_for_ml_additional = \
            ['lr_uno_864_1i5', 'mfi_720', 'mfi_288', 'hurst_576_10',
             'rtrn_1152', 'lr_duo_108_2i5', 'lr_uno_864_2i5', 'lr_uno_288_5i0',
             'hurst_432_10', 'lr_cmpr_1152_1i5', 'lr_duo_576_1i5',
             'lr_uno_576_5i0', 'hurst_1152_50', 'lr_cmpr_576_2i5',
             'macd_117_234_78', 'sr_288', 'ema_open_720', 'hurst_720_10',
             'ema_cmpr_6_720', 'lr_duo_720_2i5', 'hurst_288_25',
             'hurst_1152_10', 'cci_720', 'lr_uno_576_1i5', 'hurst_864_10',
             'lr_uno_576_2i5', 'cci_1152', 'rsi_576', 'hurst_864_25',
             'lr_uno_864_5i0', 'rtrn_720', 'rtrn_576', 'lr_uno_288_1i5',
             'hurst_288_50', 'ema_open_576', 'lr_uno_720_1i5', 'bb_rp_576_1i0',
             'lr_cmpr_864_1i5', 'dema_cmpr_6_432', 'bb_rp_576_2i0',
             'bb_rp_720_1i0', 'lr_cmpr_576_1i5', 'dema_open_432',
             'lr_uno_190_5i0', 'bb_rp_720_3i0', 'rtrn_190', 'so_k_234_2',
             'bb_rp_288_3i0', 'lr_duo_36_5i0', 'lr_uno_190_1i5', 'rsi_144',
             'so_d_234_2', 'bb_rp_576_3i0', 'bb_rp_720_2i0', 'rtrn_144',
             'lr_duo_288_2i5', 'mfi_576', 'hurst_432_25', 'lr_uno_720_2i5',
             'hurst_720_25', 'tema_open_720', 'hurst_1152_25',
             'lr_cmpr_720_2i5', 'lr_uno_720_5i0', 'lr_duo_190_1i5',
             'lr_uno_288_2i5', 'tema_36', 'cci_576', 'tema_cmpr_6_720',
             'cci_288', 'lr_cmpr_576_5i0', 'cci_432', 'ema_open_432', 'mfi_432',
             'lr_cmpr_864_2i5', 'ema_cmpr_6_288', 'dema_cmpr_6_720',
             'dema_open_576', 'tema_open_576', 'lr_cmpr_864_5i0',
             'hurst_864_50', 'tema_18', 'lr_duo_864_2i5', 'lr_duo_1440_1i5',
             'lr_duo_288_1i5', 'ema_288', 'ema_60', 'tema_6', 'tema_190',
             'ema_18', 'dema_6', 'tema_432', 'dema_576', 'tema_24',
             'lr_cmpr_1440_5i0', 'ema_6', 'dema_72', 'dema_24', 'dema_12',
             'adx_288', 'rtrn_864', 'rtrn_1440', 'tema_720', 'lr_duo_720_1i5',
             'lr_duo_1152_5i0', 'ema_720', 'lr_duo_288_5i0', 'lr_duo_576_5i0',
             'lr_duo_1152_2i5', 'lr_uno_1440_1i5', 'lr_uno_1440_5i0',
             'lr_uno_1440_2i5', 'tema_12', 'lr_duo_576_2i5', 'lr_uno_1152_2i5',
             'ema_576', 'lr_duo_720_5i0', 'lr_uno_1152_5i0', 'lr_uno_1152_1i5',
             'ema_432', 'lr_duo_864_1i5', 'adx_432', 'lr_duo_1440_2i5',
             'ema_12', 'lr_uno_190_2i5', 'dema_18', 'dema_288',
             'lr_duo_190_5i0', 'ema_144', 'hurst_1440_25', 'ema_36', 'dema_108',
             'lr_duo_1152_1i5', 'bb_rp_1440_3i0', 'lr_cmpr_1440_2i5', 'adx_190',
             'tema_144', 'dema_720', 'ema_108', 'bb_rp_1440_2i0',
             'hurst_1440_50', 'lr_duo_190_2i5', 'hurst_576_50', 'cci_1440',
             'dema_432', 'lr_duo_108_5i0', 'ema_24', 'dema_36', 'dema_144',
             'macd_s_117_234_78', 'lr_duo_72_5i0', 'dema_190', 'ema_48',
             'bb_rp_1440_1i0', 'hurst_720_50', 'rsi_720', 'lr_duo_108_1i5',
             'hurst_576_25', 'tema_108', 'adi_60', 'tema_72', 'adx_108',
             'tema_576', 'ema_72', 'adi_576', 'adx_144', 'adx_576', 'open',
             'lr_duo_864_5i0', 'hurst_432_50', 'sr_432', 'sr_720',
             'hurst_1440_10', 'adi_72', 'adi_108', 'adi_24']
        cpcv_n = 5
        cpcv_k = 2
        max_depth = 3
        n_estimators = 5
        use_pred_proba = True
        pred_proba_threshold = .505  #.505
        res_dic = {}
        res = self.cpcv_xgb(data_for_ml, data, lbl, features_for_ml_kernel, self.target_clmn,
                 start_date=self.start_date, finish_date=self.finish_date,
                 purged_period=3, cpcv_n=cpcv_n, cpcv_k=cpcv_k, max_depth=max_depth, n_estimators=n_estimators,
                 n_jobs=-1, profit_value=self.profit_value, loss_value=self.loss_value,
                 use_pred_proba=use_pred_proba, pred_proba_threshold=pred_proba_threshold,
                 save_paths_return=False, pickle_path='',
                 save_picture=False, picture_path='',
                 pred_values_series_aggregation=True, dump_model=False, print_log=True)
        #---
        sr_base = res['sr_mean']
        res_dic['_base_sr'] = (sr_base, 0.)
        print(res_dic)
        #---
        print('***************************************************************************************************')
        time_loop_start = dt.datetime.now()
        arr_len = len(features_for_ml_additional)
        for i, ftr in enumerate(features_for_ml_additional, 1):
            print('i= {0}, added feature= {1}'.format(i, ftr))
            features_for_ml = copy.deepcopy(features_for_ml_kernel)
            features_for_ml.append(ftr)
            res = self.cpcv_xgb(data_for_ml, data, lbl, features_for_ml, self.target_clmn,
                     start_date=self.start_date, finish_date=self.finish_date,
                     purged_period=3, cpcv_n=cpcv_n, cpcv_k=cpcv_k, max_depth=max_depth, n_estimators=n_estimators,
                     n_jobs=-1, profit_value=self.profit_value, loss_value=self.loss_value,
                     use_pred_proba=use_pred_proba, pred_proba_threshold=pred_proba_threshold,
                     save_paths_return=False, pickle_path='',
                     save_picture=False, picture_path='',
                     pred_values_series_aggregation=True, dump_model=False, print_log=True)
            sr_cur = res['sr_mean']
            delta = sr_cur-sr_base
            res_dic[ftr] = (sr_cur, delta)
            # print(res_dic)
            time_cur = dt.datetime.now()
            time_left = time_cur - time_loop_start
            time_eta = time_left/i*(arr_len-i)
            print('sr_cur= {0:.6f}, delta= {1:.6f}'.format(sr_cur, delta))
            print('time_eta= {0}, time_left= {1}, time_cur={2}'.format(time_eta, time_left, time_cur))
            print('***************************************************************************************************')


        with open(self.folder_name + os.sep + "ftrs_mean_incr_eff_res_dic.pickle", "wb") as pckl:
            pickle.dump(res_dic, pckl)

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))


    def execute_selection(self):
        """
        The function executes features selection cycle.
        :return: None
        """
        # --- dataframe load
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)
        print('\ndata.shape: ', data.shape)

        with open(self.label_pickle_path, "rb") as pckl:
            lbl = pickle.load(pckl)
        print('lbl.shape: ', lbl.shape)

        data_lbl = pd.concat((data, lbl), axis=1)
        print('data_lbl.shape: ', data_lbl.shape)

        self.last_clmn = data.shape[1]
        print('self.last_clmn= ', self.last_clmn)
        print('last 5 columns: ', data_lbl.columns[self.last_clmn-5 : self.last_clmn])
        # ---
        del data
        del lbl
        self.testTimes = self.set_testTimes(dt0=self.dt0, dt1=self.dt1)

        self.data_for_ml = self.select_data_for_ml(data_lbl=data_lbl, price_step=self.price_step,
                                                   target_clmn=self.target_clmn)

        # # ---
        # # загрузка датафрейма в тестовых целях
        # with open(r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/data_for_ml_test_1.0.pickle", "rb") as pckl:
        #     data_for_ml = pickle.load(pckl)
        # self.data_for_ml = data_for_ml
        # # ---

        #---
        self.features_selection(data_lbl)
        #---

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))


if __name__ == '__main__':
    req = FeaturesSelectionClass()
    # req.execute_selection()
    #---
    #---
    # res = req.features_arr_analyze(features_imp_arr_path=req.f_i_pickle_path, first_feature_number=14,
    #                                best_features_save=True, major_features_count=72, minor_features_count=128,
    #                                major_features_path=req.ftrs_major_arr_pickle_path,
    #                                minor_features_path=req.ftrs_minor_arr_pickle_path,
    #                                selection_by_return=True,
    #                                print_log=True)
    # print('major features:\n{0}\nminor features:\n{1}'.format(res[0], res[1]))
    #---
    #---
    # feat_imp_filenames_arr = [req.folder_name+os.sep+file_name for file_name in req.feat_imp_filenames_arr]
    # res_df_pickle_path = req.folder_name+os.sep+'gen_imp_df_v.1.0.pickle'
    # major_features_path = req.folder_name+os.sep+'ftrs_gen_major_arr_v1.0.pickle'
    # minor_features_path = req.folder_name+os.sep+'ftrs_gen_minor_arr_v1.0.pickle'
    # res = req.features_importance_generalization(feat_imp_filenames_arr=feat_imp_filenames_arr,
    #                                              clmn_names_arr=req.clmn_names_arr,
    #                                              first_feature_number=14,
    #                                              selection_by_return=True,
    #                                              acc_basic_level=0.5, rtrn_basic_level=0.,
    #                                              save_res_df=True, res_df_pickle_path=res_df_pickle_path,
    #                                              best_features_save=True, major_features_count=72,
    #                                              minor_features_count=128, major_features_path=major_features_path,
    #                                              minor_features_path=minor_features_path,
    #                                              print_log=True)
    #---
    #---
    # req.execute_cpcv()
    #---
    # req.cpcv_mean_decrease_efficiency()
    #---
    #---
    req.cpcv_mean_increase_efficiency()
    #---