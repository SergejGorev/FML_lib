import finfunctions
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

class FeaturesSelectionClass:
    n_loops = 2500  # количество циклов
    features_part = 0.06  # доля признаков, участвующих в тестировании в каждом проходе
    folder_name = r"/home/rom/01-Algorithmic_trading/02_1-EURUSD"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD"  #  r"/home/rom/01-Algorithmic_trading/02_1-EURUSD"
    data_pickle_file_name = "eurusd_5_v1.4.pickle"
    label_pickle_file_name = "eurusd_5_v1_lbl_0i0035_1i0_1i0.pickle"

    postfix = '_0i0035_1i0_1i0'
    version = 'v1.0'
    target_clmn_prefix = 'target_label'
    profit_value = 18
    loss_value = -10 #-25
    dump_pickle = True # dump data pickle
    f_i_pickle_prefix = "feat_imp"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\feat_imp_20180905_3.pickle"
    ftrs_major_arr_pickle_prefix = "ftrs_major_arr"
    ftrs_minor_arr_pickle_prefix = "ftrs_minor_arr"

    price_step = 0.001
    # train_start = dt.datetime(2005, 1, 1, 0, 0)
    # test_start = dt.datetime(2017, 7, 1, 0, 0)
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
        # формирование тестовых периодов
        testTimes = pd.Series(dt1, index=dt0)
        print('\ntestTimes:\n{}'.format(testTimes))
        return testTimes

    @staticmethod
    def setting_features_array(data_lbl, last_clmn):
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


    def setting_features_count(self):
        features_count = len(self.features_arr)
        n_features = int(features_count * self.features_part) if self.features_part <= 1. else features_count
        self.n_features = n_features
        print('\nfeatures_count= {0}, features_part= {1}, n_features= {2}'.format(features_count, self.features_part,
                                                                                self.n_features))


    def features_selection(self, data_lbl):
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
            pckl = open(self.f_i_pickle_path, "wb")
            pickle.dump(df_st, pckl)
            pckl.close()
            time_cur = dt.datetime.now()
            time_est = time_cur - time_start
            time_eta = (time_est/(step+1))*(self.n_loops-(step+1)) if (self.n_loops-(step+1)) != 0. else 0.
            print('\n{0:.2%} is done. time_eta= {1}'.format((step+1)/self.n_loops, time_eta))
            print('\n----------------------------------------------------------------------------------------------------\n')


    @staticmethod
    def features_arr_analyze(features_imp_arr_path='feat_imp.pickle', first_feature_number=14,
                             best_features_save=False, major_features_count=72, minor_features_count=128,
                             major_features_path='major_ftrs_arr.pickle', minor_features_path='minor_ftrs_arr.pickle',
                             selection_by_return=False, print_log=True):
        with open(features_imp_arr_path, "rb") as pckl:
            feat_imp_df = pickle.load(pckl)
            pckl.close()

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
            pos_part_count = feat_imp_df.loc[feat_imp_df.acc_score_mean > 0.5, ['acc_score_mean']].count()[0]
            pos_part = pos_part_count / n_count
            print('\nacc_mean > 0.5: n_count= {0}, pos_part_count= {1}, pos_part= {2:.2%}'.format(n_count, pos_part_count,
                                                                                                pos_part))
            n_count = feat_imp_df[['return_mean']].count()[0]
            pos_part_count = feat_imp_df.loc[feat_imp_df.return_mean > 0., ['return_mean']].count()[0]
            pos_part = pos_part_count / n_count
            print('\nreturn_mean > 0.0: n_count= {0}, pos_part_count= {1}, pos_part= {2:.2%}'.format(n_count,
                                                                                                   pos_part_count,
                                                                                                   pos_part))
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        #---
        if selection_by_return:
            feat_imp_df_part = feat_imp_df.loc[feat_imp_df.return_mean > 0., :]
        else:
            feat_imp_df_part = feat_imp_df.loc[feat_imp_df.acc_score_mean > 0.5, :]
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
                pckl.close()
            with open(minor_features_path, "wb") as pckl:
                pickle.dump(minor_features_arr, pckl)
                pckl.close()

        return major_features_arr, minor_features_arr


    def execute(self):
        # --- dataframe load
        time_start = dt.datetime.now()
        print('time_start= {}'.format(time_start))

        pckl = open(self.data_pickle_path, "rb")
        data = pickle.load(pckl)
        pckl.close()
        print('\ndata.shape: ', data.shape)

        pckl = open(self.label_pickle_path, "rb")
        lbl = pickle.load(pckl)
        pckl.close()
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
        # pckl = open(r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/data_for_ml_test_1.0.pickle", "rb")
        # data_for_ml = pickle.load(pckl)
        # pckl.close()
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
    # req.execute()

    res = req.features_arr_analyze(features_imp_arr_path=req.f_i_pickle_path, first_feature_number=14,
                                   best_features_save=True, major_features_count=72, minor_features_count=128,
                                   major_features_path=req.ftrs_major_arr_pickle_path,
                                   minor_features_path=req.ftrs_minor_arr_pickle_path,
                                   selection_by_return=True,
                                   print_log=True)
    print('major features:\n{0}\nminor features:\n{1}'.format(res[0], res[1]))