import finfunctions
import pandas as pd
import numpy as np
import datetime as dt
import pickle
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import copy

class FeaturesSelectionClass:
    n_loops = 2500  # количество циклов
    features_part = 0.10  # доля признаков, участвующих в тестировании в каждом проходе
    data_pickle_path = r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.3.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.3.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.3.pickle"
    label_pickle_path = r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"
    target_clmn = 'target_label_0i0025_1i0_1i0'
    dump_pickle = True # dump data pickle
    f_i_path_for_dump = r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\feat_imp_20180905_3.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/feat_imp_20180905_3.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\feat_imp_20180905_3.pickle"

    price_step = 0.001
    train_start = dt.datetime(2005, 1, 1, 0, 0)
    test_start = dt.datetime(2017, 7, 1, 0, 0)
    dt0 = [dt.datetime(2009, 7, 1), dt.datetime(2011, 7, 1), dt.datetime(2013, 7, 1), dt.datetime(2015, 7, 1),
           dt.datetime(2017, 7, 1)]
    dt1 = [dt.datetime(2010, 6, 15), dt.datetime(2012, 6, 15), dt.datetime(2014, 6, 15), dt.datetime(2016, 6, 15),
           dt.datetime(2018, 6, 15)]

    testTimes = None
    data_for_ml = None
    features_arr = []
    n_features = 0
    last_clmn = 0
    # ---

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
               max_depth_=3, n_estimators_=100, n_jobs_=-1, print_log=True):
        res_dict = {}
        acc_arr = []
        f1_arr = []
        conf_matrix_arr = []
        ftrs_imp_arr = []

        test_periods_count = len(testTimes_)
        if print_log: print('test_periods_count= ', test_periods_count)

        for test in zip(testTimes_.index, testTimes_):
            if print_log: print('\ntest= {0}'.format(test))
            clf = XGBClassifier(max_depth=max_depth_, n_estimators=n_estimators_, n_jobs=n_jobs_)

            df_test_iter = df_test_.loc[(test[0] <= df_test_.index) & (df_test_.index <= test[1]), :]
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

        if print_log: print('\nacc_arr= ', acc_arr)
        acc_arr_mean = np.mean(acc_arr)
        acc_arr_std = np.std(acc_arr)
        if print_log: print('acc_arr_mean= {0:.5f}, acc_arr_std= {1:.5f}'.format(acc_arr_mean, acc_arr_std))
        res_dict['acc_score_mean'] = acc_arr_mean
        res_dict['acc_score_std'] = acc_arr_std
        res_dict['acc_score_arr'] = acc_arr
        if print_log: print('\nf1_arr= ', f1_arr)
        f1_arr_mean = np.mean(f1_arr)
        f1_arr_std = np.std(f1_arr)
        if print_log: print('f1_arr_mean= {0:.5f}, f1_arr_std= {1:.5f}'.format(f1_arr_mean, f1_arr_std))
        res_dict['f1_score_mean'] = f1_arr_mean
        res_dict['f1_score_std'] = f1_arr_std
        res_dict['f1_score_arr'] = f1_arr
        if print_log: print('f1_arr_mean= {0:.5f}, f1_arr_std= {1:.5f}'.format(f1_arr_mean, f1_arr_std))
        # ---
        res_dict['conf_matrix_arr'] = conf_matrix_arr
        # ---
        # print('\nftrs_imp_arr:\n', ftrs_imp_arr)
        res_dict['ftrs_imp_arr'] = ftrs_imp_arr

        features_imp_dict = {}
        for i in range(len(ftrs_imp_arr[0])):
            feature_name = ftrs_imp_arr[0][i][0]
            feature_name = str(feature_name).replace("b'", "").replace("'", "")
            feature_arr = [ftrs_imp_arr[my_iter][i][1] for my_iter in range(test_periods_count)]
            feature_arr_mean = np.mean(feature_arr)
            if print_log: print('feature_name= {0}, feature_arr= {1}, mean= {2:.5f}'.format(feature_name,
                                                                              feature_arr, feature_arr_mean))
            features_imp_dict[feature_name] = feature_arr_mean
        res_dict['features_imp_dict'] = features_imp_dict

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
        df_columns = ['acc_score_mean', 'acc_score_std', 'acc_score_arr', 'f1_score_mean', 'f1_score_std', 'f1_score_arr']
        df_columns.extend(self.features_arr)
        df_st = pd.DataFrame(index=np.arange(0, self.n_loops), columns=df_columns)
        # ---
        df_train = self.data_for_ml  # уменьшенное количество образцов за счёт отбора
        df_test = data_lbl  # все образцы
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

            for test in zip(self.testTimes.index, self.testTimes):
                print('\ntest= {0}'.format(test))
                clf = XGBClassifier(max_depth=3, n_estimators=100, n_jobs=-1)

                df_test_iter = df_test.loc[(test[0] <= df_test.index) & (df_test.index <= test[1]), :]
                df_train_iter = df_train.loc[df_train.index.difference(df_test_iter.index)]

                X_train_iter = df_train_iter.loc[:, features_for_ml]
                y_train_iter = df_train_iter.loc[:, self.target_clmn]
                X_test_iter = df_test_iter.loc[:, features_for_ml]
                y_test_iter = df_test_iter.loc[:, self.target_clmn]

                clf.fit(X_train_iter, y_train_iter)
                y_pred_iter = clf.predict(X_test_iter)

                acc = accuracy_score(y_test_iter, y_pred_iter)
                print('accuracy= {0:.5f}'.format(acc))
                acc_arr.append(acc)
                f1_scr = f1_score(y_test_iter, y_pred_iter, average='weighted')
                print('f1_score= {0:.5f}'.format(f1_scr))
                f1_arr.append(f1_scr)
                conf_matrix = confusion_matrix(y_test_iter, y_pred_iter)
                print('\nconf_matrix:\n{}'.format(conf_matrix))

                print("\nfeature_importances:")
                f_i = list(zip(features_for_ml, clf.feature_importances_))
                dtype = [('feature', 'S30'), ('importance', float)]
                f_i_nd = np.array(f_i, dtype=dtype)
                f_i_sort = np.sort(f_i_nd, order='feature')  # f_i_sort = np.sort(f_i_nd, order='importance')[::-1]
                f_i_arr = f_i_sort.tolist()
                ftrs_imp_arr.append(f_i_arr)
                for i, imp in enumerate(f_i_arr, 1):
                    print('{0}. {1:<20} {2:.5f}'.format(i, str(imp[0]).replace("b\'", "").replace("\'", ""), imp[1]))

            print('\nacc_arr= ', acc_arr)
            acc_arr_mean = np.mean(acc_arr)
            acc_arr_std = np.std(acc_arr)  # *len(acc_arr)**-.5
            print('acc_arr_mean= {0:.5f}, acc_arr_std= {1:.5f}'.format(acc_arr_mean, acc_arr_std))
            df_st.loc[df_st.index == step, 'acc_score_mean'] = acc_arr_mean
            df_st.loc[df_st.index == step, 'acc_score_std'] = acc_arr_std
            df_st.loc[df_st.index == step, 'acc_score_arr'] = str(acc_arr)
            print('\nf1_arr= ', f1_arr)
            f1_arr_mean = np.mean(f1_arr)
            f1_arr_std = np.std(f1_arr)  # *len(f1_arr)**-.5
            print('f1_arr_mean= {0:.5f}, f1_arr_std= {1:.5f}'.format(f1_arr_mean, f1_arr_std))
            df_st.loc[df_st.index == step, 'f1_score_mean'] = f1_arr_mean
            df_st.loc[df_st.index == step, 'f1_score_std'] = f1_arr_std
            df_st.loc[df_st.index == step, 'f1_score_arr'] = str(f1_arr)

            print('\nftrs_imp_arr:\n', ftrs_imp_arr)
            for i in range(len(ftrs_imp_arr[0])):
                feature_name = ftrs_imp_arr[0][i][0]
                feature_name = str(feature_name).replace("b'", "").replace("'", "")
                feature_arr = [ftrs_imp_arr[my_iter][i][1] for my_iter in range(test_periods_count)]
                feature_arr_mean = np.mean(feature_arr)
                print('feature_name= {0}, feature_arr= {1}, mean= {2:.5f}'.format(feature_name,
                                                                                  feature_arr, feature_arr_mean))
                df_st.loc[df_st.index == step, feature_name] = feature_arr_mean

            # сохранение дампа
            pckl = open(self.f_i_path_for_dump, "wb")
            pickle.dump(df_st, pckl)
            pckl.close()
            time_cur = dt.datetime.now()
            time_est = time_cur - time_start
            time_eta = (time_est/(step+1))*(self.n_loops-(step+1)) if (self.n_loops-(step+1)) != 0. else 0.
            print('\n{0:.2%} is done. time_eta= {1}'.format((step+1)/self.n_loops, time_eta))
            print('\n----------------------------------------------------------------------------------------------------\n')


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
        self.features_selection(data_lbl)
        #---

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))


if __name__ == '__main__':
    req = FeaturesSelectionClass()
    req.execute()