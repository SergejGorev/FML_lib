import features_selection, finfunctions
import numpy as np
import pandas as pd
from scipy.special import comb
import math
import copy
import pickle
import datetime as dt
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import os
import warnings

class EnsembleClass:
    n_classifiers = 21
    n_features_in_clf = 30
    major_features_part = 0.3
    major_features_arr = \
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
         'ema_72']
    minor_features_arr = \
        ['lr_duo_1152_1i5',
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

    folder_name = r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD"  #   r"/home/rom/01-Algorithmic_trading/02_1-EURUSD"
    data_pickle_file_name = "eurusd_5_v1.4.pickle"
    label_pickle_file_name = "eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"

    postfix = '_0i0025_1i0_1i0'
    version = 'v1.0'
    target_clmn_prefix = 'target_label'
    profit_value = 23
    loss_value = -25 #-25
    dump_ensemble_pickle = True # dump ensemble pickle
    ensemble_pickle_prefix = "ensemble_eurusd_5"
    data_for_ml_pickle_prefix = "data_for_ml_test"
    ens_ftrs_arr_pickle_prefix = "ens_ftrs_arr"
    ens_clf_arr_pickle_prefix = "ens_clf_arr"
    ens_pred_df_pickle_prefix = "ens_pred_df"
    ens_pred_statcs_pickle_prefix = "ens_pred_statcs"
    ftrs_major_arr_pickle_prefix = "ftrs_major_arr"
    ftrs_minor_arr_pickle_prefix = "ftrs_minor_arr"

    ftrs_gen_major_filename = "ftrs_gen_major_arr_v1.0.pickle"
    ftrs_gen_minor_filename = "ftrs_gen_minor_arr_v1.0.pickle"

    #---
    price_step = 0.001
    train_start = dt.datetime(2005, 1, 1, 0, 0)
    test_start = dt.datetime(2017, 7, 1, 0, 0)

    #---
    data_for_ml = None
    df_train = None
    df_test = None

    clf_arr = []
    ens_ftrs_arr = None
    ens_clf = None  # ensemble classifier
    #---
    pickle_postfix = ""
    data_pickle_path = ""
    label_pickle_path = ""
    target_clmn = ""
    ensemble_pickle_path = ""
    data_for_ml_pickle_path = ""
    ens_ftrs_arr_pickle_path = ""
    ens_clf_arr_pickle_path = ""
    ens_pred_df_pickle_path = ""
    ens_pred_statcs_pickle_path = ""
    ftrs_major_arr_pickle_path = ""
    ftrs_minor_arr_pickle_path = ""


    def __init__(self):
        self.pickle_postfix = self.postfix + '_' + self.version + '.pickle'
        self.data_pickle_path = self.folder_name + os.sep + self.data_pickle_file_name  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.4.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.4.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.4.pickle"
        self.label_pickle_path = self.folder_name + os.sep + self.label_pickle_file_name  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1_lbl_0i0025_1i0_1i0.pickle"

        self.target_clmn =  self.target_clmn_prefix + self.postfix  # 'target_label_0i0025_1i0_1i0'
        self.ensemble_pickle_path = self.folder_name + os.sep + self.ensemble_pickle_prefix + self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ensemble_eurusd_5_v1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/ensemble_eurusd_5_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ensemble_eurusd_5_v1.0.pickle"
        self.data_for_ml_pickle_path = self.folder_name + os.sep + self.data_for_ml_pickle_prefix + self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\data_for_ml_test_1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/data_for_ml_test_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\data_for_ml_test_1.0.pickle"
        self.ens_ftrs_arr_pickle_path = self.folder_name + os.sep + self.ens_ftrs_arr_pickle_prefix + self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_ftrs_arr_v1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/ens_ftrs_arr_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_ftrs_arr_v1.0.pickle"
        self.ens_clf_arr_pickle_path = self.folder_name + os.sep + self.ens_clf_arr_pickle_prefix + self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_clf_arr_v1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/ens_clf_arr_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_clf_arr_v1.0.pickle"
        self.ens_pred_df_pickle_path = self.folder_name + os.sep + self.ens_pred_df_pickle_prefix+ self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_pred_df_v1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/ens_pred_df_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_pred_df_v1.0.pickle"
        self.ens_pred_statcs_pickle_path = self.folder_name + os.sep + self.ens_pred_statcs_pickle_prefix + self.pickle_postfix  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_pred_statcs_v1.0.pickle"  # r"/home/rom/01-Algorithmic_trading/02_1-EURUSD/ens_pred_statcs_v1.0.pickle"  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\ens_pred_statcs_v1.0.pickle"
        self.ftrs_major_arr_pickle_path = self.folder_name + os.sep + self.ftrs_major_arr_pickle_prefix + self.pickle_postfix
        self.ftrs_minor_arr_pickle_path = self.folder_name + os.sep + self.ftrs_minor_arr_pickle_prefix + self.pickle_postfix
        # print('data_pickle_path= '+self.data_pickle_path)
        # print('label_pickle_path= ' + self.label_pickle_path)
        # print('target_clmn= ' + self.target_clmn)
        # print('ensemble_pickle_path= ' + self.ensemble_pickle_path)
        # print('data_for_ml_pickle_path= ' + self.data_for_ml_pickle_path)
        # print('ens_ftrs_arr_pickle_path= ' + self.ens_ftrs_arr_pickle_path)
        # print('ens_clf_arr_pickle_path= ' + self.ens_clf_arr_pickle_path)
        # print('ens_pred_df_pickle_path= ' + self.ens_pred_df_pickle_path)
        # print('ens_pred_statcs_pickle_path= ' + self.ens_pred_statcs_pickle_path)

    @staticmethod
    def ensemble_accuracy(n_classifiers, accuracy):
        """
        The function calculates ensemble accuracy based on the values of the basic accuracy of individual classifiers.

        Returns ensemble accuracy.

        Parameters
        ----------
        n_classifiers :  integer
            The number of individual classifiers.
        accuracy :  float
            The basic accuracy of individual classifiers.

        Returns
        -------
        ensemble_accuracy : float
            Ensemble accuracy value.
        """
        k_start = int(math.ceil(n_classifiers / 2.0))
        probs = [comb(n_classifiers, k) *
                 accuracy**k *
                 (1 - accuracy)**(n_classifiers - k)
                 for k in range(k_start, n_classifiers + 1)]
        return sum(probs)


    def import_features_array(self, use_generalized_ftrs_arr=True, print_log=True):
        if use_generalized_ftrs_arr:
            self.ftrs_major_arr_pickle_path = self.folder_name + os.sep + self.ftrs_gen_major_filename
            self.ftrs_minor_arr_pickle_path = self.folder_name + os.sep + self.ftrs_gen_minor_filename

        with open(self.ftrs_major_arr_pickle_path, "rb") as pckl:
            self.major_features_arr = pickle.load(pckl)
        if print_log: print('major_features_arr:\n {0}'.format(self.major_features_arr))
        with open(self.ftrs_minor_arr_pickle_path, "rb") as pckl:
            self.minor_features_arr = pickle.load(pckl)
        if print_log: print('minor_features_arr:\n {0}'.format(self.minor_features_arr))


    @staticmethod
    def features_distribution(n_classifiers, n_features_in_clf, major_features_part,
                              major_features_arr, minor_features_arr,
                              save_dump=False, dump_file_path=r'ens_ftrs_arr.pickle',
                              print_log=False):
        """
        Function for features distribution between the classifiers.

        :param n_classifiers: integer.
            The number of basic classifiers.
        :param n_features_in_clf: integer.
            The number of features in basic classifier.
        :param major_features_part: float in range [0., 1.].
            The part of major features in the total number of features.
        :param major_features_arr: array.
            Array of more predictible effective features.
        :param minor_features_arr: array.
            Array of less predictible effective features.
        :param save_dump: boolean.
            The need to dump of features array.
        :param dump_file_path: string.
            Dump file path.
        :param print_log: boolean.
            The need to print the log.

        :return:
            Array of size (n_classifiers, n_features_in_clf) of features for basic classifiers.
        """

        feat_arr = []
        for i in range(n_classifiers):
            features_set = []
            features_for_select = copy.deepcopy(major_features_arr)
            n_major_feat = int(n_features_in_clf*major_features_part)
            n_minor_feat = n_features_in_clf - n_major_feat
            if print_log: print('n_major_feat= {0}, n_minor_feat= {1}'.format(n_major_feat, n_minor_feat))
            #--- major features selection
            for i in range(n_major_feat):
                sel_len = len(features_for_select)
                rnd = np.random.randint(0, sel_len)
                feat_to_add = features_for_select[rnd]
                if print_log: print('sel_len= {0}, rnd= {1}, feat_to_add= {2}'.format(sel_len, rnd, feat_to_add))
                features_set.append(feat_to_add)
                features_for_select.pop(rnd)

            if print_log: print('major features_set:\n', features_set)
            #--- minor features selection
            minor_features_set = []
            features_for_select = copy.deepcopy(minor_features_arr)
            for i in range(n_minor_feat):
                sel_len = len(features_for_select)
                rnd = np.random.randint(0, sel_len)
                feat_to_add = features_for_select[rnd]
                if print_log: print('sel_len= {0}, rnd= {1}, feat_to_add= {2}'.format(sel_len, rnd, feat_to_add))
                minor_features_set.append(feat_to_add)
                features_for_select.pop(rnd)

            if print_log: print('minor features_set:\n', minor_features_set)
            #--- full features array
            features_set.extend(minor_features_set)
            if print_log: print('full features_set:\n', features_set)
            feat_arr.append(features_set)
            if print_log: print('-------------------------------------------------------------------------------------\n')
        if print_log: print('feat_arr:\n', feat_arr)
        if print_log: print('feat_arr shape= [{0}, {1}]'.format(len(feat_arr), len(feat_arr[0])))
        #--- array dumping
        if save_dump:
            with open(dump_file_path, "wb") as pckl:
                pickle.dump(feat_arr, pckl)
            print('\nFeatures array dump (\'{0}\') is saved.'.format(dump_file_path))
        #---
        return feat_arr


    def data_preparation(self, use_data_for_ml_dump=False, save_data_for_ml_dump=False, print_log=True):
        # --- dataframe load
        time_start = dt.datetime.now()
        if print_log: print('time_start= {}'.format(time_start))

        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)
        if print_log: print('\ndata.shape: ', data.shape)

        with open(self.label_pickle_path, "rb") as pckl:
            lbl = pickle.load(pckl)
            #--- replacing column names
            label_buy, label_sell = 'label_buy' + self.postfix, 'label_sell' + self.postfix
            lbl['label_buy'] = lbl[label_buy]
            lbl['label_sell'] = lbl[label_sell]
            lbl.drop(columns=[label_buy, label_sell], inplace=True)
        if print_log: print('lbl.shape: ', lbl.shape)

        data_lbl = pd.concat((data, lbl), axis=1)
        print('data_lbl.shape: ', data_lbl.shape)

        self.last_clmn = data.shape[1]
        if print_log: print('self.last_clmn= ', self.last_clmn)
        if print_log: print('last 5 columns: ', data_lbl.columns[self.last_clmn - 5: self.last_clmn])
        # ---
        del data
        del lbl

        if use_data_for_ml_dump:
            # ---
            # загрузка датафрейма в тестовых целях
            with open(self.data_for_ml_pickle_path, "rb") as pckl:
                data_for_ml = pickle.load(pckl)
            self.data_for_ml = data_for_ml
            # ---
        else:
            self.data_for_ml = features_selection.FeaturesSelectionClass.select_data_for_ml(
                                    data_lbl=data_lbl, price_step=self.price_step, target_clmn=self.target_clmn)

        if save_data_for_ml_dump:
            with open(self.data_for_ml_pickle_path, "wb") as pckl:
                pickle.dump(self.data_for_ml, pckl)

        df_test = data_lbl
        df_train = self.data_for_ml
        self.df_test = df_test.loc[df_test.index >= self.test_start, :]
        del df_test
        del data_lbl
        self.df_train = df_train.loc[(df_train.index>=self.train_start) & (df_train.index<=self.test_start), :]
        del df_train


    def ensemble_fit(self, n_classifiers, max_depth=3, n_estimators=100, n_jobs=-1,
                     use_ens_ftrs_arr_dump=False, save_ens_clf_arr=True,
                     use_data_for_ml_dump=False, save_data_for_ml_dump=False, print_log=True):
        """
        Create ensemble of fitted classifiers.

        :return:
            None
        """
        #--- data preparation
        self.data_preparation(use_data_for_ml_dump=use_data_for_ml_dump, save_data_for_ml_dump=save_data_for_ml_dump,
                              print_log=print_log)
        #--- features loading
        if use_ens_ftrs_arr_dump:
            with open(self.ens_ftrs_arr_pickle_path, "rb") as pckl:
                self.ens_ftrs_arr = pickle.load(pckl)
        else:
            self.ens_ftrs_arr = self.features_distribution(n_classifiers=self.n_classifiers, n_features_in_clf=self.n_features_in_clf,
                            major_features_part=self.major_features_part, major_features_arr=self.major_features_arr,
                            minor_features_arr=self.minor_features_arr, print_log=False, save_dump=True,
                            dump_file_path=self.ens_ftrs_arr_pickle_path)
        #---
        #--- classifiers training cycle
        basic_clf = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, n_jobs=n_jobs)

        #---
        if print_log: print()
        for i in range(n_classifiers):
            features_for_ml = self.ens_ftrs_arr[i]
            if print_log: print('i= {0}, features= {1}'.format(i, features_for_ml))
            X_train_iter = self.df_train.loc[:, features_for_ml]
            y_train_iter = self.df_train.loc[:, self.target_clmn]

            basic_clf.fit(X_train_iter, y_train_iter)
            clf_copy = copy.deepcopy(basic_clf)
            self.clf_arr.append(clf_copy)
            #---
        if save_ens_clf_arr:
            with open(self.ens_clf_arr_pickle_path, "wb") as pckl:
                pickle.dump(self.clf_arr, pckl)
        #---


    def ensemble_predict(self, n_classifiers, use_ens_clf_arr_dump=False, save_pred_df=True,
                         save_pred_statistics=True, print_log=True):
        if self.ens_ftrs_arr is None:
            with open(self.ens_ftrs_arr_pickle_path, "rb") as pckl:
                self.ens_ftrs_arr = pickle.load(pckl)
            if print_log: print('\nens_ftrs_arr has been loaded.')

        if (self.clf_arr is None) or use_ens_clf_arr_dump:
            with open(self.ens_clf_arr_pickle_path, "rb") as pckl:
                self.ens_clf = pickle.load(pckl)
            if print_log: print('\nens_clf has been loaded.')
        if self.df_test is None:
            self.data_preparation(use_data_for_ml_dump=True, save_data_for_ml_dump=False, print_log=print_log)

        df_pred = pd.DataFrame(index=self.df_test.index)
        acc_arr = []
        f1_arr = []
        conf_matrix_arr = []
        ftrs_imp_arr = []
        rtrn_arr = []
        sr_arr = []
        for i in range(n_classifiers):
            features_for_ml = self.ens_ftrs_arr[i]
            X_test_iter = self.df_test.loc[:, features_for_ml]
            y_test_iter = self.df_test.loc[:, self.target_clmn]

            clf = self.ens_clf[i]

            y_pred_iter = clf.predict(X_test_iter)
            df_pred['pred_'+str(i)] = y_pred_iter

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
            f_i = list(zip(features_for_ml, clf.feature_importances_))
            dtype = [('feature', 'S30'), ('importance', float)]
            f_i_nd = np.array(f_i, dtype=dtype)
            f_i_sort = np.sort(f_i_nd, order='feature')  # f_i_sort = np.sort(f_i_nd, order='importance')[::-1]
            f_i_arr = f_i_sort.tolist()
            ftrs_imp_arr.append(f_i_arr)
            if print_log:
                for i, imp in enumerate(f_i_arr, 1):
                    print('{0}. {1:<30} {2:.5f}'.format(i, str(imp[0]).replace("b\'", "").replace("\'", ""), imp[1]))
            # --- financial statistics calculation
            y_pred_series = pd.Series(data=y_pred_iter, index=self.df_test.index)
            fin_res = finfunctions.pred_fin_res(y_pred=y_pred_series, label_buy=self.df_test['label_buy'],
                                                label_sell=self.df_test['label_sell'], profit_value=self.profit_value,
                                                loss_value=self.loss_value)
            rtrn_arr.append(fin_res[0])
            sr_arr.append(fin_res[1])
            if print_log:
                print('return= {0:.2f}, SR= {1:.4f}'.format(fin_res[0], fin_res[1]))
                print('-------------------------------------------------------------------------------\n')
        if save_pred_df:
            with open(self.ens_pred_df_pickle_path, "wb") as pckl:
                pickle.dump(df_pred, pckl)
        if save_pred_statistics:
            pred_statcs_df = pd.DataFrame(data=list(zip(acc_arr, f1_arr, rtrn_arr, sr_arr, conf_matrix_arr,
                                        ftrs_imp_arr)), columns=['accuracy', 'f1', 'return', 'sharpe', 'conf_matrix',
                                                                 'ftrs_importance'])
            if print_log: print('pred_statcs_df:\n', pred_statcs_df)
            with open(self.ens_pred_statcs_pickle_path, "wb") as pckl:
                pickle.dump(pred_statcs_df, pckl)


if __name__ == '__main__':
    # #--- расчёт требуемого количества классификаторов в ансамбле
    # n_classifiers = 21
    # basic_accuracy = 0.53
    # ens_acc = EnsembleClass.ensemble_accuracy(n_classifiers=n_classifiers, accuracy=basic_accuracy)
    # print('\nEnsemble accuracy for {0} classifiers with basic accuracy {1:.4f} = {2:.4f}'.format(n_classifiers,
    #                                                                                             basic_accuracy, ens_acc))
    #---
    time_start = dt.datetime.now()
    print('time_start= {}'.format(time_start))

    req = EnsembleClass()
    req.import_features_array(use_generalized_ftrs_arr=True)
    feat_arr = req.features_distribution(n_classifiers=req.n_classifiers, n_features_in_clf=req.n_features_in_clf,
                              major_features_part=req.major_features_part, major_features_arr=req.major_features_arr,
                              minor_features_arr=req.minor_features_arr, save_dump=True,
                              dump_file_path=req.ens_ftrs_arr_pickle_path, print_log=True)
    print("\nfeat_arr:\n", feat_arr)

    req.ensemble_fit(n_classifiers=req.n_classifiers, use_ens_ftrs_arr_dump=True, save_ens_clf_arr=True,
                     use_data_for_ml_dump=True, save_data_for_ml_dump=False, print_log=True)

    req.ensemble_predict(n_classifiers=req.n_classifiers, use_ens_clf_arr_dump=True, save_pred_df=True,
                         save_pred_statistics=True, print_log=True)

    time_finish = dt.datetime.now()
    time_duration = time_finish - time_start
    print('time_finish= {0}, duration= {1}'.format(time_finish, time_duration))
