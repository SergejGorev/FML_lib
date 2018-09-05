from xgboost import XGBClassifier
import warnings
import finfunctions
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from matplotlib import pyplot as plt


def visualize_1_sample_clf(n_estimators_, max_depth_,
                         df_train_, df_test_,
                         test_period_,
                         objective_, features_,
                         pl_buy_clmn_, pl_sell_clmn_,
                         test_folder_name_,
                         target_clmn_='target_label',
                         threshold_m_=0.7, threshold_p_=0.7,
                         save_pic=False, excel_export=False, plot_suptitle_='Test result',
                         minimalistic_mode=False):
    '''
    df_train - датафрейм с выборочными данными
    df_test  - датафрейм с полными данными
    '''
    rtrn_digit = 2

    score_prelim_res = []
    score_opt_res = []
    SR_res = []
    warnings.filterwarnings('ignore')

    clf = XGBClassifier(n_jobs=-1, n_estimators=n_estimators_, max_depth=max_depth_, objective=objective_,
                        booster='gbtree')
    # if minimalistic_mode == False:
    #print('\ntest[0]= {0}, test[1]= {1}'.format(test[0], test[1]))
    # --- определяем данные для обучения и тестирования
    #print("\ndf_test_.sample(10).index:\n", df_test_.sample(10).index)
    df_test_iter = df_test_.loc[(test_period_[0] <= df_test_.index) & (df_test_.index <= test_period_[1]), :]
    df_train_iter = df_train_.loc[df_train_.index.difference(df_test_iter.index)]
    X_train_iter = df_train_iter.loc[:, features_]
    y_train_iter = df_train_iter.loc[:, target_clmn_]
    X_test_iter = df_test_iter.loc[:, features_]
    y_test_iter = df_test_iter.loc[:, target_clmn_]
    # ---
    clf.fit(X=X_train_iter, y=y_train_iter)
    # формируем предсказание для train
    y_train_pred = clf.predict(X_train_iter)
    y_train_proba = clf.predict_proba(X_train_iter)
    pred_proba_train_1m = [x[0] for x in y_train_proba]
    pred_proba_train_1p = [x[1] for x in y_train_proba]
    m_train_mean = np.mean(pred_proba_train_1m)
    p_train_mean = np.mean(pred_proba_train_1p)
    m_train_std = np.std(pred_proba_train_1m)
    p_train_std = np.std(pred_proba_train_1p)
    m_train_min, m_train_max = np.min(pred_proba_train_1m), np.max(pred_proba_train_1m)
    p_train_min, p_train_max = np.min(pred_proba_train_1p), np.max(pred_proba_train_1p)
    m_train_25, m_train_50, m_train_75 = np.percentile(pred_proba_train_1m, (25, 50, 75))
    p_train_25, p_train_50, p_train_75 = np.percentile(pred_proba_train_1p, (25, 50, 75))

    #koef_p_m = p_train_50 / m_train_50
    # ---
    # на основании предсказания рассчитываем финансовые показатели
    df_train_iter['pred'] = y_train_pred
    rtrn = list(map(finfunctions.check_pred_r, zip(y_train_pred, df_train_iter[pl_buy_clmn_].values,
                                      df_train_iter[pl_sell_clmn_].values)))
    df_train_iter['pred_dir'] = y_train_pred
    df_train_iter['return_train'] = rtrn
    df_train_iter['return_train_cumsum'] = df_train_iter['return_train'].cumsum()

    # формируем предсказание для test
    y_pred = clf.predict(X_test_iter)
    y_pred_proba = clf.predict_proba(X_test_iter)
    score_prelim = f1_score(y_test_iter, y_pred)
    score_prelim_res.append(score_prelim)
    conf_matrix = confusion_matrix(y_test_iter, y_pred)
    rtrn = list(
        map(finfunctions.check_pred_r, zip(y_pred, df_test_iter[pl_buy_clmn_].values, df_test_iter[pl_sell_clmn_].values)))
    rtrn_sum = np.sum(rtrn)
    SR_test_prelim = finfunctions.sharpe_ratio(df_test_iter.index.values, rtrn)
    # на основании предсказания рассчитываем финансовые показатели
    thr_p = threshold_p_
    thr_m = threshold_m_
    pred_dir_func = lambda x: -1 if ((x[0] > x[1]) & (x[0] >= thr_m)) else \
        (1 if ((x[1] > x[0]) & (x[1] >= thr_p)) else 0)
    pred_proba_1m = [x[0] for x in y_pred_proba]
    pred_proba_1p = [x[1] for x in y_pred_proba]

    pred_dir = list(map(pred_dir_func, zip(pred_proba_1m, pred_proba_1p)))
    score_opt = f1_score(y_test_iter, pred_dir)
    conf_matrix_opt = confusion_matrix(y_test_iter, pred_dir)
    rtrn_opt = list(
        map(finfunctions.check_pred_r, zip(pred_dir, df_test_iter[pl_buy_clmn_].values, df_test_iter[pl_sell_clmn_].values)))
    rtrn_opt_sum = np.sum(rtrn_opt)
    SR_opt = finfunctions.sharpe_ratio(df_test_iter.index.values, rtrn_opt)
    SR_res.append(SR_opt)

    df_test_iter['pred'] = pred_dir
    df_test_iter['pred_dir'] = pred_dir
    df_test_iter['return_test'] = rtrn_opt
    df_test_iter['return_test_cumsum'] = df_test_iter['return_test'].cumsum()
    ##---
    SR_train = finfunctions.sharpe_ratio(df_train_iter.index.values, df_train_iter['return_train'].values)
    train_return = df_train_iter['return_train'].sum()
    SR_test = finfunctions.sharpe_ratio(df_test_iter.index.values, df_test_iter['return_test'].values)
    test_return = df_test_iter['return_test'].sum()
    SR_relation = 0 if SR_train == 0.0 else SR_test / SR_train
    # ---

    if minimalistic_mode == False:
        # --- train
        plt.figure(figsize=(15, 18))
        plt.subplot(3, 1, 1)
        df_adv = pd.concat([df_train_iter, df_test_iter])
        # print('\nduplicates:\n', df_adv[df_adv.index.duplicated()])  # выводим дубликаты
        # print('df_train_.shape= {0}, df_test.shape= {1}, df_adv.shape= {2}'.format(df_train.shape,
        #                                                                          df_test.shape, df_adv.shape))
        df_adv['open_train'] = df_train_iter.open
        df_adv['open_test'] = df_test_iter.open
        # ---
        df_adv.open_train.plot()
        df_adv.open_test.plot(color='r')
        plt.ylabel('audjpy', size=14)
        plt.grid(True)
        # ---
        plt.subplot(3, 1, 2)
        df_adv.return_train_cumsum.plot()
        df_adv.return_test_cumsum.plot(color='r')
        plt.ylabel('Return', size=14)
        plt.grid(True)
        text_to_pic = 'Train: SR= {0:.4f}, rtrn= {1:.{9}f}\nTest:  SR= {2:.4f}, rtrn= {3:.{9}f}\
                            \nSR test/train relation=   {4:.4f}\n\nn_est= {5}\nmax_depth= {6}\
                            \nthr_m= {7:.2f}, thr_m= {8:.2f}'.format(
            SR_train, train_return,
            SR_test, test_return, SR_relation, \
            n_estimators_, max_depth_, \
            threshold_m_, threshold_p_, \
            rtrn_digit)
        # ---
        x_ = df_adv.index.min()
        y_min = np.min(
            (np.min(df_adv.return_train_cumsum.dropna().values), np.min(df_adv.return_test_cumsum.dropna().values)))
        y_max = np.max(
            (np.max(df_adv.return_train_cumsum.dropna().values), np.max(df_adv.return_test_cumsum.dropna().values)))
        # print('y_min= {0}, y_max= {1}'.format(y_min, y_max))
        y_ = (y_max - y_min) * 0.5 + y_min
        plt.text(x=x_, y=y_, s=text_to_pic)

        plt.subplot(3, 1, 3)
        df_test_iter['return_test_cumsum'].plot(color='r')
        plt.ylabel('Return (TEST)', size=14)
        plt.grid(True)
        plt.suptitle(plot_suptitle_, size=16)
        # ---
        if (save_pic):
            # obj = lambda x: 'soft' if x=='multi:softprob' else 'custom'
            file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                        str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                        str(threshold_m_) + str(threshold_p_) + ".png"
            # +'_'+str(timeperiod_bb)+'_'+ \
            # str(nbdev)+'_'+str(timeperiod_al)+'_'+str(mult_al)+'_'+str(max_lag)+'_'+str(timeperiod_h)+'_'+ \
            # str(timeperiod_lr)+'_'+str(mult_lr)
            plt.savefig(file_name, format='png', dpi=100)
        # ---
        plt.show()
        print(text_to_pic)
        print("               '-1'    '+1'".format())
        print("train_mean:  {0:.4f}  {1:.4f}".format(m_train_mean, p_train_mean))
        print("train_std:   {0:.4f}  {1:.4f}".format(m_train_std, p_train_std))
        print("train_min:   {0:.4f}  {1:.4f}".format(m_train_min, p_train_min))
        print("train_25%:   {0:.4f}  {1:.4f}".format(m_train_25, p_train_25))
        print("train_50%:   {0:.4f}  {1:.4f}".format(m_train_50, p_train_50))
        print("train_75%:   {0:.4f}  {1:.4f}".format(m_train_75, p_train_75))
        print("train_max:   {0:.4f}  {1:.4f}".format(m_train_max, p_train_max))

        print('\nscore= {0:.6f}'.format(score_prelim))
        print('conf_matrix:\n{}'.format(conf_matrix))
        print('rtrn= {0:.2f}, SR= {1:.4f}'.format(rtrn_sum, SR_test_prelim))
        print('\nscore_opt= {0:.6f}'.format(score_opt))
        print('conf_matrix_opt:\n{}'.format(conf_matrix_opt))
        print("\nfeature_importances:")
        f_i = list(zip(features_, clf.feature_importances_))
        dtype = [('feature', 'S15'), ('importance', float)]
        f_i_nd = np.array(f_i, dtype=dtype)
        f_i_sort = np.sort(f_i_nd, order='importance')[::-1]
        print(f_i_sort.tolist())
        print(
            '-------------------------------------------------------------------------------------------------------')
        if (excel_export == True):
            file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                        str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                        "_train.xlsx"
            writer = pd.ExcelWriter(file_name)
            df_train_iter.to_excel(writer, 'data')
            writer.save()

            file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                        str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                        "_test.xlsx"
            writer = pd.ExcelWriter(file_name)
            df_test_iter.to_excel(writer, 'data')
            writer.save()

    warnings.filterwarnings('default')

    return [score_prelim_res, score_opt_res, SR_res]



def cvScoreWithSelection(n_estimators_, max_depth_,
                         df_train_, df_test_,
                         testTimes_,
                         objective_, features_,
                         test_folder_name_,
                         pl_buy_clmn_, pl_sell_clmn_,
                         target_clmn_='target_label',
                         threshold_m_=0.7, threshold_p_=0.7,
                         save_pic=False, excel_export=False, plot_suptitle_='Test result',
                         minimalistic_mode=False):
    '''
    df_train - датафрейм с выборочными данными
    df_test  - датафрейм с полными данными
    '''
    rtrn_digit = 2

    score_prelim_res = []
    score_opt_res = []
    SR_res = []
    warnings.filterwarnings('ignore')
    for test in zip(testTimes_.index, testTimes_):
        clf = XGBClassifier(n_jobs=1, n_estimators=n_estimators_, max_depth=max_depth_, objective=objective_,
                            booster='gbtree')
        # if minimalistic_mode == False:
        #print('\ntest[0]= {0}, test[1]= {1}'.format(test[0], test[1]))
        # --- определяем данные для обучения и тестирования
        #print("\ndf_test_.sample(10).index:\n", df_test_.sample(10).index)
        df_test_iter = df_test_.loc[(test[0] <= df_test_.index) & (df_test_.index <= test[1]), :]
        df_train_iter = df_train_.loc[df_train_.index.difference(df_test_iter.index)]
        X_train_iter = df_train_iter.loc[:, features_]
        y_train_iter = df_train_iter.loc[:, target_clmn_]
        X_test_iter = df_test_iter.loc[:, features_]
        y_test_iter = df_test_iter.loc[:, target_clmn_]
        # ---
        clf.fit(X=X_train_iter, y=y_train_iter)
        # формируем предсказание для train
        y_train_pred = clf.predict(X_train_iter)
        y_train_proba = clf.predict_proba(X_train_iter)
        pred_proba_train_1m = [x[0] for x in y_train_proba]
        pred_proba_train_1p = [x[1] for x in y_train_proba]
        m_train_mean = np.mean(pred_proba_train_1m)
        p_train_mean = np.mean(pred_proba_train_1p)
        m_train_std = np.std(pred_proba_train_1m)
        p_train_std = np.std(pred_proba_train_1p)
        m_train_min, m_train_max = np.min(pred_proba_train_1m), np.max(pred_proba_train_1m)
        p_train_min, p_train_max = np.min(pred_proba_train_1p), np.max(pred_proba_train_1p)
        m_train_25, m_train_50, m_train_75 = np.percentile(pred_proba_train_1m, (25, 50, 75))
        p_train_25, p_train_50, p_train_75 = np.percentile(pred_proba_train_1p, (25, 50, 75))

        koef_p_m = p_train_50 / m_train_50
        # ---
        # на основании предсказания рассчитываем финансовые показатели
        df_train_iter['pred'] = y_train_pred
        rtrn = list(map(finfunctions.check_pred_r, zip(y_train_pred, df_train_iter[pl_buy_clmn_].values,
                                          df_train_iter[pl_sell_clmn_].values)))
        df_train_iter['pred_dir'] = y_train_pred
        df_train_iter['return_train'] = rtrn
        df_train_iter['return_train_cumsum'] = df_train_iter['return_train'].cumsum()

        # формируем предсказание для test
        y_pred = clf.predict(X_test_iter)
        y_pred_proba = clf.predict_proba(X_test_iter)
        score_prelim = f1_score(y_test_iter, y_pred)
        score_prelim_res.append(score_prelim)
        conf_matrix = confusion_matrix(y_test_iter, y_pred)
        rtrn = list(
            map(finfunctions.check_pred_r, zip(y_pred, df_test_iter[pl_buy_clmn_].values, df_test_iter[pl_sell_clmn_].values)))
        rtrn_sum = np.sum(rtrn)
        SR_test_prelim = finfunctions.sharpe_ratio(df_test_iter.index.values, rtrn)
        # на основании предсказания рассчитываем финансовые показатели
        thr_p = threshold_p_
        thr_m = threshold_m_
        pred_dir_func = lambda x: -1 if ((x[0] > x[1]) & (x[0] >= thr_m)) else \
            (1 if ((x[1] > x[0]) & (x[1] >= thr_p)) else 0)
        pred_proba_1m = [x[0] for x in y_pred_proba]
        pred_proba_1p = [x[1] for x in y_pred_proba]

        pred_dir = list(map(pred_dir_func, zip(pred_proba_1m, pred_proba_1p)))
        score_opt = f1_score(y_test_iter, pred_dir)
        conf_matrix_opt = confusion_matrix(y_test_iter, pred_dir)
        rtrn_opt = list(
            map(finfunctions.check_pred_r, zip(pred_dir, df_test_iter[pl_buy_clmn_].values, df_test_iter[pl_sell_clmn_].values)))
        rtrn_opt_sum = np.sum(rtrn_opt)
        SR_opt = finfunctions.sharpe_ratio(df_test_iter.index.values, rtrn_opt)
        SR_res.append(SR_opt)

        df_test_iter['pred'] = pred_dir
        df_test_iter['pred_dir'] = pred_dir
        df_test_iter['return_test'] = rtrn_opt
        df_test_iter['return_test_cumsum'] = df_test_iter['return_test'].cumsum()
        ##---
        SR_train = finfunctions.sharpe_ratio(df_train_iter.index.values, df_train_iter['return_train'].values)
        train_return = df_train_iter['return_train'].sum()
        SR_test = finfunctions.sharpe_ratio(df_test_iter.index.values, df_test_iter['return_test'].values)
        test_return = df_test_iter['return_test'].sum()
        SR_relation = 0 if SR_train == 0.0 else SR_test / SR_train
        # ---

        if minimalistic_mode == False:
            # --- train
            plt.figure(figsize=(15, 18))
            plt.subplot(3, 1, 1)
            df_adv = pd.concat([df_train_iter, df_test_iter])
            # print('\nduplicates:\n', df_adv[df_adv.index.duplicated()])  # выводим дубликаты
            # print('df_train_.shape= {0}, df_test.shape= {1}, df_adv.shape= {2}'.format(df_train.shape,
            #                                                                          df_test.shape, df_adv.shape))
            df_adv['open_train'] = df_train_iter.open
            df_adv['open_test'] = df_test_iter.open
            # ---
            df_adv.open_train.plot()
            df_adv.open_test.plot(color='r')
            plt.ylabel('audjpy', size=14)
            plt.grid(True)
            # ---
            plt.subplot(3, 1, 2)
            df_adv.return_train_cumsum.plot()
            df_adv.return_test_cumsum.plot(color='r')
            plt.ylabel('Return', size=14)
            plt.grid(True)
            text_to_pic = 'Train: SR= {0:.4f}, rtrn= {1:.{9}f}\nTest:  SR= {2:.4f}, rtrn= {3:.{9}f}\
                                \nSR test/train relation=   {4:.4f}\n\nn_est= {5}\nmax_depth= {6}\
                                \nthr_m= {7:.2f}, thr_m= {8:.2f}'.format(
                SR_train, train_return,
                SR_test, test_return, SR_relation, \
                n_estimators_, max_depth_, \
                threshold_m_, threshold_p_, \
                rtrn_digit)
            # ---
            x_ = df_adv.index.min()
            y_min = np.min(
                (np.min(df_adv.return_train_cumsum.dropna().values), np.min(df_adv.return_test_cumsum.dropna().values)))
            y_max = np.max(
                (np.max(df_adv.return_train_cumsum.dropna().values), np.max(df_adv.return_test_cumsum.dropna().values)))
            # print('y_min= {0}, y_max= {1}'.format(y_min, y_max))
            y_ = (y_max - y_min) * 0.5 + y_min
            plt.text(x=x_, y=y_, s=text_to_pic)

            plt.subplot(3, 1, 3)
            df_test_iter['return_test_cumsum'].plot(color='r')
            plt.ylabel('Return (TEST)', size=14)
            plt.grid(True)
            plt.suptitle(plot_suptitle_, size=16)
            # ---
            if (save_pic):
                # obj = lambda x: 'soft' if x=='multi:softprob' else 'custom'
                file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                            str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                            str(threshold_m_) + str(threshold_p_) + ".png"
                # +'_'+str(timeperiod_bb)+'_'+ \
                # str(nbdev)+'_'+str(timeperiod_al)+'_'+str(mult_al)+'_'+str(max_lag)+'_'+str(timeperiod_h)+'_'+ \
                # str(timeperiod_lr)+'_'+str(mult_lr)
                plt.savefig(file_name, format='png', dpi=100)
            # ---
            plt.show()
            print(text_to_pic)
            print("               '-1'    '+1'".format())
            print("train_mean:  {0:.4f}  {1:.4f}".format(m_train_mean, p_train_mean))
            print("train_std:   {0:.4f}  {1:.4f}".format(m_train_std, p_train_std))
            print("train_min:   {0:.4f}  {1:.4f}".format(m_train_min, p_train_min))
            print("train_25%:   {0:.4f}  {1:.4f}".format(m_train_25, p_train_25))
            print("train_50%:   {0:.4f}  {1:.4f}".format(m_train_50, p_train_50))
            print("train_75%:   {0:.4f}  {1:.4f}".format(m_train_75, p_train_75))
            print("train_max:   {0:.4f}  {1:.4f}".format(m_train_max, p_train_max))

            print('\nscore= {0:.6f}'.format(score_prelim))
            print('conf_matrix:\n{}'.format(conf_matrix))
            print('rtrn= {0:.2f}, SR= {1:.4f}'.format(rtrn_sum, SR_test_prelim))
            print('\nscore_opt= {0:.6f}'.format(score_opt))
            print('conf_matrix_opt:\n{}'.format(conf_matrix_opt))
            print("\nfeature_importances:")
            f_i = list(zip(features_, clf.feature_importances_))
            dtype = [('feature', 'S15'), ('importance', float)]
            f_i_nd = np.array(f_i, dtype=dtype)
            f_i_sort = np.sort(f_i_nd, order='importance')[::-1]
            print(f_i_sort.tolist())
            print(
                '-------------------------------------------------------------------------------------------------------')
            if (excel_export == True):
                file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                            str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                            "_train.xlsx"
                writer = pd.ExcelWriter(file_name)
                df_train_iter.to_excel(writer, 'data')
                writer.save()

                file_name = test_folder_name_ + 'Rtrn_SR_' + str(round(SR_test, 2)) + '_' + \
                            str(n_estimators_) + '_' + str(max_depth_) + '_' + \
                            "_test.xlsx"
                writer = pd.ExcelWriter(file_name)
                df_test_iter.to_excel(writer, 'data')
                writer.save()

    warnings.filterwarnings('default')

    return [score_prelim_res, score_opt_res, SR_res]