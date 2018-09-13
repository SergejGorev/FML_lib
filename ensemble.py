import numpy as np
from scipy.special import comb
import math
import copy
import pickle
from sklearn.ensemble import VotingClassifier

class EnsembleClass:
    n_classifier = 21
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

    clf_arr = []

    @staticmethod
    def ensemble_accuracy(n_classifier, accuracy):
        """
        The function calculates ensemble accuracy based on the values of the basic accuracy of individual classifiers.

        Returns ensemble accuracy.

        Parameters
        ----------
        n_classifier :  integer
            The number of individual classifiers.
        accuracy :  float
            The basic accuracy of individual classifiers.

        Returns
        -------
        ensemble_accuracy : float
            Ensemble accuracy value.
        """
        k_start = int(math.ceil(n_classifier / 2.0))
        probs = [comb(n_classifier, k) *
                 accuracy**k *
                 (1 - accuracy)**(n_classifier - k)
                 for k in range(k_start, n_classifier + 1)]
        return sum(probs)


    @staticmethod
    def feat_distr(n_classifier, n_features_in_clf, major_features_part, major_features_arr, minor_features_arr,
                   print_log=False, save_dump=False, dump_file_path=r'ensemble_features_array.pickle'):
        """
        Function for features distribution between the classifiers.

        :param n_classifier: integer.
            The number of basic classifiers.
        :param n_features_in_clf: integer.
            The number of features in basic classifier.
        :param major_features_part: float in range [0., 1.].
            The part of major features in the total number of features.
        :param major_features_arr: array.
            Array of more predictible effective features.
        :param minor_features_arr: array.
            Array of less predictible effective features.
        :param print_log: boolean.
            The need to print the log.
        :param save_dump: boolean.
            The need to dump of features array.
        :param dump_file_path: string.
            Dump file path.

        :return:
            Array of size (n_classifier, n_features_in_clf) of features for basic classifiers.
        """
        feat_arr = []
        for i in range(n_classifier):
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
            pckl = open(dump_file_path, "wb")
            pickle.dump(feat_arr, pckl)
            pckl.close()
            print('\nFeatures array dump (\'{0}\') is saved.'.format(dump_file_path))
        #---
        return feat_arr


if __name__ == '__main__':
    # #--- расчёт требуемого количества классификаторов в ансамбле
    # n_classifier = 21
    # basic_accuracy = 0.53
    # ens_acc = EnsembleClass.ensemble_accuracy(n_classifier=n_classifier, accuracy=basic_accuracy)
    # print('\nEnsemble accuracy for {0} classifiers with basic accuracy {1:.4f} = {2:.4f}'.format(n_classifier,
    #                                                                                             basic_accuracy, ens_acc))
    # #---
    req = EnsembleClass()
    feat_arr = req.feat_distr(n_classifier=req.n_classifier, n_features_in_clf=req.n_features_in_clf,
                              major_features_part=req.major_features_part, major_features_arr=req.major_features_arr,
                              minor_features_arr=req.minor_features_arr, print_log=False, save_dump=True)
    print("\nfeat_arr:\n", feat_arr)