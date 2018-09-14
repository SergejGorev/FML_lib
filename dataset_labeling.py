import processing
import multiprocessing as mp
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime as dt
import time
import pickle
import sys

class LabelingClass:
    #---
    num_threads = 16
    trgt = 0.0025 # target price relative change
    tpSL = [1., 0.5]
    trgt_arr = [0.0015 , 0.002, 0.0025, 0.003, 0.004, 0.0045]
    tpSL_arr = [[1., 0.5], [1., 1.], [1., 1.5], [1., 2.]]
    pos_duration_max = 3 # (days)
    data_pickle_path = r'/home/rom/01-Algorithmic_trading/02_1-EURUSD/eurusd_5_v1.pickle'  # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\eurusd_5_v1.1.pickle"
    labels_pickle_path = r'/home/rom/01-Algorithmic_trading/02_1-EURUSD/'   # r"d:\20-ML_projects\01-Algorithmic_trading\02_1-EURUSD\\"
    labels_file_name = "eurusd_5_v1_lbl"
    lbl_bsc_names = ['label_buy', 'label_sell', 'target_label']
    #---
    __postfix = ''
    __buy_sl_clmn = ''
    __buy_tp_clmn = ''
    __sell_sl_clmn = ''
    __sell_tp_clmn = ''
    __labels_buy_clmn = ''
    __labels_sell_clmn = ''
    __target_clmn = ''

    #---
    # TRIPLE-BARRIER LABELING METHOD (twosides version)
    """
    ◦ close : A pandas series of prices.
    ◦ events : A pandas dataframe, with columns,
        ◦ t1 : The timestamp of vertical barrier. When the value is np.nan , there will
            not be a vertical barrier.
        ◦ trgt : The unit width of the horizontal barriers.
    ◦ tpSL : A list of two non-negative float values:
        ◦ tpSL[0] :The factor that multiplies trgt to set the width of the upper barrier.
            If 0, there will not be an upper barrier.
        ◦ tpSL[1] :The factor that multiplies trgt to set the width of the lower barrier.
            If 0, there will not be a lower barrier.
    ◦ molecule : A list with the subset of event indices that will be processed by a
        single thread. Its use will become clear later on in the chapter.
    """
    def clmn_def(self):
        postfix = '_' + processing.digit_to_text(self.trgt) + '_' + processing.digit_to_text(self.tpSL[0]) + \
                  '_' + processing.digit_to_text(self.tpSL[1])
        self.__postfix = postfix
        self.__buy_sl_clmn = 'buy_sl' + postfix
        self.__buy_tp_clmn = 'buy_tp' + postfix
        self.__sell_sl_clmn = 'sell_sl' + postfix
        self.__sell_tp_clmn = 'sell_tp' + postfix
        self.__labels_buy_clmn = self.lbl_bsc_names[0] + postfix
        self.__labels_sell_clmn = self.lbl_bsc_names[1] + postfix
        self.__target_clmn = self.lbl_bsc_names[2] + postfix

    def apply_tpSLOnT1_twosides(self, df, events, tpSL, molecule=None):
        open_ask = df['open_ask']
        high_ask = df['high_ask']
        low_ask = df['low_ask']
        open_bid = df['open_bid']
        high_bid = df['high_bid']
        low_bid = df['low_bid']
        # apply stop loss/profit taking, if it takes place before t1 (end of event)
        if (molecule is None): events_=events.loc[:]
        else: events_=events.loc[molecule]
        #print(events_)
        out = events_[['t1']].copy(deep = True)
        #print('\n')
        #print(out)
        if tpSL[0]>0:tp = tpSL[0]*events_['trgt']
        else:pt = pd.Series(index = events.index) # NaNs
        #print(pt)
        if tpSL[1]>0:sl = -tpSL[1]*events_['trgt']
        else:sl = pd.Series(index = events.index) # NaNs
        #print(pt)
        for loc,t1 in events_['t1'].fillna(open_ask.index[-1]).iteritems():
            df_high_ask = high_ask[loc:t1] # path prices
            c_h_ask = (df_high_ask/open_bid[loc]-1) # path returns
            df_low_ask = low_ask[loc:t1]  # path prices
            c_l_ask = (df_low_ask/open_bid[loc]-1) # path returns
            df_high_bid = high_bid[loc:t1] # path prices
            c_h_bid = (df_high_bid/open_ask[loc]-1) # path returns
            df_low_bid = low_bid[loc:t1]  # path prices
            c_l_bid = (df_low_bid/open_ask[loc]-1) # path returns
            #print('\n'); print(df0)
            # for buy side
            out.loc[loc,self.__buy_sl_clmn] = c_l_bid[c_l_bid<sl[loc]].index.min() # earliest stop loss.
            out.loc[loc,self.__buy_tp_clmn] = c_h_bid[c_h_bid>tp[loc]].index.min() # earliest profit taking.
            # for sell side
            out.loc[loc,self.__sell_sl_clmn] = c_h_ask[c_h_ask>-sl[loc]].index.min() # earliest stop loss.
            out.loc[loc,self.__sell_tp_clmn] = c_l_ask[c_l_ask<-tp[loc]].index.min() # earliest profit taking.
        return out

    def linParts(self, numAtoms,numThreads):
        # partition of atoms with a single loop
        parts = np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
        parts = np.ceil(parts).astype(int)
        return parts

    def nestedParts(self, numAtoms,numThreads,upperTriang = False):
        # partition of atoms with an inner loop
        parts,numThreads_ = [0],min(numThreads,numAtoms)
        for num in range(numThreads_):
            part = 1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
            part = (-1+part**.5)/2.
            parts.append(part)
        parts = np.round(parts).astype(int)
        if upperTriang: # the first rows are the heaviest
            parts = np.cumsum(np.diff(parts)[::-1])
            parts = np.append(np.array([0]),parts)
        return parts

    def processJobs_(self, jobs):
        # Run jobs sequentially, for debugging
        out = []
        for job in jobs:
            out_ = self.expandCall(job)
            out.append(out_)
        return out

    def reportProgress(self, jobNum,numJobs,time0,task, printLog=False):
        # Report progress as asynch jobs are completed
        msg = [float(jobNum)/numJobs,(time.time()-time0)/60.]
        msg.append(msg[1]*(1/msg[0]-1))
        timeStamp = str(dt.datetime.fromtimestamp(time.time()))
        msg = timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
            str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
        if jobNum<numJobs:
            sys.stderr.write(msg+' \r')
            if printLog: print(msg)
        else:
            sys.stderr.write(msg+' \n')
            if printLog: print(msg)
        return

    def processJobs(self, jobs, task=None, numThreads=8):
        # Run in parallel.
        # jobs must contain a ’func’ callback, for expandCall
        if task is None: task = jobs[0]['func'].__name__
        pool = mp.Pool(processes=numThreads)
        outputs, out, time0 = pool.imap_unordered(self.expandCall, jobs), [], time.time()
        # Process asynchronous output, report progress
        for i, out_ in enumerate(outputs, 1):
            #print('i= {0}, out_= {1}'.format(i, out_))
            out.append(out_)
            self.reportProgress(i, len(jobs), time0, task)
        pool.close(); pool.join() # this is needed to prevent memory leaks
        return out

    def expandCall(self, kargs):
        # Expand the arguments of a callback function, kargs[’func’]
        func = kargs['func']
        del kargs['func']
        out = func(**kargs)
        return out

    def mpPandasObj(self, func, pdObj, numThreads=8, mpBatches=1, linMols=True, **kargs):
        '''
        Parallelize jobs, return a DataFrame or Series
        + func: function to be parallelized. Returns a DataFrame
        + pdObj[0]: Name of argument used to pass the molecule
        + pdObj[1]: List of atoms that will be grouped into molecules
        + kargs: any other argument needed by func
        Example: df1 = mpPandasObj(func,(’molecule’,df0.index),24,**kargs)
        '''
        if linMols: parts = self.linParts(len(pdObj[1]), numThreads*mpBatches)  # linParts(len(argList[1]),numThreads*mpBatches)
        else: parts = self.nestedParts(len(pdObj[1]), numThreads*mpBatches)  # nestedParts(len(argList[1]),numThreads*mpBatches)
        jobs = []
        for i in range(1,len(parts)):
            job ={pdObj[0]: pdObj[1][parts[i-1]:parts[i]], 'func': func}
            job.update(kargs)
            jobs.append(job)
        if numThreads==1: out = self.processJobs_(jobs)
        else: out = self.processJobs(jobs, numThreads=numThreads)
        #print(out)
        if isinstance(out[0],pd.DataFrame): df0 = pd.DataFrame()
        elif isinstance(out[0],pd.Series): df0 = pd.Series()
        else: return out
        for i in out: df0 = df0.append(i)
        df0 = df0.sort_index()
        return df0

    # функция для определения метки на основании данных времени TP и SL
    def label_bins(self, x):
        sl_dt, tp_dt = x[0], x[1]
        res = 0
        if (str(sl_dt)=='NaT') & (str(tp_dt)!='NaT'): res=1
        elif (str(sl_dt)!='NaT') & (str(tp_dt)=='NaT'): res=-1
        elif (str(sl_dt)=='NaT') & (str(tp_dt)=='NaT'): res=0
        elif (sl_dt<tp_dt): res=-1
        elif (sl_dt>tp_dt): res=1
        return res

    def execute(self, trgt=0.0075, tpSL=[1., 1]):
        time_start = dt.datetime.now()
        print('step: time_start= {}'.format(time_start))

        self.trgt = trgt
        self.tpSL = tpSL
        self.clmn_def()

        with open(self.data_pickle_path, "rb") as pckl:
            data = pickle.load(pckl)

        clmns_sel = ['open_ask', 'high_ask', 'low_ask', 'open_bid', 'high_bid', 'low_bid']

        data = data.loc[:, clmns_sel]  # data.loc[data.index[:10000], clmns_sel]

        events = pd.DataFrame(data.index, index=data.index)
        events.columns = ['t1']
        events['t1'] = events['t1'] + BDay(req.pos_duration_max)  # + pd.Timedelta(days=1)
        events['trgt'] = self.trgt

        labels = req.mpPandasObj(func=self.apply_tpSLOnT1_twosides, pdObj=('molecule', events.index), \
                                 numThreads=self.num_threads, df=data, events=events, tpSL=self.tpSL)

        labels_buy = list(map(self.label_bins, labels.loc[:, [self.__buy_sl_clmn, self.__buy_tp_clmn]].values))
        labels_sell = list(map(self.label_bins, labels.loc[:, [self.__sell_sl_clmn, self.__sell_tp_clmn]].values))

        labels[self.__labels_buy_clmn] = labels_buy
        labels[self.__labels_sell_clmn] = labels_sell
        #---
        lbl_func = lambda x: 1 if ((x[self.__labels_buy_clmn] == 1) & (x[self.__labels_sell_clmn] == -1)) else \
            (-1 if ((x[self.__labels_buy_clmn] == -1) & (x[self.__labels_sell_clmn] == 1)) else 0)

        labels[self.__target_clmn] = labels.apply(lbl_func, axis=1)
        #---
        labels = labels.loc[:, ['t1', self.__buy_sl_clmn, self.__buy_tp_clmn, self.__labels_buy_clmn,
                                self.__sell_sl_clmn, self.__sell_tp_clmn, self.__labels_sell_clmn, self.__target_clmn]]

        file_name = self.labels_pickle_path + self.labels_file_name + self.__postfix + '.pickle'
        with open(file_name, 'wb') as pckl:
            pickle.dump(labels, pckl)

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('step: time_finish= {0}, duration= {1}'.format(time_start, time_duration))

    def execute_cycle(self, trgt_arr, tpSL_arr):
        time_start = dt.datetime.now()
        print('cycle: time_start= {}'.format(time_start))
        i = 0
        for trgt in trgt_arr:
            for tpSL in tpSL_arr:
                i += 1
                print('i= {0}: trgt= {1}, tpSL= {2}'.format(i, trgt, tpSL))
                self.execute(trgt=trgt, tpSL=tpSL)

        time_finish = dt.datetime.now()
        time_duration = time_finish - time_start
        print('cycle: time_finish= {0}, duration= {1}'.format(time_start, time_duration))


if __name__ == '__main__':
    req = LabelingClass()
    # req.execute(trgt=0.0080, tpSL=[1., 1.])
    req.num_threads = 16
    req.execute_cycle(trgt_arr=req.trgt_arr, tpSL_arr=req.tpSL_arr)

