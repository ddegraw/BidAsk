# -*- coding: utf-8 -*-
import blpfunctions as blp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split

def proc_bidask(sec, start, end, quant=0.5):
    event = ['BID','ASK','TRADE']

    bidask = pd.DataFrame()
    #for sec in secs:
    bidask=blp.get_Ticks(sec,event,start,end)
    #bidask=bidask.combine_first(df) 

    bidask['ind'] = np.zeros(len(bidask.index))
    bidask['trdind'] = pd.Series(np.nan,index=bidask.index)

    for i in range(len(bidask.index)):
        if bidask['type'][i] == 'BID':
            bidvol = bidask['size'][i]
            bidprc = bidask['price'][i]
        elif bidask['type'][i] == 'ASK':
            askvol = bidask['size'][i]
            askprc = bidask['price'][i]
        elif bidask['type'][i] == 'TRADE':
            if 'askprc' not in vars() or 'bidprc' not in vars():
                continue
            bidask['ind'].iloc[i] = (bidvol - askvol)/(1.0*(askvol + bidvol))
            #bidask['spr'].iloc[i] = 2.0*(askprc - bidprc)/(1.0*bidprc + askprc)
            trdvol = bidask['size'][i]
            trdprc = bidask['price'][i]
            if trdprc == askprc:
                bidask['trdind'].iloc[i] = 1
            else:
                bidask['trdind'].iloc[i] = 0
              
    #look at only trades in the top 50%ile, drop first and last trades
    bar = bidask.dropna()
    bar = bar.drop(bar.index[0])
    bar = bar.drop(bar.index[-1])
    foo = bar.loc[bar['size'] > bar['size'].quantile(q=quant)]
   
    #Split the dataset into training and testing
    #Xtrain, Xtest = train_test_split(foo['ind'], test_size = testsize)
    #ytrain, ytest = train_test_split(foo['trdind'], test_size = testsize)
    #Xtrain = pd.DataFrame(Xtrain)
    #Xtest = pd.DataFrame(Xtest)
    
    return pd.DataFrame(foo['ind']), foo['trdind']

#plt.scatter(foo['ind'],foo['trdind'],s=100*(foo['size']-foo['size'].mean())/foo['size'].std())
#plt.show()
'''
sec_list =['5202 JP Equity']
#ind = "TPX100 Index"
#sec_list = blp.get_index(ind)
amstart = "2016-08-11T09:00:00"
amend = "2016-08-11T11:30:00" 
pmstart = "2016-08-11T12:30:00"
pmend = "2016-08-11T15:00:00" 
'''
roclist = []
iceptlist = []
coefflist = []

for secs in sec_list:
    Xtrain, ytrain = proc_bidask(secs, amstart, amend)
    Xtest, ytest = proc_bidask(secs, pmstart, pmend)

    lg = LogisticRegression()
    lg.fit(Xtrain, ytrain)
    b = lg.intercept_
    m = lg.coef_

    disbursed=lg.predict_proba(Xtest)
    fpr, tpr, _ = roc_curve(ytest, disbursed[:,1])
    roc_auc = auc(fpr, tpr)
    
    roclist.append(roc_auc)
    iceptlist.append(b.item())
    coefflist.append(m.item())
    
     
lgparams = pd.DataFrame({'rocauc': roclist,
                         'icept': iceptlist,
                         'coeff': coefflist, }, index=sec_list)    
lgparams.rename(columns=lambda x: x[:4], inplace=True)