#from blpfunctions 
import blpfunctions as blp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import callbacks

remote = callbacks.RemoteMonitor(root='http://localhost:9000')

#from sklearn.neural_network import MLPClassifier


def proc_bidask(sec, start, end, quant=0.5):
    event = ['BID','ASK','TRADE']

    bidask = pd.DataFrame()
    #for sec in secs:
    bidask=blp.get_Ticks(sec,event,start,end)
    #bidask=bidask.combine_first(df) 

    bidask['ind'] = np.zeros(len(bidask.index))
    bidask['trdind'] = pd.Series(np.nan,index=bidask.index)
    bidask['spr'] = np.zeros(len(bidask.index))
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
            bidask['spr'].iloc[i] = 20000.0*(askprc - bidprc)/(1.0*(bidprc + askprc))
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
    #foo = bar.loc[bar['size'] > bar['size'].quantile(q=quant)]
    inputs = pd.DataFrame(bar.loc[:,['ind','size']])
    #foo = bar.loc[:,['ind','size']]
    outputs = bar['trdind']
    #Split the dataset into training and testing
    #Xtrain, Xtest = train_test_split(foo['ind'], test_size = testsize)
    #ytrain, ytest = train_test_split(foo['trdind'], test_size = testsize)
    #Xtrain = pd.DataFrame(Xtrain)
    #Xtest = pd.DataFrame(Xtest)
    
    return inputs, outputs

#plt.scatter(foo['ind'],foo['trdind'],s=100*(foo['size']-foo['size'].mean())/foo['size'].std())
#plt.show()
'''
secs = '2768 JP Equity'
#ind = "NKY Index"
#sec_list = blp.get_index(ind)
amstart = "2016-09-05T09:00:00"
amend = "2016-09-09T15:00:00" 
pmstart = "2016-09-12T09:00:00"
pmend = "2016-09-12T15:00:00" 

roclist = []
iceptlist = []
coefflist = []

Xtrain, ytrain = proc_bidask(secs, amstart, amend)
Xtest, ytest = proc_bidask(secs, pmstart, pmend)
ytrain=ytrain.as_matrix().transpose()
ytest=ytest.as_matrix().transpose()
'''

model = Sequential()
model.add(Dense(1, input_shape=(2,), activation ='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
#rms = rmsprop(lr=0.01)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


model.fit(Xtrain.as_matrix(), ytrain, nb_epoch=100, batch_size=16, validation_split=0.2,callbacks=[remote])
score = model.evaluate(Xtest.as_matrix(),ytest, verbose=0)
disb = model.predict_proba(Xtest.as_matrix())
fpr, tpr, _ = roc_curve(ytest, disb)
roc_auc = auc(fpr, tpr)
print ('ROC_AUC:',  roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


print('Loss Test score:', score[0])
print('Accuracy Test score:', score[1])

#clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#clf = LogisticRegression()
#clf.fit(Xtrain, ytrain)
#b = lg.intercept_
#m = lg.coef_
#disbursed=clf.predict_proba(Xtest)
#fpr, tpr, _ = roc_curve(ytest, disbursed[:,1])
#roc_auc = auc(fpr, tpr)

#plt.title('Receiver Operating Characteristic')
#plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
#plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
#plt.xlim([-0.1,1.1])
#plt.ylim([-0.1,1.1])
#plt.ylabel('True Positive Rate')
#plt.xlabel('False Positive Rate')