# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:36:59 2019

@author: hmtkv
"""
import time
import random
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_svmlight_file    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, log_loss
from timeit import default_timer as timer

N = []
error_data = []
error_sub =[]

subset_mse = []
dataset_mse =[]
weights = []
timedict = []

data = load_svmlight_file("mg.txt")
X, y = data[0], data[1] 

N_size = 100
lng = len(y)
train_pct_index = np.ceil(lng/N_size)
loop = 10
loop2 = 10
#train_pct_index = np.round(lng/N_size)

for resample in range(loop):
    mse_data = []
    mse_sub = []   
    w = []  
    timeFunc = []
  
    for i in range(1, N_size + 1):
        int_length = int(train_pct_index*(i))       
        if int_length >= lng:          
            X_sub = X[0:, :]
            y_sub = y[0:] 
        else:         
            index=random.sample(range(lng), int_length)
            X_sub=X[index]
            y_sub=y[index]
        
        if resample == loop - 1:
            N.append(int_length)
        
        t1 = timer()
        linmodel = LinearRegression().fit(X_sub, y_sub) 
        t2 = timer()
        timeFunc.append(t2-t1)
        
        y_hat_subset = linmodel.predict(X_sub)
        y_hat_dataset = linmodel.predict(X)
        
       
        mse_sub.append(mean_squared_error(y_sub, y_hat_subset))
        mse_data.append(mean_squared_error(y, y_hat_dataset))
      
        w.append(linmodel.coef_)   

    weights.append(w)
    subset_mse.append(mse_sub)
    dataset_mse.append(mse_data)
    timedict.append(timeFunc)

avg_dataset_acc = np.mean(dataset_mse,0)
avg_subset_acc =  np.mean(subset_mse,0)
mean_time=np.mean(timedict,0)

plt.rcParams.update({'font.size': 12})
f1=plt.figure(10)
plt.plot( N,timeFunc)
plt.xlabel('Number of samples')
plt.ylabel('Avg. Time (s)')


f2=plt.figure(20)
plt.plot(N, avg_subset_acc,'b', N, avg_dataset_acc,'r')
plt.legend(['Subset of N','Full dataset'])
plt.xlabel('Sample(N) Size')
plt.ylabel('mse')

#
shape=range(1,np.shape(w[0])[0]+1)

seq_w=[]
for i in range(1,len(shape)+1):
    seq_w.append(i)
f2 =  plt.figure(1)   
plt.stem(seq_w, w[0],'k')
plt.xlabel('Weight')
plt.ylabel('Value')


f4=plt.figure(2)
plt.stem(seq_w,w[9],'k')
plt.xlabel('Weight')
plt.ylabel('Value')

f5=plt.figure(3)
plt.stem(seq_w,w[99],'k')
plt.xlabel('Weight ')
plt.ylabel('Value')

f6=plt.figure(4)
plt.plot(N, w)
plt.xlabel('Sample(N) Size')
plt.ylabel('Weights')



