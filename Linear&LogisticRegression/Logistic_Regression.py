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
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, log_loss


#def plot(N, accuracyDataSet, accuracySubSet):   
#    num=[]
#    for i in range(1,11):
#        num.append(i)   
#    plt.figure(figsize=(8,4))
#    plt.plot(N,accuracySubSet,'b-', label='Subset')
#    plt.plot(N,accuracyDataSet,'r-', label='DataSet')
#    plt.legend()    
#    
#    plt.xlabel('Number of N (N)', fontsize=18)
#    plt.ylabel('Square loss', fontsize=18)

N = []
error_data = []
error_sub =[]

dataset_error = []
subset_error = []
dataset_plot=[]
subset_plot=[]

dataset_acc = []
subset_acc = []
timedict = []
mean_time = []
data = load_svmlight_file("diabetes.txt")
X, y = data[0], data[1] 

N_size = 100
lng = len(y)
train_pct_index = np.ceil(lng/N_size)
loop = 10

#train_pct_index = np.round(lng/N_size)

for resample in range(loop):
    acc_data = []
    acc_sub = []
    error_data = []
    error_sub =[]
    weights = []
    timeFunc = []
    sayac = 0
    for i in range(1, N_size + 1):
        int_length = int(train_pct_index*(i))
        if int_length >= lng:          
            X_sub = X[0:, :]
            y_sub = y[0:] 
        else:         
            index=random.sample(range(lng), int_length)
            X_sub=X[index]
            y_sub=y[index]
        
        if resample == loop - 2:
            N.append(X_sub.shape[0])



        t1 = time.time()
        logmodel = LogisticRegression().fit(X_sub, y_sub) 
        t2 = time.time()
        timeFunc.append(t2-t1)

#       logmodel.fit(X_train_N, y_train_N)
        
        y_hat_subset = logmodel.predict_proba(X_sub)
        y_hat_dataset = logmodel.predict_proba(X)
        
        
        acc_sub.append(logmodel.score(X_sub, y_sub))
        acc_data.append(logmodel.score(X, y))
        
        error_sub.append(log_loss(y_sub, y_hat_subset, labels=[0.,1.]))
        error_data.append(log_loss(y, y_hat_dataset, labels=[0,1]))
             
        weights.append(logmodel.coef_)
                  
            
    dataset_acc.append(acc_data)
    dataset_error.append(error_data)
    
    subset_acc.append(acc_sub)
    subset_error.append(error_sub) 
    timedict.append(timeFunc)
 

avg_dataset_error = np.mean(dataset_error,0)
avg_subset_error = np.mean(subset_error,0)
avg_dataset_acc = np.mean(acc_data,0)
avg_subset_acc = np.mean(subset_acc,0)
mean_time=np.mean(timeFunc,0)

plt.rcParams.update({'font.size': 12})
plt.figure()
plt.plot( N,timeFunc)
plt.xlabel('Number of samples')
plt.ylabel('Avg. Time (s)')

coefs = []
for sublist in weights:
    for item in sublist:
        coefs.append(item)
        
plt.figure()
plt.plot( N,coefs)
plt.xlabel('Number of samples')
plt.ylabel('Weights')

for item in coefs[98:99]:
    plt.figure()
    markerline, stemlines, baseline = plt.stem(item, linefmt='black', markerfmt='o', bottom=0, use_line_collection=True)
    plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
    plt.setp(stemlines, 'linestyle', 'dotted')
    plt.title("Coefficients change stem plot", fontsize = 24)
    plt.show()


            
            
            
            
            
            
            
            
            
            
            
            
            