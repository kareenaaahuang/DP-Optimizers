#utils.py
#Karina Huang, Lipika Ramaswamy

#This package includes functions for modeling preprocessing.

#import dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def getXY(data, response_var, clip_cols, clip_vals, dummy_cols, quant_cols, random_state = 208):
    '''format data and return X and Y'''
    #avoid overwriting the original dataset
    dataNew = data.copy()


    #clip columns give lower and upper bound
    for i in range(len(clip_cols)):
        lower = clip_vals[i][0]
        upper = clip_vals[i][1]
        dataNew[clip_cols[i]] = np.clip(dataNew[clip_cols[i]], lower, upper)

    #normalize quantitative variabels
    scaler = MinMaxScaler()
    dataNew[quant_cols] = scaler.fit_transform(dataNew[quant_cols])

    #checkpoint
    print('Normalizaton complete.')

    #dummify categorical columns
    dataNew = pd.get_dummies(dataNew, columns = dummy_cols)

    #checkpoint
    print('Dummify complete.')

    #get X and Y
    Y = dataNew[response_var]
    X = dataNew.drop(response_var, axis = 1)

    return X, Y
