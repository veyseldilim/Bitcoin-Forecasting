# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:28:39 2021

@author: veyse
"""
import keras
import tensorflow as tf
#print(tf.__version__)
#print(keras.__version__)
import random as rn
import math
import numpy as np
import pandas as pd
import os
import glob
import datetime
import pandas as pd
from pandas_datareader import data
from pandas_datareader._utils import RemoteDataError
import pandas as pd
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.x13
from statsmodels.tsa.x13 import x13_arima_select_order, _find_x12
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, AveragePooling1D,MaxPooling1D
from keras.layers import Conv1D,AveragePooling1D,MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, Nadam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2,l1_l2
from keras import backend as K
from numpy import concatenate
from pytrends.request import TrendReq
from pytrends import dailydata
import sqlite3
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

def invert_scaling(x_minmax,y_predicted,n_steps_in):
    reshaped_pred = y_predicted.reshape((y_predicted.shape[0],1))
    matrix = concatenate((x_minmax[n_steps_in-1:],reshaped_pred), axis=1)
    matrix_min_max_invert = min_max_scaler.inverse_transform(matrix)
    inv_pred = matrix_min_max_invert[:,-1]
    return inv_pred




def plot_results(actual_y,predicted_y,method,date):
    mse = mean_squared_error(actual_y,predicted_y)
    
    plt.figure(figsize=(16,4))
    plt.plot(date,actual_y)
    plt.plot(date,predicted_y)
    plt.legend(['Actual','Predicted'])
    plt.title(f'{method} (MSE: {mse})')
#    print('MSE: ',f'{method} (MSE: {mse})')   
   # print(float(float(f'{mse}') / 1000000))
   # print(float(float(f'{mse}')))
    mse = mean_squared_error(predicted_y,actual_y)
    print("mse: ", mse)


    r2 = r2_score(predicted_y,actual_y)
    print("r2_score: ", r2)


    mae = mean_absolute_error(predicted_y,actual_y)
    print("mae: ", mae)


    rmse = math.sqrt(mean_squared_error(predicted_y,actual_y))
    print("rmse: ", rmse)

    plt.show()
    

#split a multivariate sequence into samples that preserve the temporal structure of the data
#SOURCE:https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def test_stationarity(df, ts):

    # Determing rolling statistics
    rolmean = df[ts].rolling(window = 12, center = False).mean()
    rolstd = df[ts].rolling(window = 12, center = False).std()

    # Plot rolling statistics:
    orig = plt.plot(df[ts], color = 'blue',label = 'Original')
    mean = plt.plot(rolmean, color = 'red',label = 'Rolling Mean')
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation for %s' %(ts))
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()

    # Perform Dickey-Fuller test:
    # Null Hypothesis (H_0): time series is not stationary
    # Alternate Hypothesis (H_1): time series is stationary
    print ('Results of Dickey-Fuller Test:')
    dftest = statsmodels.tsa.stattools.adfuller(df[ts], autolag='AIC') #add kpss
    
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value', '# Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    

#Defining a function to calculate percentage change
def percentChange(x,numLags):
    '''
    INPUTS:
    x: Column for which we want to calculate percent change
    numLags: The number of days from when the change needs to be calculated. 
            Example : If using daily data - numLags = 1 for daily change
                                            numLags = 30 for monthly change
                                            numLags = 365 for yearly change       
    OUTPUT:
    percentage change in variable
    '''
    y = (x - x.shift(numLags))/x.shift(numLags)
    return y


conAllDF = sqlite3.connect('allDF.db')
df = pd.read_sql_query("SELECT * FROM FormulaallDataTable" , conAllDF, index_col = 'index')


df['Date'] = pd.to_datetime(df['Date']) #always format the date
df.reset_index(drop=True, inplace=True )

df = df.rename(columns={'sums': 'TwitterSentimentAnalysis'})

#df = df[df['Date'] > '2018-1-1']

#usdIndexPrice
#goldPrice
#ethClose
#btcTweetCount
#sums
#Open
#High
#Low
#Volume
#Adj Close
#Close

#df = df[['Date','Open','High','Low','Volume','Adj Close','Close']]

df = df[['Date', 'usdIndexPrice', 'goldPrice','ethClose','btcTweetCount','TwitterSentimentAnalysis',
        'Open','High','Low','Volume','Adj Close','Close']]

test_stationarity(df, 'Close')

dataForMl = pd.DataFrame()
dataForMl['Date'] = df['Date']

#here, I only have level variables so I do not need separate my variables into level vs non-level variables
levelVars = df.columns[1:]
for levelVar in levelVars:
    dataForMl[f'{levelVar}Ret'] = percentChange(df[levelVar],1)
    
dataForMl = dataForMl[1:] #ignoring the first row as it contains null values


test_stationarity(dataForMl, 'CloseRet')

tsaplots.plot_acf(dataForMl['CloseRet'].astype(float),lags =20)
tsaplots.plot_pacf(dataForMl['CloseRet'].astype(float),lags =20)


tsaplots.plot_acf(df['Close'].astype(float),lags =20)
tsaplots.plot_pacf(df['Close'].astype(float),lags =20)



# Since we're going to forecast the one day ahead Nifty stock returns, the minimum lag considered by me is 1


minLagNum = 1
maxLagNum = 2
dataForMl = df.sort_values(['Date'])
for column in dataForMl.columns[1:]:
    for lag in range(minLagNum,maxLagNum+1):
        dataForMl[f'{column}Lag_{lag}'] = dataForMl[f'{column}'].shift(lag)
        



dataForMl.head(5)


#sort by date
dataForMl = dataForMl.sort_values(['Date'])

#removing columns if nan value in a column
dataForMl = dataForMl.dropna()

#specifying independent variables:including only lagged versions of variables and excluding date variables
final_vars = [col for col in dataForMl.columns if (col.find('Lag')!=-1) & (col.find('Date')==-1)]

#specifying the dependent variable
dep_var = 'Close'

#always make the dependent ariable the last column in the dataset
final_vars.append(dep_var)

#for later use
dataForMl_copy = dataForMl

#keeping only relevant 
dataForMl = dataForMl[final_vars]





#breaking the testing data into validation and out of sample data


#fit and transform training data




test_percent = 0.4
no_test_obs =  int(np.round(test_percent*len(dataForMl)))
training = dataForMl[:-no_test_obs]
testing = dataForMl[-no_test_obs:]

validation_percent = 0.90
no_validation_obs = int(np.round(validation_percent*len(testing)))
validation = testing[:-no_validation_obs]
outOfSample = testing[-no_validation_obs:]



min_max_scaler = MinMaxScaler(feature_range=(0, 2))
trainMinmax = min_max_scaler.fit_transform(training.values) 
valMinmax = min_max_scaler.transform(validation.values)
outSampleMinmax = min_max_scaler.transform(outOfSample.values)


#breaking the data into independent variables (x) and dependent variables (y)

#training independent, dependent
trainMinmax_x,trainMinmax_y = trainMinmax[:,:-1],trainMinmax[:,-1] 

#validation independent, dependent
valMinmax_x,valMinmax_y = valMinmax[:,:-1],valMinmax[:,-1]

#out of sample testing independent, dependent
outSampleMinmax_x,outSampleMinmax_y = outSampleMinmax[:,:-1],outSampleMinmax[:,-1]

# 5 icin %64, 7 icin %69, 9 icin %70
n_steps_in = 11 #number of observations from the past that we assume to be relevant across time for forecasting
n_steps_out = 1 #number of units ahead that we want to forecast into the future

#training sequence
trainSeq_x, trainSeq_y = split_sequences(trainMinmax, n_steps_in,n_steps_out)

#out of sample sequence
validationSeq_x, validationSeq_y= split_sequences(valMinmax, n_steps_in,n_steps_out)

#out of sample sequence
outSampleSeq_x, outSampleSeq_y= split_sequences(outSampleMinmax, n_steps_in,n_steps_out)



############################# For Replicability : Always run this as one cell ##########################################
#SOURCE :
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

#tf.random.set_random_seed(1234)
tf.random.set_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
###########################################################################################################################

#While training the neural network, it is important that we use the MSE error of the validation set to decide when to 
#stop training our network. If we use the MSE of the training set, we will not get good predictions in the test set due 
#to over fitting. However, unlike the error in the training set, the error in the validation set does not reduce with 
#every passing epoch. Sometimes, it increases for a while before it starts declining. The patience argument in Earlystop allows us 
#to decide how many times we want the validation error to keep increasing before we stop training the neural network.

EarlyStop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto',restore_best_weights=True)

epochs = 1000 #number of times the feed forward mechanism and back propagation are repeated 

bs = 100 #batch size for SGD :show what happens when batch size very small
lr = 0.1 #learning rate: the degree to which the weights are updated by each batch of SGD

sgd = SGD(lr=lr) #type of optimizer - Alternative: ADAM, NADAM

X, y = split_sequences(trainMinmax, n_steps_in,n_steps_out)
n_features = X.shape[2]

np.random.seed(0)

model = Sequential() 


model.add(Conv1D(filters=128,
                 kernel_size=2,
                 strides=1, 
                 activation='relu',
                 input_shape=(n_steps_in, n_features))) 

model.add(MaxPooling1D(pool_size=2)) 

model.add(Dropout(0.1)) 

model.add(Flatten())


model.add(Dense(50,
                activation='relu',
                kernel_regularizer=l2(0.01))) 
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') 

model.fit(trainSeq_x, trainSeq_y,
          batch_size=bs,
          epochs=epochs, 
          callbacks= [EarlyStop] ,
          verbose=0, 
          shuffle=False,
          validation_data =(validationSeq_x, validationSeq_y))
                         
model.summary()


# validation metrics 
cnnValPred = model.predict(validationSeq_x)


plot_results(validationSeq_y ,cnnValPred  ,'CNN Validation',range(len(validationSeq_y)))

mse = mean_squared_error(validationSeq_y,cnnValPred)
print(mse)

#testing prediction


print('-------------------- Normalized -----------------')
cnnOutSamplePred = model.predict(outSampleSeq_x)







plot_results(outSampleSeq_y ,cnnOutSamplePred  ,'CNN Testing',range(len(outSampleSeq_y)))



"""
mse = mean_squared_error(outSampleSeq_y,cnnOutSamplePred)
print("MSE normalized = " ,mse)

real_direction = []
predicted_direction = []




#up = 1
#down = 0
for i in range(len(outSampleSeq_y) -1):
    if(outSampleSeq_y[i] > outSampleSeq_y[i+1]):
        real_direction.append(0)
    else:
        real_direction.append(1)
    
for i in range(len(cnnOutSamplePred) -1):
    if(cnnOutSamplePred[i] > cnnOutSamplePred[i+1]):
        predicted_direction.append(0)
    else:
        predicted_direction.append(1)    





accuracy = accuracy_score(real_direction,predicted_direction)

print('accuracy score : ', accuracy * 100, '% ')

"""


print(" ---------- INVERTED --------------")
inv_yhat = invert_scaling(outSampleMinmax_x,cnnOutSamplePred,n_steps_in)
plot_results(dataForMl_copy['Close'][-(no_validation_obs-n_steps_in+1):].values,
             inv_yhat,
             'CNN Original Values',
             dataForMl_copy['Date'][-(no_validation_obs-n_steps_in+1):])









