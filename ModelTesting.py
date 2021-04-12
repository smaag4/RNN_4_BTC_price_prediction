import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os

day = 1

## --------------------- config -------------------


### DATASETS
d_dataset = pd.read_csv('Data/complete_dataset_v10.csv')

## ---------------------- end config ----------------


# select open and high values only (1:2 = position in dataset = open:1)
d_dataset_open = d_dataset.iloc[:, 2:].values

### DATA SCALER
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn import preprocessing

# scale bitcoin values
min_max_scaler = MinMaxScaler()
minmax_scale = preprocessing.MinMaxScaler().fit(d_dataset_open)

# get date and bitcoin open price from datasets
d_dataset_dates = d_dataset.iloc[:,1].values
d_dataset_openPrices = d_dataset.iloc[:,2].values

#only keep last 20% of dates and prices
n_cases = len(d_dataset_dates)
n_train = int(round(n_cases*.7))
n_val = int(round(n_cases*.1))
d_dataset_dates = d_dataset_dates[n_train+n_val:n_cases]
d_dataset_openPrices = d_dataset_openPrices[n_train+n_val:n_cases]

#from 20% - delete last 5 days - we do not have any predictions for them
d_dataset_dates = d_dataset_dates[:len(d_dataset_dates)-5]
d_dataset_openPrices = d_dataset_openPrices[:len(d_dataset_openPrices)-5
]
#create output file and fill up with all zeros
outputFile = np.zeros(shape=(len(d_dataset_dates), 27))

#add date and bitcoin open price to output file
outputFile[:,0] = d_dataset_dates[:]
outputFile[:,1] = d_dataset_openPrices[:]

#create output Dataframe Column Headings
outputFileColumns = ['Date','Todays Value']


#loop over predictions to predict values and input them to output file
#variable to assign values to output file
cnt_Position = 2
while day < 6:

    #load model
    model =  tf.keras.models.load_model("models/model_day"+str(day))

    #load test data 
    X_test = pd.read_pickle('Data/Input/sequences_test.pkl')
    y_test = pd.read_pickle('Data/Input/day'+str(day)+'/Labels_test.pkl')
   
     ### TRANSFORM DATA INTO NUMPY ARRAY
    # change to numpy array with shape (Amount of rows, features per row)
    X_test, y_test = np.array(X_test), np.array(y_test)
    # reshape to same architecture
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    #predict
    predicted_stock_price = model.predict(X_test)   

    # inverse Transform the values back
    # add zeros to have the same shape
    predicted_stock_price_inverseTransform = np.zeros(shape=(len(predicted_stock_price), d_dataset_open.shape[1]))
    predicted_stock_price_inverseTransform[:, 0] = predicted_stock_price[:, 0]
    predicted_stock_price_inverseTransform = minmax_scale.inverse_transform(predicted_stock_price_inverseTransform)



    y_test_inverseTransform = np.zeros(shape=(len(y_test), d_dataset_open.shape[1]))
    y_test_inverseTransform[:, 0] = y_test[:]
    y_test_inverseTransform = minmax_scale.inverse_transform(y_test_inverseTransform)

    
    #add values to output file

    # real value
    outputFile[:, cnt_Position] = y_test_inverseTransform[:, 0]
    # predicted value
    outputFile[:, cnt_Position+1] = predicted_stock_price_inverseTransform[:, 0]

    # relative difference of actual and predicted value
    outputFile[:, cnt_Position+2] = np.divide(np.absolute(np.subtract( outputFile[:, cnt_Position], outputFile[:, cnt_Position+1])),  outputFile[:, cnt_Position])

 
    
    #add label if actual value has gone up compared to todays value
    outputFile[:,cnt_Position+3] = np.where(outputFile[:,cnt_Position]>=outputFile[:,1] , 1, -1)
    #add label if predicted value has gone up in compared  to todays value
    outputFile[:,cnt_Position+4] = np.where(outputFile[:,cnt_Position+1]>=outputFile[:,1] , 1, -1)
   
    #add column labelings
    outputFileColumns.extend(["Value Day "+str(day), "Predicted Value Day " + str(day),"Relative Difference Day "+ str(day), "Label Day "+str(day), "Predicted Value Day" +str(day)])

    # # calculate average relative difference over all predictions
    # totalAveragePrediction = np.mean(testValues2, axis=0)[2]
    # print('TestWerte2', testValues2)
    # print('Durchschnittsabweichung', totalAveragePrediction*100, '%')
    day += 1
    cnt_Position+=5

# convert array into dataframe
DF = pd.DataFrame(outputFile)
DF.columns = outputFileColumns
# save the dataframe as a csv file
DF.to_csv("Data/Output/outputData.csv", index=False)
