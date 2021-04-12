import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import random
import time
import os

## --------------------- config -------------------
## define steps to go back to predict, Try steps = [15,50]

steps = 14

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
df_minmax = minmax_scale.transform(d_dataset_open)


## Creating data with timestamps
dataSequence = []

#-4 so that we do not get out of bounds for day5 prediction
for i in range(steps, len(df_minmax)-4):
    
    # get values from coming days
    
    today = df_minmax[i-1, 0]
    day1 = df_minmax[i, 0]
    day2 = df_minmax[i+1,0]
    day3 = df_minmax[i+2,0]
    day4 = df_minmax[i+3,0]
    day5 = df_minmax[i+4,0]
  
    btc_value = df_minmax[i-steps:i, :df_minmax.shape[1]]
    
    # add absolute values as labels
    dataSequence.append([btc_value, day1,day2,day3,day4,day5])

#split dataset
# Split dataset into 70% training, 10% Validation and 20% Test data
n_cases = len(dataSequence)
n_train = int(round(n_cases*.7))
n_val = int(round(n_cases*.1))
n_test = int(round(n_cases*.2))

dataSequence_trval = dataSequence[:n_train+n_val]

dataSequence_test = dataSequence[n_train+n_val:n_cases]

# random data through shuffeling
random.shuffle(dataSequence_trval)


#store training and validation data
X = []
day1 = []
day2 = []
day3 = []
day4 = []
day5 = []


for seq, d1,d2,d3,d4,d5 in dataSequence_trval:  # going over our new sequential data
    X.append(seq)  # X is the sequences
    day1.append(d1)
    day2.append(d2)
    day3.append(d3)
    day4.append(d4)
    day5.append(d5)


pd.to_pickle(X,'Data/Input/sequences_trval.pkl')
pd.to_pickle(day1,'Data/Input/day1/Labels_trval.pkl')
pd.to_pickle(day2,'Data/Input/day2/Labels_trval.pkl')
pd.to_pickle(day3,'Data/Input/day3/Labels_trval.pkl')
pd.to_pickle(day4,'Data/Input/day4/Labels_trval.pkl')
pd.to_pickle(day5,'Data/Input/day5/Labels_trval.pkl')



#store test data

X_test = []
day1_test = []
day2_test = []
day3_test = []
day4_test = []
day5_test = []

for seq, d1,d2,d3,d4,d5 in dataSequence_test:  # going over our new sequential data
    X_test.append(seq)  # X is the sequences
    day1_test.append(d1)
    day2_test.append(d2)
    day3_test.append(d3)
    day4_test.append(d4)
    day5_test.append(d5)


pd.to_pickle(X_test,'Data/Input/sequences_test.pkl')
pd.to_pickle(day1_test,'Data/Input/day1/Labels_test.pkl')
pd.to_pickle(day2_test,'Data/Input/day2/Labels_test.pkl')
pd.to_pickle(day3_test,'Data/Input/day3/Labels_test.pkl')
pd.to_pickle(day4_test,'Data/Input/day4/Labels_test.pkl')
pd.to_pickle(day5_test,'Data/Input/day5/Labels_test.pkl')

#to load pickled dataset: 
# df = pd.read_pickle('Data/day1Labels.pkl')