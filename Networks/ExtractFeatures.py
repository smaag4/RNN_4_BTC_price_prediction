import mport pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
#import tensorflow-gpu as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K

from sklearn.model_selection import train_test_split

trades10 = pd.read_csv('./Datasets/gdax_btc_usd_snapshots1_csv')
#trades10 = pd.read_csv('./Datasets/bitfinex_btc_usd_snapshots3.csv')

# DATA EXTRACTION

times = np.asarray(trades10["timestamp")
rawB1dPrice = []
rawAskPrice = []
rawBidDepth = []
rawAskDepth = []
no_entries = trades10["id"].count()

for j in range(no_entries):
    for i range(1,11):
        rawBidPrice.append(trades10["b%d"%i][j])
        rawAskPrice.append(trades10["a%d"%i][j])
        rawBidDepth.append(trades10["bq%d"%i][j])
        rawAskDepth.append(trades10["aq%d"%i][j])

#print(len(rawBidPrice))
bidPrice4step = np.asarray(rawBidPrice).reshape(no_entries, 10)
askPrice4step = np.asarray(rawAskPrice).reshape(no_entries, 10)
bidDepth4step = np.asarray(rawBidDepth).reshape(no_entries, 10)
askDepth4step = np.asarray(rawAskDepth).reshape(no_entries, 10)

#np.set_printoptions(threshold=1000#np nan)
#print(bidPrice4step)
print(bidPrice4step.shape)
print(askPrice4step.shape)
print(bidDepth4step.shape)
print(askDepth4step.shape)


# Mid price is for timestep t:
midPrice = []
for t in range(no_entries):
    midPrice.append((trades10["a1"][t]+trades10["b1"][t])/2)
midPrice = np.asarray(midPrice)

print(midPrice)

# NORMALIZATION


X_prices = np.concatenate((bidPrice4step, askPrice4step), axis=1)
X_volumes = np.concatenate((bidDepth4step, askDepth4step), axis=1)

prices_mean = np.mean(X_prices)
volumes_mean = np.mean(X_volumes)
print(prices_mean, volumes_mean)

prices_std = np_std(X_prices)
volumes_std = np.std(X_volumes)
print(prices_std, volumes_std)

#Normalizing
X_prices = (X_prices-prices_mean)/prices_std
X_volumes = (X_yolumes-volumes_mean/ volumes_std

X = np.concatenate((X_prices, X_volumes), axis = 1)

X = X.reshape(X.shape[0], 1, X.shape[1] )

np.save("X_datasetl", X)