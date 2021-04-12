import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import glob
import os
from datetime import datetime
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adadelta, Adam, RMSprop
from hyperopt import fmin, hp, tpe, STATUS_OK, space_eval, Trials
from keras import backend, optimizers
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Data/complete_dataset_v10.csv')
df = df.iloc[:, 2:].values

steps = 17
### DATA SCALER
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn import preprocessing

# scale bitcoin values
min_max_scaler = MinMaxScaler()
minmax_scale = preprocessing.MinMaxScaler().fit(df)
df_minmax = minmax_scale.transform(df)

# Hyperparameters
hyper_space = {
    'lstm_units_1': hp.choice('lstm_units_1', np.arange(0, 30, 1)),
    'lstm_units_2': hp.choice('lstm_units_2', np.arange(0, 30, 1)),
    'lstm_units_3': hp.choice('lstm_units_3', np.arange(0, 30, 1)),
    'lstm_units_4': hp.choice('lstm_units_4', np.arange(0, 30, 1)),
    'dense_units_1': hp.choice('dense_units_1', np.arange(0, 8, 1)),
    'dense_units_2': hp.choice('dense_units_2', np.arange(0, 8, 1)),
    'dense_units_3': hp.choice('dense_units_3', np.arange(0, 8, 1)),
    'dense_units_4': hp.choice('dense_units_4', np.arange(0, 8, 1)),
    'dense_units_5': hp.choice('dense_units_5', np.arange(0, 8, 1)),
    'lstm_dropout_1' : hp.uniform('lstm_dropout_1', 0, 0.9),
    'lstm_dropout_2' : hp.uniform('lstm_dropout_2', 0, 0.9),
    'lstm_dropout_3' : hp.uniform('lstm_dropout_3', 0, 0.9),
    'lstm_dropout_4' : hp.uniform('lstm_dropout_4', 0, 0.9),
    'dense_dropout_1' : hp.uniform('dense_dropout_1', 0, 0.9),
    'dense_dropout_2' : hp.uniform('dense_dropout_2', 0, 0.9),
    'dense_dropout_3' : hp.uniform('dense_dropout_3', 0, 0.9),
    'dense_dropout_4' : hp.uniform('dense_dropout_4', 0, 0.9),
    'rec_dropout_1': hp.uniform('rec_dropout_1', 0, 0.9),
    'rec_dropout_2': hp.uniform('rec_dropout_2', 0, 0.9),
    'rec_dropout_3': hp.uniform('rec_dropout_3', 0, 0.9),
    'rec_dropout_4': hp.uniform('rec_dropout_4', 0, 0.9),
    'batch_size': hp.choice('batch_size', np.arange(1,60,10)),
    'timesteps': hp.choice('timesteps', np.arange(1,30,1)),
    'metric': hp.choice('metrics', ['mse', 'mae', 'mape']),
    'loss': hp.choice('loss', ['mean_squared_error', 'mean_squared_logarithmic_error', 'mean_absolute_error']),
    'optimizer': hp.choice('optimizer', ['adadelta', 'adam', 'rmsprop'])
}

def lstm_in_out(dataframe, timesteps):
    X = []
    y = []
    steps = 17
    # -4 so that we do not get out of bounds for day5 prediction
    for i in range(timesteps, len(dataframe) - 4):
        # get values from coming days
        today = df_minmax[i - 1, 0]
        day1 = df_minmax[i, 0]
        day2 = df_minmax[i + 1, 0]
        day3 = df_minmax[i + 2, 0]
        day4 = df_minmax[i + 3, 0]
        day5 = df_minmax[i + 4, 0]
        btc_value = df_minmax[i - steps:i, :df_minmax.shape[1]]
        # add absolute values as labels
        X.append(btc_value)
        y.append(day1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

    return X_train, y_train

def train_hyper_model(dataframe, hyper_params):
    X, y = lstm_in_out(dataframe, 17)
    model = Sequential()

    model.add(LSTM(
        units=hyper_params['lstm_units_1'],
        activation='relu',
        recurrent_dropout=hyper_params['rec_dropout_1'],
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ))

    model.add(Dropout(hyper_params['lstm_dropout_1']))

    model.add(LSTM(
        units=hyper_params['lstm_units_2'],
        recurrent_dropout=hyper_params['rec_dropout_2'],
        return_sequences=True,
        activation='relu',
    ))

    model.add(Dropout(hyper_params['lstm_dropout_2']))

    model.add(LSTM(
        units=hyper_params['lstm_units_3'],
        recurrent_dropout=hyper_params['rec_dropout_3'],
        return_sequences=True,
        activation='relu',
    ))

    model.add(Dropout(hyper_params['lstm_dropout_3']))

    model.add(LSTM(
        units=hyper_params['lstm_units_4'],
        recurrent_dropout=hyper_params['rec_dropout_4'],
        return_sequences=False,
        activation='relu',
    ))

    model.add(Dropout(hyper_params['lstm_dropout_4']))
    model.add(Dense(units=hyper_params['dense_units_1'], activation='relu'))
    model.add(Dropout(hyper_params['dense_dropout_1']))
    model.add(Dense(units=hyper_params['dense_units_2'], activation='relu'))
    model.add(Dropout(hyper_params['dense_dropout_2']))
    model.add(Dense(units=hyper_params['dense_units_4'], activation='relu'))
    model.add(Dropout(hyper_params['dense_dropout_3']))
    model.add(Dense(units=hyper_params['dense_units_5'], activation='relu'))
    model.add(Dense(units=2, activation="softmax"))
    model.compile(optimizer=hyper_params['optimizer'], loss=hyper_params['loss'], metrics=hyper_params['metric'])
    history = model.fit(
        X,
        y,
        batch_size=hyper_params['batch_size'],
        validation_split=0.2,
        epochs=20,
        shuffle=True,
        verbose=0)

    # take the last 8 accuracy values, and return their mean value:
    return np.mean(history.history['val_loss'][-8:])


### OBJECTIVE FUNCTION
def hyperopt_fn(hyper_params):
    accuracy = train_hyper_model(df_minmax, hyper_params) # X,Y are globally defined!
    backend.clear_session() # clear session to avoid models accumulation in memory
    return {'loss': accuracy, 'status': STATUS_OK}

trials = Trials()

while True:
    try:
        opt_params = fmin(
                        fn=hyperopt_fn,
                        space=hyper_space,
                        algo=tpe.suggest,
                        max_evals=20, # stop searching after 18 iterations
                        trials=trials,
                        rstate=np.random.RandomState(1)
                        )

        # store trials in a file
        f = open('store_trials_LSTM.pkl', 'wb')
        pickle.dump(trials, f)
        f.close()
        print('Best Hyperparameters: ', opt_params)
        print(space_eval(hyper_space, opt_params))
        print('number of trials:', len(trials.trials))
        break
    except:
        continue




print(trials.trials[-1]['misc']['vals']) # print last hyperparameters value














