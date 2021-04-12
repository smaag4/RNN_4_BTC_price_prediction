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


## --------------------- config -------------------
## define steps to go back to predict, Try steps = [15,50]

steps = 14
# model values
epochs = 20
batch_size = 41

### load pkl DATASETS with trainval features
sequences_trval = pd.read_pickle('Data/Input/sequences_trval.pkl')

# Split trval dataset into 70% training, 10% Validation 
n_cases = len(sequences_trval)
n_train = int(round(n_cases*.875))
n_val = int(round(n_cases*.125))

# count variable for Y features
day = 1

while day < 6:
    NAME = "model_day" + f"{day}"
    #load testtrain labels
    labels_trval = pd.read_pickle('Data/Input/day' + str(day) + '/Labels_trval.pkl')
    
    y_train = labels_trval[:n_train]
    y_val = labels_trval[n_train:n_train+n_val]
    
    X_train = sequences_trval[:n_train]
    X_val = sequences_trval[n_train:n_train+n_val]


    ### TRANSFORM DATA INTO NUMPY ARRAY
    # change to numpy array with shape (Amount of rows, features per row)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    print('x shape: ', X_train.shape, ' and y shape: ', y_train.shape)
    # reshape to same architecture
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))

    ### BUILDING THE LSTM
    model = Sequential()
    # 1 Layer
    model.add(LSTM(units=26, recurrent_dropout=0.3631078489530095, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.333874509832846))
    # add(Dropout(0.2))
    # 2 Layer
    model.add(LSTM(units=5, recurrent_dropout=0.39129750557657583, return_sequences=True, activation='relu'))
    model.add(Dropout(0.7783976411734224))
    # model.add(Dropout(0.2))<
    model.add(BatchNormalization())
    # 3 Layer
    model.add(LSTM(units=5, recurrent_dropout=0.026470176030291892, return_sequences=True, activation='relu'))
    model.add(Dropout(0.8525098143144538))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # 4 Layer
    model.add(LSTM(units=24, recurrent_dropout=0.3011458906809744, return_sequences=True, activation='relu'))
    model.add(Dropout(0.771703097508716))

    model.add(Dense(units=5, activation='relu'))
    model.add(Dropout(0.11796227911966399))
    model.add(Dense(units=1, activation='relu'))
    model.add(Dropout(0.1862112266098212))
    model.add(Dense(units=1, activation='relu'))
    model.add(Dropout(0.03168534114119703))
    model.add(Dense(units=1, activation='relu'))
    model.add(Dense(units=2, activation="softmax"))

    
    ### optimizing for classification###
    # try lr=0.1 or 0.01
    opt = tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta')
   
    
    ###optimizing with absolute values
    model.compile(
        loss='mean_squared_logarithmic_error',
        optimizer=opt,
        metrics=['mape']
    )
    model.summary()
       ### CREATE TENSORBOARD TO DISPLAY THE VALIDATION
    tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))
    filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint(
        "models\\{}.model".format(filepath, monitor='val_loss', verbose=1,
        save_best_only=True, mode='min'))  # saves only the best ones
    print(X_train.shape, y_train.shape)
    ### FIT THE MODEL
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint],
    )
    
    score = model.evaluate(X_val, y_val, verbose=0)


    
    # Save model
    model.save("models\\{}".format(NAME))

    day+= 1

