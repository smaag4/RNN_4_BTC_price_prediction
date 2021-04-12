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

steps = 17
# threshold for labelling
alpha = 0.0025

# model values
epochs = 150
batch_size = 32

# labbeling values
smallChange = .025
bigChange = .05

### DATASETS
d_dataset = pd.read_csv('Data/complete_dataset_v9.csv')

## ---------------------- end config ----------------


# select open and high values only (1:2 = position in dataset = open:1)
d_dataset_open = d_dataset.iloc[:, 2:].values

print(d_dataset.describe())
### DATA SCALER
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn import preprocessing

# scale bitcoin values
min_max_scaler = MinMaxScaler()
minmax_scale = preprocessing.MinMaxScaler().fit(d_dataset_open)
df_minmax = minmax_scale.transform(d_dataset_open)

# print('scaled data: ', df_minmax)
# maxInColumns = np.amax(df_minmax,axis=0)
# print('max values in column', maxInColumns)

## Creating data with timestamps
dataSequence = []
# labelUp = 0
# labelDown = 0
# labelSame = 0

bigDecrease = 0
smallDecrease = 0
staysTheSame = 0
smallIncrease = 0
bigIncrease = 0

while steps < 18:
    timeNow = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    NAME = f"{steps}-{str(timeNow)}"
    # take x steps the open value and append them to x_train to predict 61st open price, which is put into y_train
    dataSequence = []
    for i in range(steps, len(df_minmax)):
        
        # check if followed step is higher [2] same [1] or lower [0]
        
        today = df_minmax[i-1, 0]
        tomorrow = df_minmax[i, 0]
        # day_after_tomorrow=

        # calculate relative change form tomorrow to today
        change = tomorrow/today
        
        # 1: big decrease
        if change < 1-bigChange:
            ylabel = 0
            bigDecrease += 1
        # 2 small decrease
        elif change < 1-smallChange:
            ylabel = 1
            smallDecrease += 1
        # 3 stays the same
        elif change > 1-smallChange and change < 1+smallChange:
            ylabel = 2
            staysTheSame += 1
        # 4 small increase
        elif change > 1+smallChange and change < 1+bigChange:
            ylabel = 3
            smallIncrease += 1
        # 5 big decrease
        else:
            ylabel = 4
            bigIncrease += 1
        
        """ old labelling process
         ylabel = 1
        if (tomorrow - today) > alpha:
            ylabel = 2
            labelUp+=1
        elif (tomorrow - today) < (-alpha):
            ylabel = 0
            labelDown+=1
        else:
            labelSame+=1
        """
        btc_value = df_minmax[i-steps:i, :df_minmax.shape[1]]
        # print('btc value: ', btc_value)

        # add classifications (up down stays the same) as labels
        # dataSequence.append([btc_value, ylabel])

        # add absolute values as labels
        dataSequence.append([btc_value, tomorrow])
    # random data
    random.shuffle(dataSequence)
    
    print(len(dataSequence))

    # SPLIT DATA SET
    
    # check our labeling
    # print("big decrease: ", bigDecrease)
    # print("small decrease:",smallDecrease)
    # print("stays the same",staysTheSame)
    # print("small increase: ", smallIncrease)
    # print("big increase: ",bigIncrease)
    
    ### SPLIT DATASET INTO TRAINING, VALIDATION AND TEST DATA
    # split data into x and y data
    X = []
    y = []
    
    for seq, target in dataSequence:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (bitcoinprice)
    
    # Split dataset into 70% training, 10% Validation and 20% Test data
    n_cases = len(X)
    n_train = int(round(n_cases*.7))
    n_val = int(round(n_cases*.1))
    n_test = int(round(n_cases*.2))
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:n_cases]
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:n_cases]
    
    ### TRANSFORM DATA INTO NUMPY ARRAY
    # change to numpy array with shape (Amount of rows, features per row)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)
    X_test, y_test = np.array(X_test), np.array(y_test)
    # reshape to same architecture
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    ### BUILDING THE LSTM
    model = Sequential()
    # 1 Layer
    model.add(
        LSTM(units=256, recurrent_dropout=0.1, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    # add(Dropout(0.2))
    model.add(BatchNormalization())
    # 2 Layer
    model.add(LSTM(units=256, recurrent_dropout=0.1, return_sequences=True))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))<
    model.add(BatchNormalization())
    # 3 Layer
    model.add(LSTM(units=256, recurrent_dropout=0.1, return_sequences=True))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # 4 Layer
    model.add(LSTM(units=256, recurrent_dropout=0.1))
    model.add(Dropout(0.1))
    # model.add(Dropout(0.2))
    model.add(BatchNormalization())
    # 5 Layer
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dropout(0.1))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.2))
    
    # 6 Layer
    # model.add(Dense(5, activation='softmax'))

    # 6 layer
    # add dense layer for absolute value prediction
    model.add(Dense(units=1))
    
    ### optimizing for classification###
    # try lr=0.1 or 0.01
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-4)
    # Compile the model
    # model.compile(
    #   loss='sparse_categorical_crossentropy',
    #  optimizer=opt,
    # metrics=['accuracy']

    # )
    # model.summary()
    
    ###optimizing with absolute values
    ##metrics can be exchanged: mean_absolute_percentage_error , mean_absolute_error
    model.compile(
        loss='mean_squared_error',
        optimizer=opt,
        metrics=['mean_absolute_percentage_error', 'mean_absolute_error']
    )
    
    ### CREATE TENSORBOARD TO DISPLAY THE VALIDATION
    tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))
    filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint(
        "models\\{}.model".format(filepath, monitor='mean_absolute_percentage_error, mean_absolute_error', verbose=1,
                                 save_best_only=True, mode='max'))  # saves only the best ones
    
    ### FIT THE MODEL
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard, checkpoint],
    )
    
    """
    # summarize history for loss
    def plot_graphs(history, string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    # summarize history for loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.show()

    """
    
    # Score model
    score = model.evaluate(X_val, y_val, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
  

    steps += 1
    # Save model
    model.save("models\\{}".format(NAME))

# predict price
predicted_stock_price = model.predict(X_test)

# inverse Transform the values back
# add zeros to have the same shape
predicted_stock_price_inverseTransform = np.zeros(shape=(len(predicted_stock_price), d_dataset_open.shape[1]))
predicted_stock_price_inverseTransform[:, 0] = predicted_stock_price[:, 0]
predicted_stock_price_inverseTransform = minmax_scale.inverse_transform(predicted_stock_price_inverseTransform)

# X_test_inverseTransform = np.zeros(shape = (len(X_test),d_dataset_open.shape[1]))
# X_test_inverseTransform[:,0] = X_test[:,0,0]
# X_test_inverseTransform = minmax_scale.inverse_transform(X_test_inverseTransform)
# inverse transform the values back
# predicted_stock_price =min_max_scaler.inverse_transform(predicted_stock_price)
# y_test = min_max_scaler.inverse_transform(y_test.reshape(-1,1))

y_test_inverseTransform = np.zeros(shape=(len(y_test), d_dataset_open.shape[1]))
y_test_inverseTransform[:, 0] = y_test[:]
y_test_inverseTransform = minmax_scale.inverse_transform(y_test_inverseTransform)

testValues2 = np.zeros(shape=(len(y_test), 3))

"""
#createArray with test values
testValues = np.zeros(shape = (len(X_test),3))
testValues2 = testValues
# insert actual Values
testValues[:,0] =  X_test_inverseTransform[:,0]
testValues[:,1] =  predicted_stock_price_inverseTransform[:,0]
testValues[:,2] = np.divide(testValues[:,0],np.absolute(np.subtract(X_test_inverseTransform[:,0],predicted_stock_price_inverseTransform[:,0])))
"""

# print('TestWerte',testValues)
# print('Durchschnittsabweichung', testValues[2].mean()/100,'%')

# insert actual Values

# real value
testValues2[:, 0] = y_test_inverseTransform[:, 0]
# predicted value
testValues2[:, 1] = predicted_stock_price_inverseTransform[:, 0]

# relative difference of actual and predicted value
testValues2[:, 2] = np.divide(np.absolute(np.subtract(testValues2[:, 0], testValues2[:, 1])), testValues2[:, 0])

# calculate average relative difference over all predictions
totalAveragePrediction = np.mean(testValues2, axis=0)[2]
print('TestWerte2', testValues2)
print('Durchschnittsabweichung', totalAveragePrediction*100, '%')

# convert array into dataframe
DF = pd.DataFrame(testValues2)
DF.columns = ['Real Value', 'Predicted Value', 'Relative Difference']
# save the dataframe as a csv file
DF.to_csv("Data/Output/trainedData.csv", index=False)