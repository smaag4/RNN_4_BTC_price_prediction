import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


## --------------------- config -------------------
## define steps to go back to predict
steps = 50

epochs = 20 

dataset_train = pd.read_csv('../Data/_old Datasets/sample_dataset.csv')

## ---------------------- end config ----------------

#select open and high values only (1:2 = position in dataset = open:1 ; high:2)
training_set = dataset_train.iloc[:, 1:2].values

## Feature scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

## Creating data with timestamps
X_train = []
y_train = []

#take 60 times the open value and append them to x_train to predict 61st open price, which is put into y_train
for i in range(steps, len(dataset_train)):
   
    X_train.append(training_set_scaled[i-steps:i, 0])
    y_train.append(training_set_scaled[i, 0])

# change to numpy array with shape (Amount of rows, features per row)
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)

## Building the LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = epochs, batch_size = 32)

# predict price
predicted_stock_price = regressor.predict(X_test)

#inverse transform the values back
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
y_test = sc.inverse_transform(y_test.reshape(-1,1))


## Plotting the results
plt.plot(y_test, color = 'black', label = 'Bitcoin Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Bitcoin Price')
plt.title('Bitcoin Price Preciction with ' + str(steps) +' steps back')
plt.xlabel('Timesteps')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
