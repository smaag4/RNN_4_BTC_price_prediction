import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

### Read dataset
df = pd.read_csv('../Data/_old Datasets/sample_dataset.csv', delimiter=",")

# Count entries in dataset
no_entries = df["timestamp"].count()
print("Number of entries: ",no_entries)

### Calculate the mid price for the specific timestamp

midPrice = []
for t in range(no_entries):
    midPrice.append((df["open"][t]+df["close"][t])/2)
midPrice = np.asarray(midPrice)
# Create column Mid_price in data
df['mid_price'] = midPrice


### Add the price from the day before to the current timestamp
df['open_from_day_before'] = df['open'].shift(1)

### Add the procentual change from the day before to each timestamp
df['change_rate'] = (df['open'] / df ['open_from_day_before']) -1

### First row has a NaN -> Drop the data
df = df.dropna()

changes = df['change_rate']

### LABEL CREATION, ALPHA AND K SET

labels = []
alpha = 1.
k = 20 #number of timestamps which should be included
for i in range(0, len(changes) - k):
    if i < (no_entries-k) and i > k:
        kPrev=0
        kFollow=0
        for stepK in range(1,k+1):
            kPrev=kPrev+midPrice[i-stepK]
            kFollow=kFollow+midPrice[i+stepK]
        kPrev=kPrev/k
        kFollow=kFollow/k

        if kFollow>(kPrev+alpha):
            labels.append(2)
        elif kFollow<(kPrev-alpha):
            labels.append(0)
        else:
            labels.append(1)
    else:
        labels.append(1)

#print(labels)
print("Number of Os: ", labels.count(0))
print("Number of 1s: ", labels.count(1))
print("Number of 2s: ", labels.count(2))

labels = np.asarray(labels)
labels = to_categorical(labels)

### Prepare data: Fill X and Y
#input shape: (number_of_train_examples, sequence_length (bei uns k), input_dim(wieviele dimensionen -> open + close +midprice..))
X = []
Y = []

for i in range(0, len(changes) - k):
    Y.append(changes[i+1])
    X.append(np.array(changes[i+1:i+k+1]))
X = np.array(X)

### NORMALIZATION to split data into test, validation and training data

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=False)
_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

#reshape da wir jeweils nur 1 Datenwert nehmen.. Wenn mehrere dann müsste die letzte 1 gewechselt wird

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
# X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
# Y = np.array(Y)
# y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
# y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))
#X_train= X_train.reshape(-1, 1, 20)
X_test  = X_test.reshape(-1, 1, 20)
y_train = y_train.reshape(-1, 1, 3)
y_test = y_test.reshape(-1, 1, 3)

### Build LSTM and train it

number_of_runs = 10

for run in range(10,number_of_runs+1):

    model = Sequential()
    #input shape = 20 Zeitschritte und dazu jeweils 1 Wert
    model.add(LSTM(50, activation='tanh', dropout=0.2, return_sequences=True, input_shape=(1, X_train.shape[2:])))
    #model.add(LSTM(15, activation='tanh', dropout=0.0, recurrent_dropout=0.0, input_shape=(1, X_train.shape[2])))
    # model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    #model.add(Flatten())
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
    #model.fit(X, Y, batch_size=32, epochs=10)

    print('X train', X_train.shape)
    print('X test', X_test.shape)
    print('Y train', y_train.shape)
    print('Y test', y_test.shape)
    predictions = model.predict(X)
    predictions = predictions.reshape(-1)
    # Add 20 zeros, so that it runs until the end -> Otherwise it wouldn't be able to predict the last 20 entries
    predictions = np.append(predictions, np.zeros(20))
    # Add predictions to the data
    df['predictions'] = predictions
    df['predicted_open'] = df['open_from_day_before'] * (1 + df['predictions'])

    # zum testen X-Achse einfach mit der Zeilen ID beschriftet
    plt.plot(df.index, df['open'], label='original')
    plt.plot(df.index, df['predicted_open'], label='prediction')
    plt.legend()
    print(plt.show())

    print(df.head)

