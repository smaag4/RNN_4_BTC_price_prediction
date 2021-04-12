import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt

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

### Prepare data: Fill X and Y
#input shape: (number_of_train_examples, sequence_length (bei uns k), input_dim(wieviele dimensionen -> open + close +midprice..))
X = []
Y = []
k = 20 # Number of past timestamps which should be included

for i in range(0, len(changes) - k):
    Y.append(changes[i+1])
    X.append(np.array(changes[i+1:i+k+1]))

#reshape da wir jeweils nur 1 Datenwert nehmen.. Wenn mehrere dann müsste die letzte 1 gewechselt wird
X = np.array(X).reshape(-1, 20, 1)
Y = np.array(Y)

### Build LSTM
model = Sequential()
#input shape = 20 Zeitschritte und dazu jeweils 1 Wert
model.add(LSTM(1, input_shape=(20, 1)))
model.compile(optimizer="rmsprop", loss="mse")
model.fit(X, Y, batch_size=32, epochs=10)

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
print(plt.show())

print(df.head)
