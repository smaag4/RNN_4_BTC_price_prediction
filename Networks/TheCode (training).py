import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import tensorflow
#import tensorflow-gpu as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Flatten
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K

from sklearn.model_selection import train_test_split

# pd.set_option('display.mpl_style', 'default) # Make the graphs a bit prettier
# plt.rcParams['figure.figsize'] = (15, 5)

#trades10 = pd.read_cs ''./Datasetsigdax_btc_usd_snapshots1.csv')
trades10 = pd.read_csv('../Data/_old Datasets/sample_dataset.csv')


#print(trades10[:5])
trades10["timestamp"].plot() # PLOT TIMESTAMPS
# trades10["timestamp"]

trades10["open"].plot() # PLOT PRICES OF THE DATASET

# DATA EXTRACTION

# NORMALIZATION

# FROM FILE - EASY WAY OUT
numpy_array = np.genfromtxt("../Data/_old Datasets/sample_dataset.csv", delimiter=",", skip_header=1)
np.save("np_sample_dataset.npy", numpy_array)
X = np.load("np_sample_dataset.npy")

no_entries = trades10["timestamp"].count()
print("no_entries",no_entries)
# Mid price im for timestep t:
midPrice = []
for t in range(no_entries):
    midPrice.append((trades10["open"][t]+trades10["close"][t])/2)
midPrice = np.asarray(midPrice)

print(midPrice)

# LABEL CREATION, ALPHA AND K SET

labels = []
alpha = 1.
k = 20
for i in range(no_entries):
    if i<(no_entries-k) and i>k:
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


# NORMALIZATION
print('X: ', X.shape)
print('labels: ', labels.shape)
 #TRAIN TEST VALIDATION SPLIT
 #

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle = False)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle = False)


number_of_runs = 10

for run in range(10,number_of_runs+1):

    # BUILDING THE NETWORK

    model = Sequential()

    #model.add(LSTM(15, activation='tanh', dropout=0.0, recurrent_dropout=0.0, input_shape=(1,X_train.shape[1],)))
    #model.add(Flatten())
    #model.add(Dense(3, activation='softmax'))
 
    model.add(LSTM(15, activation='tanh', dropout=0.0, recurrent_dropout=0.0, input_shape=(1, X_train.shape[2])))
    # model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    #model.add(Flatten())
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])

    print(model.summary())

    # TRAINING

    history = model.fit(X_train, y_train, epochs=100, batch_size=64, shuffle=False, validation_data=(X_val, y_val))

    # TESTING

    evaluation = model.evaluate(x=X_test, y=y_test, batch_size=64, verbose=1, sample_weight=None, steps=None)
    print('Results (test loss, test acc): ', evaluation)

    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.show()
    # Building the CONFUSION MATRIX

    predictions = model.predict(X_test, batch_size=64, verbose=0, steps=None)

    predictions = np.argmax(predictions, axis=1)
    grounds = np.argmax(y_test, axis=1)

    print("Grounds", grounds)
    print("Prediction", predictions)

    confusion_matrix = np.zeros((3, 3))

    for i in range(len(predictions)):
        x = grounds[i]
        y = predictions[i]
        confusion_matrix[x][y] += 1

    print("Confusion matrix", confusion_matrix)

    values = []
    for l in range(3):
        p = confusion_matrix[l][l] #number of correctly assigned to l class
        print("correctly assigned values", l, p)
        n = 0 #number of cases correctly predicted to be not class 'a'
        u = 0 #number of cases which should have been predicted to be class 'a' -> false negatives
        o = 0 #number of cases predicted to be class a which weren't -> false positives
        for i in range(3):
            for j in range (3):
                if i != l and j != l:
                    n = n +confusion_matrix[i][j]
                if i != l and j ==l:
                    u = u + confusion_matrix[i][j]
                if i ==l and j != l:
                    o = o + confusion_matrix[i][j]
        ## Calculate Matthews Correlation Coefficient (MCC)
        mcc = (p*n - u*o)/((p+u)*(p+o)*(n+u)*(n+o))**(1/2)
        mcc1 = (p*n - u*o)
        mcc2 = ((p+u)*(p+o)*(n+u)*(n+o))**(1/2)
        values.append(mcc)
    print("mcc1: ", mcc1)
    print("mcc2: ", mcc2)
    print("values",values)

    final_coefficient = (values[0] + values[1] + values[2])/3
    a = values[0]
    b = values[1]
    c = values[2]

    print("final coefficient",final_coefficient)

    plt.clf()
    print('history values', history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model train vs validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('Data/Output/Run%d K%d alpha%.2f accuracy.png' %(run, k, alpha))
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train cs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('Data/Output/Dataset1 Run%d K%d alpha%.2f loss.png' %(run, k, alpha))
    plt.clf()

    maif = open('../accuracy.txt', 'a')
    maif.write('Run number %d k=%d alpha=%.2f Accuracy=%.2f\t Matthews Coefficient = %.4f (%.2f, %.2f, %.2f) \n' % (run, k, alpha, evaluation[1], final_coefficient, a, b, c))
    maif.close()

    model.save('Data/Output/run%dk%dalpha%.2f.h5' % (run, k, alpha))

    K.clear_session()
