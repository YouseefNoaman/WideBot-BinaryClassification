# Import libraries for data wrangling, preprocessing and visualization
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score

X = pd.read_csv('C:/Users/SefoNoaman/Downloads/binary_classifier_data/training - Copy.csv')
Y = pd.read_csv('C:/Users/SefoNoaman/Downloads/binary_classifier_data/validation - Copy.csv')

# X = pd.read_csv('C:/Users/SefoNoaman/Downloads/binary_classifier_data/training.csv')
# Y = pd.read_csv('C:/Users/SefoNoaman/Downloads/binary_classifier_data/validation.csv')

seed = 7
np.random.seed(seed)

# print(X.dtypes)
# X["variable1"].value_counts()
# X["variable4"].value_counts()
# X["variable5"].value_counts()
# X["variable6"].value_counts()
# X["variable7"].value_counts()
# X["variable9"].value_counts()
# X["variable10"].value_counts()
# X["variable12"].value_counts()
# X["variable13"].value_counts()
# X["variable18"].value_counts()
# X["classLabel"].value_counts()
#
#
# cleanup_nums = {"variable1": {"b": 1, "a": 2},
#                "variable4": {"u": 1, "y":2,"l":3},
#                "variable5": {"g":1,"p":2,"gg":3},
#                "variable6": {"c": 1, "q":2,"cc":3,"x":4,"W": 5, "aa":6, "d":7,"ff":8,"i":9,"m":10,"k":11,"e":12,"j":13,"r":14},
#                "variable7": {"v":1, "h":2, "bb":3, "ff":4,"n":5,"o":6,"z":7,"dd":8,"j":9},
#                "variable9": {"t":1,"f":2},
#                "variable10": {"t":1,"f":2},
#                "variable12": {"t":1,"f":2},
#                "variable13": {"g":1,"s":2,"p":3},
#                "variable18": {"t":1,"f":2},
#                "classLabel": {"yes.":1,"no.":2}}
# The reason for this error is because of the number of times you execute the cell. The first time you execute the cell the transformation happens ('Y' -> 1, and 'N' -> 0). At this point if you try to rerun the same cell, the data type for physicalEvidence and contact is now an int rather than a string. So when the replace method runs, it is look for a 'Y' or 'N', but DOES NOT find any because all of the values have been replaced with 1 or 0.

# If this happens, you'll just need to reload the dataframe (df) again and rerun the cell one time.
# X.replace(cleanup_nums, inplace=True)
# Y.replace(cleanup_nums, inplace=True)
# X.head()
# Y.head()
#
# print(X.dtypes)
# print(X.head())


sns.heatmap(X.corr(), annot=True)
X_train = X.drop(columns=['classLabel'])
y_train = X['classLabel']
X_test = Y.drop(columns=['classLabel'])
y_test = Y['classLabel']

print(X_train[:1])
print(y_train[:1])

print(X_train.shape)
print(y_train.shape)

# defifne a sequentail Model
model = Sequential()

# Hidden Layer-1
model.add(Dense(100, activation='relu', input_dim=18, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None, seed=None))

# Hidden Layer-2
model.add(Dense(10, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3, noise_shape=None, seed=None))

# Output layer
model.add(Dense(1, activation='softmax'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

model_output = model.fit(X_train, y_train, epochs=5, batch_size=50, verbose=1, validation_data=(X_test, y_test), )

print('Training Accuracy : ', np.mean(model_output.history["acc"]))
print('Validation Accuracy : ', np.mean(model_output.history["val_acc"]))

# Plot training & validation accuracy values
plt.plot(model_output.history['acc'])
plt.plot(model_output.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(model_output.history['loss'])
plt.plot(model_output.history['val_loss'])
plt.title('model_output loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

X_train.head()
X_train.dtypes
X_test.head()
y_train.head()
y_test.dtypes

print(X_test)
y_pred = model.predict(X_test)
y_pred.shape
print(y_pred)
rounded = [round(x[0]) for x in y_pred]
y_pred1 = np.array(rounded, dtype='float')
precision_score(y_test, y_pred1)
confusion_matrix(y_test, y_pred1)
y_pred1.shape
model.save("WideBotBinaryCalssifier.h5")
