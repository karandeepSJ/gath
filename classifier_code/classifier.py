import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
import pickle
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras import models
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras import optimizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, Activation, BatchNormalization, regularizers


### Reading pickled data
pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)
pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open("X_validate.pickle","rb")
X_validate = pickle.load(pickle_in)
pickle_in = open("y_validate.pickle","rb")
y_validate = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)
pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

### Normalizing Data
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_validate = tf.keras.utils.normalize(X_validate,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)


print("Shape of Training data " + str(X_train.shape))
print("Shape of Validation data " + str(X_validate.shape))
print("Shape of Test data " + str(X_test.shape))
##### Changes
number_of_classes = 200
batch_size = 64
epochs = 20

model = models.Sequential()

### Layer 1
model.add(Conv2D(32, kernel_size=(5,5), strides=(2,2), padding='same', activation=None, input_shape=(100,100,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

### Layer 2
model.add(Conv2D(64, kernel_size=(5,5), strides=(2,2), padding='same', activation=None))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

### Layer 3
model.add(Conv2D(128, kernel_size=(5,5), strides=(2,2), padding='same', activation=None))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

### Layer 4
model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='same', activation=None))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

### Layer 5
model.add(Conv2D(512, kernel_size=(3,3), strides=(2,2), padding='same', activation=None))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))

model.add(Flatten())
## Layer 6
model.add(Dense(1024,activation=None))
model.add(LeakyReLU(alpha=0.1))

### Layer 7
model.add(Dense(number_of_classes,activation=None))
model.add(Activation('sigmoid'))

adam_ = optimizers.Adam(lr=0.0001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam_,
              metrics=['accuracy'])

model.summary()

model.fit(X_train,y_train, validation_data=(X_validate,y_validate), batch_size=batch_size, epochs=epochs)
score = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict_classes(X_test,verbose = 1)

### Displaying Confusion matrix
mat = confusion_matrix(y_test, y_pred)
# df_cm = pd.DataFrame(mat, range(number_of_classes),range(number_of_classes))
# plt.figure(figsize = (200,200))
# sn.heatmap(df_cm, annot=True)
# plt.savefig('confusion_matrix.png')
pickle_out = open("confusion_matrix.pickle","wb")
pickle.dump(mat,pickle_out)
pickle_out.close()

print('Test loss:', score[0])
print('Test accuracy:', score[1])