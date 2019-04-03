import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import to_categorical
from keras.layers import LeakyReLU, ReLU
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Dropout, Input, Flatten, Activation, BatchNormalization, regularizers, MaxPooling2D
import pandas as pd


au_df = pd.read_csv('/scratch/aditya/auc_df.csv')
au_df['face_path'] = au_df['face_path'].apply(lambda x : x[:-3] + 'bmp')
au_df


# In[84]:


imagegen = ImageDataGenerator(rescale=1./255)


# In[88]:


gen = imagegen.flow_from_dataframe(au_df, x_col = 'face_path',
                                   y_col = ['au1','au2','au3','au4','au5','au6','au7','au8','au9','au10','au11','au12','au13','au14','au15','au16','au17','au18'],
                                   directory = '/scratch/aditya/',
                                   target_size = (100,100),
                                   color_mode = 'rgb',
                                   batch_size = 64,
                                   class_mode='other'
                                  )

test_gen = imagegen.flow_from_dataframe(au_df, x_col = 'face_path',
                                   y_col = ['au1','au2','au3','au4','au5','au6','au7','au8','au9','au10','au11','au12','au13','au14','au15','au16','au17','au18'],
                                   directory = '/scratch/aditya/',
                                   target_size = (100,100),
                                   color_mode = 'rgb',
                                   batch_size = 1,
                                   class_mode='other'
                                  )



batch_size = 64
epochs = 300


input_layer = Input(shape = (100,100,3))
## Layer 1
x = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation=None, input_shape=(100,100,3))(input_layer)
x = Conv2D(64, kernel_size=(5,5), strides=(1,1), padding='same', activation=None)(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

## Layer 2
x = Conv2D(96, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
x = Conv2D(96, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

## Layer 3
x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
y = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=None)(x)
z = MaxPooling2D(pool_size=(3,3), strides=(2,2))(y)

z = Flatten()(z)

## Layer 4
z = Dense(1024,activation=None)(z)
z = LeakyReLU(alpha=0.1)(z)
z = Dropout(0.5)(z)

## Layer 5
z = Dense(1024,activation=None)(z)
z = LeakyReLU(alpha=0.1)(z)
z = Dropout(0.5)(z)

## Layer 6
z = Dense(18,activation=None)(z)
z = Activation('sigmoid')(z)

# model = Model(inputs = [input_layer], outputs = [z])

adam_ = optimizers.Adam(lr=1e-4)
model.compile(loss='mean_squared_error', optimizer=adam_, metrics=['accuracy'])
model.summary()

model.fit_generator(gen, epochs=epochs, steps_per_epoch = 200)
q,w = next(test_gen)
print(model.predict(q),w)

score = model.evaluate(X_test, y_test, verbose=0)
model.save('complete_aue.h5')

# ### Deleting 9 layers
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
model.layers.pop()
new_model = Model(inputs=[input_layer], outputs = [y])
new_model.compile(loss='categorical_crossentropy', optimizer=adam_, metrics=['accuracy'])
new_model.set_weights(model.get_weights())
new_model.summary()

new_model.save('action_unit_model.h5')

