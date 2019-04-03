#!/usr/bin/env python
# coding: utf-8

# In[2]:


import keras
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import pandas as pd
import numpy as np
import cv2
import os
import csv
import sys

# In[3]:


au_df = pd.DataFrame({'au':[],'target_face':[]})
for actor in os.listdir('/scratch/aditya/cv_data/ravdess_csv/'):
    for file in os.listdir('/scratch/aditya/cv_data/ravdess_csv/'+actor):
        with open(os.path.join('/scratch/aditya/cv_data/ravdess_csv/',actor, file)) as f:
            reader = csv.reader(f)
            next(reader)
            au_df = au_df.append({'au':list(map(float,next(reader))), 'target_face':os.path.join('/scratch/aditya/cv_data/ravdess_faces/',actor, file)}, ignore_index = True)

# In[3]:
print(au_df.columns)

df = pd.DataFrame({'face_path':[],'directory':[],'au':[],'name':[], 'target_face':[]})
for actor in os.listdir('/scratch/aditya/cv_data/faces_dataset/'):
    for file in os.listdir('/scratch/aditya/cv_data/faces_dataset/'+actor):
        idx = np.random.randint(0,1680,10)
        au = au_df['au'].iloc[idx].values
        target = au_df['target_face'].iloc[idx].values
        df_temp = pd.DataFrame({'face_path': [os.path.join(actor, file)]*10, 'directory': ['/scratch/aditya/cv_data/faces_dataset/']*10,'name':[actor]*10,'au':au.flatten(), 'target_face':target})
        df = df.append(df_temp, ignore_index = True)
    print(actor)


# In[4]:


print(df.shape)
df.to_csv('final_df_small.csv')
sys.exit()

# In[7]:


image_gen = ImageDataGenerator(rescale=1./255)


# In[265]:


def createGenerator():
    while True:
        idx = np.random.permutation(df.shape[0])
        df_df = df.iloc[idx]
        batches = image_gen.flow_from_dataframe(df_df, x_col = 'face_path',
                                    y_col = 'name',
                                    directory = '/scratch/aditya/cv_data/faces_dataset/',
                                    target_size = (100,100),
                                    color_mode = 'rgb',
                                    batch_size = 64,
                                    drop_duplicates = False,
                                    shuffle = False)
        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]
            au_coeff = df['au'].values[idx[idx0:idx1]]
            au_final = np.zeros((64,18))
            for i in range(64):
                au_final[i,:] = au_coeff[i]
            yield [batch[0], au_final], [batch[1], batch[0], au_final]
            idx0 = idx1
            if idx1 >= df.shape[0]:
                break


# In[266]:


g = createGenerator()
# train_gen = generator(gen, df, 64)


# In[267]:


next(g)[0][1]
