#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import scipy.io as io


# In[4]:


metadata = io.loadmat('celebrity2000_meta.mat')


# In[8]:


data = metadata['celebrityData']
image_data = metadata['celebrityImageData']


# In[51]:


name_df = pd.DataFrame({'id':[],'name':[]})
image_df = pd.DataFrame({'id':[],'path':[]})


# In[54]:


name_df['id'] = data[0][0][1].flatten()
names = data[0][0][0].flatten()
for i in range(len(names)):
    names[i] = names[i][0]
name_df['name'] = names


# In[79]:


image_df['id'] = image_data[0][0][1].flatten()
paths = image_data[0][0][7].flatten()
for i in range(len(paths)):
    paths[i] = paths[i][0]
image_df['path'] = paths


# In[97]:


import os
names.sort()
for name in names:
    os.mkdir(os.path.join('CACD2000',name.replace('_',' ')))


# In[112]:


import shutil
for c_id,path in image_df.values:
    name = name_df[name_df['id']==c_id]
    folder = name['name'].values[0].replace('_',' ')
    shutil.move('CACD2000/'+path,os.path.join('CACD2000',folder,path))


# In[151]:


import numpy as np
uniq_ids, cou = np.unique(image_data[0][0][1].flatten(),return_counts=True)


# In[137]:


ix = np.argsort(cou)[::-1]


# In[175]:


dirs_to_select = name_df[name_df['id'].isin(uniq_ids[ix[:200]])].name.values


# In[178]:


for i, name in enumerate(dirs_to_select):
    print(i)
    shutil.copytree(os.path.join('CACD2000',name.replace('_',' ')),os.path.join('cacd_dataset',name.replace('_',' ')))    


# In[ ]:




