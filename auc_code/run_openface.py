#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess


# In[2]:


names = os.listdir('/media/karan/My Passport/frames1')


# In[8]:


for name in names:

    print(name)
    try:
        subprocess.run(['../../OpenFace/build/bin/FaceLandmarkImg','-fdir','/media/karan/My Passport/frames1/'+name,'vis_align','-nomask','-vis_aus','-simsize','100','-out_dir','../../experession_au/'+name])
    except:
        pass

# In[ ]: