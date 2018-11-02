
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras import optimizers


# In[ ]:


x_train = [[]]
y_train = []

x_test = [[]]
y_test = []

x_val = [[]]
y_val = []

train_split = 0.7     #fraction of total data used for training.
train_val_split = 0.3 #fraction of train_data used for validation.

def prepare_data(csv_filename):
    df = pd.read_csv(csv_filename)
    
    num_train = int(len(df.iloc[:,0])*train_split)
    
    df_train = df.iloc[:num_train,:]
    
    df_test = df.iloc[num_train:,:]
    
    num_train = int(len(df_train.iloc[:,0]*(1-train_val_split)))
    
    x_train = df_train.iloc[:num_train,2:]
    
    x_val = df_train.iloc[num_train:,2:]
    
    y_train = df_train.iloc[:num_train,1]
    
    y_val = df_train.iloc[num_train:,1]
    
    x_test = df_test.iloc[:,2:]
    
    y_test = df_test.iloc[:,1]
    
    
    
    
    
    

