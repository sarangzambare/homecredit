
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
    

# Cannot be executed yet, the below arguments are only dummies. 
    
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(partial_x_train,partial_y_train,epochs=20  ,batch_size=512,validation_data=(x_val,y_val))

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)

#results = model.evaluate(x_test,y_test)

# print(results)

plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#print(vectorize(train_data))




#print(train_data[0])

# model = model.Sequential()
# model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1,activation='sigmoid'))
    
    
    
    

