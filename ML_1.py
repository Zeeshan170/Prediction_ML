#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv(r"C:\Users\muham\Downloads\insurance_data.csv")
df.head()


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age','affordibility']],df.bought_insurance,test_size=0.2, random_state=25)


# In[8]:


X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100


# In[9]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=50)


# In[10]:


import numpy as np

input_data = np.array([[45, 0.8]])

# Scale the input data
input_data_scaled = input_data.copy()
input_data_scaled[:, 0] = input_data_scaled[:, 0] / 100


# In[11]:


prediction = model.predict(input_data_scaled)

# Print the prediction
print("Prediction:", prediction)


# In[12]:


import numpy as np
import keras

# Assuming the model is already trained and X_train_scaled is available

# Example input: age = 45, affordibility = 0.8
input_data = np.array([[45, 0.8]])

# Scale the input data
input_data_scaled = input_data.copy()
input_data_scaled[:, 0] = input_data_scaled[:, 0] / 100

# Make the prediction
prediction = model.predict(input_data_scaled)

# Print the prediction
print("Prediction:", prediction)

# Threshold the prediction to get a binary output if needed
predicted_class = (prediction > 0.5).astype(int)
print("Predicted class:", predicted_class)


# In[ ]:




