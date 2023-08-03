#!/usr/bin/env python
# coding: utf-8

# In[1]:


from warnings import filterwarnings
filterwarnings("ignore")


# In[2]:


import pandas as pd


# In[3]:


A= pd.read_csv("C:/Users/barsh/Downloads/Restaurant_Reviews.tsv",sep="\t")


# In[4]:


A.head(10)


# # preprocessing

# In[5]:


Q=[]
for i in A.Review:
    j=i.lower()
    import re
    k = re.sub("[^a-z0-9 ]","",j)
    Q.append(k)


# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
data = cv.fit_transform(Q).toarray()
cols = cv.get_feature_names()


# In[7]:


data.shape


# In[8]:


len(cols)


# In[9]:


A.shape


# # Declare X and Y

# In[10]:


X = data
Y = A.Liked.values


# In[11]:


X.shape


# In[12]:


Y.shape


# # create NN

# In[13]:


from keras.models import Sequential
from keras.layers import Dense,Dropout


# In[14]:


nn = Sequential()
nn.add(Dense(2067,input_dim=(2067)))
nn.add(Dense(2067))
nn.add(Dropout(0.1))
nn.add(Dense(1,activation="sigmoid"))


# In[15]:


nn.compile(loss="binary_crossentropy",metrics="accuracy")
model = nn.fit(X,Y,validation_split=0.2,epochs=10)


# # predictions

# In[16]:


A1=pd.read_csv("C:/Users/barsh/Downloads/rest_reviews_testing.csv")


# In[17]:


A1.head()


# In[18]:


Q = []
for i in A1.Review:
    j = i.lower()
    import re
    k = re.sub("[^a-z0-9 ]","",j)
    Q.append(k)


# In[19]:


data_pred = cv.transform(Q).toarray()


# In[ ]:





# In[20]:


#nn.predict_classes(data_pred)


# In[21]:


#import numpy as np


# In[22]:


#np.argmax(nn.predict(data_pred), axis= 1)


# In[23]:


T = []
for i in nn.predict(data_pred):
    if(round(i[0],2) < 0.5):
        T.append(0)
    else:
        T.append(1)


# In[24]:


T


# In[25]:


A1['Liked']=T


# In[26]:


A1


# In[ ]:





# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()


# In[42]:


tf.fit_transform(A).toarray()


# In[43]:


tf.get_feature_names()


# In[ ]:




