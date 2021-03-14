#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time
from itertools import product
import lightgbm as lgb
import gc
import pickle


# In[2]:


# 這是拿別台電腦經過上面步驟處理的資料，整理成pkl檔
data = pd.read_pickle('data_simple.pkl')


# In[3]:


X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']
X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
Y_true = data[data.date_block_num == 34]['item_cnt_month']


# In[4]:


ts = time.time()
train_data = lgb.Dataset(data=X_train, label=Y_train)
valid_data = lgb.Dataset(data=X_valid, label=Y_valid)

time.time() - ts
    
params = {"objective" : "regression", "metric" : "rmse", 'n_estimators':10000, 'early_stopping_rounds':100,
              "num_leaves" : 200, "learning_rate" : 0.01, "bagging_fraction" : 0.9,
              "feature_fraction" : 0.3, "bagging_seed" : 0}
    
lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100) 
Y_test = lgb_model.predict(X_test).clip(0, 20)


# In[5]:


df_output = pd.DataFrame()
aux = pd.read_csv('sample_submission.csv')
df_output['ID'] = aux['ID']
df_output['item_cnt_month'] = Y_test
df_output[['ID', 'item_cnt_month']].to_csv('feature engineer lightGBM.csv', index=False)


# In[ ]:




