#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


train=pd.read_csv('sales_train.csv')
test=pd.read_csv('test.csv')

train.info()
test.info()


# In[3]:


# 計算相關性
corrmat = train.corr()
k = 10 
cols = corrmat.nlargest(k, 'item_cnt_day')['item_cnt_day'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[4]:


#挑選特徵值
selected_features = ['item_id','shop_id' ]

X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['item_cnt_day']

# 進行特徵向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
X_test = dict_vec.transform(X_test.to_dict(orient='record'))


# In[5]:


# 用GradientBoostingRegressor進行預測
from sklearn.ensemble import GradientBoostingRegressor
rfr = GradientBoostingRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)


rfr_submission = pd.DataFrame({'ID': test['ID'], 'item_cnt_month': rfr_y_predict})
rfr_submission.to_csv('sub.csv', index=False,sep=',')


# In[ ]:




