
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from itertools import product
import time


# In[2]:


item_categories = pd.read_csv('item_categories.csv')
items = pd.read_csv('items.csv')
sales_train = pd.read_csv('sales_train.csv')
sample_submission = pd.read_csv('sample_submission.csv')
shops = pd.read_csv('shops.csv')
test = pd.read_csv('test.csv')


# In[3]:


item_categories


# In[4]:


items


# In[5]:


sales_train


# In[6]:


shops


# In[7]:


test


# ## 資料前處理

# In[8]:


print('----------head---------')
print(sales_train.head(5))
print('------information------')
print(sales_train.info())
print('-----missing value-----')
print(sales_train.isnull().sum())
print('--------nan value------')
print(sales_train.isna().sum())


# In[9]:


sales_by_item_id = sales_train.pivot_table(index=['item_id'], values=['item_cnt_day'], 
                                           columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)
sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)
sales_by_item_id.columns.values[0] = 'item_id'
sales_by_item_id


# In[10]:


# 近 12 個月沒有銷量的 item
outdated_items = sales_by_item_id[sales_by_item_id.loc[:, '21':].sum(axis=1) == 0]
print('Outdated items:', len(outdated_items))


# In[11]:


# test 資料集共有 5100 項 item，其中有 84 項 item 近 12 個月銷量為 0
print('unique items in test set:', test['item_id'].nunique())
print('Outdated items in test set:', test[test['item_id'].isin(outdated_items['item_id'])]['item_id'].nunique())


# In[12]:


# 合併訓練集中每個 item id 對應的 category
categories = items[['item_id','item_category_id']]
sales_train_merge_cat = pd.merge(sales_train, categories, on = 'item_id', how = 'left')
sales_train_merge_cat


# In[13]:


# 畫出離群值
plt.figure(figsize=(10,4))
plt.xlim(-100,3000)
sns.boxplot(x = sales_train['item_cnt_day'])
print('Sale volume outliers:',sales_train['item_cnt_day'][sales_train['item_cnt_day']>1001].unique())
plt.figure(figsize=(10,4))
plt.xlim(-10000,320000)
sns.boxplot(x = sales_train['item_price'])
print('Sale price outliers:',sales_train['item_price'][sales_train['item_price']>300000].unique())


# In[14]:


sales_train = sales_train[sales_train['item_cnt_day'] < 1001]
sales_train = sales_train[sales_train['item_price'] < 300000]
plt.figure(figsize=(10,4))
plt.xlim(-100,3000)
sns.boxplot(x = sales_train['item_cnt_day'])

plt.figure(figsize=(10,4))
plt.xlim(-10000,320000)
sns.boxplot(x = sales_train['item_price'])


# In[15]:


# 將 item_price < 0 的資料用 median price取代 
# (相同 date_block_num、shop_id、item_id)
sales_train[sales_train['item_price'] < 0]


# In[16]:


median = sales_train[(sales_train['date_block_num'] == 4) & (sales_train['shop_id'] == 32)                     & (sales_train['item_id'] == 2973) & (sales_train['item_price']>0)].item_price.median()
sales_train.loc[sales_train['item_price']<0,'item_price'] = median
print(median)


# ## test 資料集

# In[17]:


good_sales = test.merge(sales_train, on=['item_id','shop_id'], how='left').dropna()
good_pairs = test[test['ID'].isin(good_sales['ID'])]
no_data_items = test[~(test['item_id'].isin(sales_train['item_id']))]    # 訓練集有出現過、但測試集沒有出現過的 item
print('1. Number of good pairs:', len(good_pairs))
print('2. No Data Items:', len(no_data_items))
print('3. Only Item_id Info:', len(test)-len(no_data_items)-len(good_pairs))


# In[18]:


test['ID'].isin(good_sales['ID'])


# In[19]:


sales_train.head(10)


# In[20]:


shops


# In[21]:


test


# ### 商店編碼

# In[22]:


shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()
shops['shop_city'] = shops['shop_name'].str.partition(' ')[0]
shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'NO_DATA')
shops


# In[23]:


shops['shop_city_code'] = LabelEncoder().fit_transform(shops['shop_city'])
shops['shop_type_code'] = LabelEncoder().fit_transform(shops['shop_type'])
shops


# In[24]:


lines1 = [26,27,28,29,30,31]
lines2 = [81,82]
for index in lines1:
    category_name = item_categories.loc[index,'item_category_name']
#     print(category_name)
    category_name = category_name.replace('Игры','Игры -')
#     print(category_name)
    categories.loc[index,'item_category_name'] = category_name

for index in lines2:
    category_name = item_categories.loc[index,'item_category_name']
#    print(category_name)
    category_name = category_name.replace('Чистые','Чистые -')
#    print(category_name)
    categories.loc[index,'item_category_name'] = category_name

category_name = item_categories.loc[32,'item_category_name']
#print(category_name)
category_name = category_name.replace('Карты оплаты','Карты оплаты -')
#print(category_name)
item_categories.loc[32,'item_category_name'] = category_name


# In[25]:


item_categories


# In[26]:


item_categories['split'] = item_categories['item_category_name'].str.split('-')
item_categories['type'] = item_categories['split'].map(lambda x:x[0].strip())
item_categories['subtype'] = item_categories['split'].map(lambda x:x[1].strip() if len(x) > 1 else x[0].strip())
item_categories = item_categories[['item_category_id','type','subtype']]
item_categories.head()


# In[27]:


items = items.drop(columns = ['item_name'])
items


# In[28]:


items = items.merge(item_categories, how='left', on='item_category_id')


# In[29]:


items


# In[30]:


labelencoder = LabelEncoder()
items['type'] = labelencoder.fit_transform(items['type'])
items['subtype'] = labelencoder.fit_transform(items['subtype'])
items = items.drop(columns=['item_category_id'])
items


# In[31]:


items


# In[32]:


sales_train.columns


# In[33]:


train = sales_train[['date_block_num', 'item_price', 'shop_id','item_id','item_cnt_day']].groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_day':'sum'}).reset_index()
train = train.rename(columns={'item_cnt_day' : 'item_cnt_month'})


# In[34]:


train = train.merge(shops[['shop_id', 'shop_city_code', 'shop_type_code']], how='left', on=['shop_id'])


# In[35]:


train


# In[36]:


train = train.merge(items,how='left',on='item_id')


# In[37]:


train


# In[38]:


X_train = train[['shop_city_code', 'shop_type_code','type','subtype']]


# In[39]:


X_train


# In[40]:


y_train = train['item_cnt_month']


# In[41]:


y_train


# In[42]:


test = test.merge(shops[['shop_id', 'shop_city_code', 'shop_type_code']], how='left', on=['shop_id'])


# In[43]:


test = test.merge(items,how='left',on='item_id')


# In[44]:


test


# In[45]:


X_test = test[['shop_city_code', 'shop_type_code','type','subtype']]


# In[46]:


X_test


# In[50]:


import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()


# In[60]:


X_train.shape


# In[63]:


X_train, y_train = np.array(X_train), np.array(y_train)


# In[64]:


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[52]:


# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 16, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))


# In[53]:


# Adding the output layer
regressor.add(Dense(units = 1))


# In[54]:


# Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[55]:


regressor.summary()


# In[65]:


# 進行訓練
regressor.fit(X_train, y_train, epochs = 10, batch_size = 32)


# In[68]:


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[73]:


pred = regressor.predict(X_test)


# In[103]:


pred.reshape(214200)


# In[102]:


pred.shape


# In[105]:


# submits

submit = pd.DataFrame({"ID": test['ID'],
                      "item_cnt_month":pred.reshape(214200).tolist()})
submit.to_csv("submit_Base.csv",index=False)

