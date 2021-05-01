#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# In[52]:


df=pd.read_csv("E:\Machinfy\secion 14/ames_housing_trimmed_processed.csv - ames_housing_trimmed_processed.csv.csv",sep=',',encoding='utf-8')


# In[53]:


df.info()


# In[54]:


df.head()


# In[55]:


df.corr()


# In[56]:


plt.figure(figsize=(10,10))
plt.hist(x=df['SalePrice'],bins=30)


# In[57]:


sns.boxplot(y=df['SalePrice'])


# In[58]:


df['SalePrice'].describe()


# In[59]:


x=df.iloc[:,:-1].values
y=df['SalePrice'].values


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train , x_test ,y_train ,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[62]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[63]:


x_train_scaled=sc.fit_transform(x_train)
x_test_scaled=sc.fit_transform(x_test)


# In[64]:


from sklearn.linear_model import LinearRegression


# In[65]:


regressor=LinearRegression()


# In[66]:


regressor.fit(x_train_scaled,y_train)


# In[67]:


regressor.score(x_train_scaled,y_train)


# In[68]:


regressor.score(x_test_scaled,y_test)


# In[69]:


pred=regressor.predict(x_test_scaled)


# In[70]:


rmse=np.sqrt(((pred-y_test)**2).mean())


# In[71]:


rmse


# In[72]:


xg_reg=xgb.XGBRegressor(objective='reg:linear',n_estimators=10,seed=123)


# In[73]:


xg_reg.fit(x_train_scaled,y_train)


# In[74]:


xg_reg.score(x_train_scaled,y_train)


# In[75]:


xg_reg.score(x_test_scaled,y_test)


# In[76]:


pred2=xg_reg.predict(x_test_scaled)


# In[77]:


rmse2=np.sqrt(((pred2-y_test)**2).mean())


# In[78]:


rmse2


# In[ ]:




