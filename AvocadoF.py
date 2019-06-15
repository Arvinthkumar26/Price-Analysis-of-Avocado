#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 08:44:57 2019

@author: Arvinthkumar
"""

import numpy as np 
import pandas as pd 
df = pd.read_csv('avocado.csv')

# drop unnamed column and rename undefined columns;
df.drop('Unnamed: 0',axis=1,inplace=True)

df.head()

df.info()

df['region'].nunique()

df['type'].nunique()

df_final=pd.get_dummies(df.drop(['region','Date'],axis=1),drop_first=True)

X=df_final.iloc[:,1:11]
y=df_final['AveragePrice']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)

from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
lr.score(X_test, y_test)


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
pred=dtr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
dtr.score(X_test, y_test)


from sklearn.ensemble import RandomForestRegressor
rdr = RandomForestRegressor()
rdr.fit(X_train,y_train)
pred=rdr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, pred))
print('MSE:', metrics.mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))
rdr.score(X_test, y_test)