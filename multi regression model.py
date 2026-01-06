

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r'C:\Users\Aniruddha Sultane\Downloads\Investment.csv')

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)

m=regressor.coef_
print()

c = regressor.intercept_
print(c)

X=np.append(arr= np.ones((50,1)).astype(int),values=X, axis=1)

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,2,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()

import statsmodels.api as sm
X_opt = X[:,[0,1]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
regressor_OLS.summary()