import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r'C:\Users\Aniruddha\Downloads\Investment.csv')

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,4]

X = pd.get_dummies(X,dtype=int)

from sklearn.model_selection import train_test_split
X_train, X_test, Y-train, Y_test = train_test_split(X,y,test_size=0.2, random_state=0)


