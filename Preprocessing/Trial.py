# -*- coding: utf-8 -*-


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset and reading data
dataset = pd.read_csv("Preprocessing/Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#filling missing values
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='median')
imp = imp.fit(x[:,1:3])
x[:,1:3] = imp.transform(x[:,1:3])


#Chaning string categorical data to int 
"""
col1   ==> A    B   C
A          1    0   0
B          0    1   0
A          1    0   0
B          0    1   0
C          0    0   1
C          0    0   1
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder()
x[:,0]=le.fit_transform(x[:,0])
transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [0])],remainder='passthrough')
x = np.array(transformer.fit_transform(x), dtype=np.float)
le2 = LabelEncoder()
y = le2.fit_transform(y)

#spliting dataset into training and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)


#Feature Scaling
#Sometimes two different columns may have a very large difference in their 
#data range like here we have age and salary values are numeric but
#the difference is large so we need to nromalize the values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)













