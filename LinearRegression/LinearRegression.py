#!/usr/bin/env python3
#Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Importing the dataset
# dataset = pd.read_csv('Salary_Data.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values

dataset = pd.read_csv("Salary_Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=54)

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=1/3,random_state=54)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_reg = reg.predict(x_test)

# #Fitting Simple Linear regression model to training set
# from sklearn.linear_model import LinearRegression
# reg = LinearRegression()
# reg.fit(X_train,Y_train)

# #predicting for test cases
# y_reg = reg.predict(X_test)

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
# plt.title()
# #plottig graph for training dataset
# plt.scatter(X_train,Y_train,color='red')
# plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title("Salary vs Experience(Training)")
plt.xlabel("Salary")
plt.ylabel("Years of Experience")
plt.show()

# #plottig graph for test dataset
plt.scatter(x_test,y_test,color='red')
plt.plot(x_test,reg.predict(x_test),color='blue')
plt.title("Salary vs Experience(Test)")
plt.xlabel("Salary")
plt.ylabel("Years of Experience")
plt.show()

# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""



