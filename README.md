# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Keerthana C
RegisterNumber: 212224220047
*/
```
```

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```
## Output:

Data Head:

<img width="390" height="265" alt="image" src="https://github.com/user-attachments/assets/92cd8fae-d35e-4210-bf32-9c76ae8b2cda" />

Data Info:

<img width="603" height="237" alt="image" src="https://github.com/user-attachments/assets/37205034-b458-4a01-8658-23e123bde8ce" />

isnull() sum():

<img width="201" height="88" alt="image" src="https://github.com/user-attachments/assets/950a7477-add4-41e4-ab8e-a939a9eebf29" />

Data Head for salary:

<img width="323" height="234" alt="image" src="https://github.com/user-attachments/assets/6b084354-323c-434f-a394-e846017fb144" />

Mean Squared Error :


<img width="239" height="38" alt="image" src="https://github.com/user-attachments/assets/7017f6a8-a9b0-4e28-8fee-6def00b8c28a" />


r2 Value:


<img width="1065" height="41" alt="image" src="https://github.com/user-attachments/assets/a684592d-1458-4aa5-a568-149448a2c798" />


Data prediction :


<img width="311" height="38" alt="image" src="https://github.com/user-attachments/assets/4bb8a477-b1d8-4438-ab4d-9165949e158a" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
