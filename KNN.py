import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold


data = pd.read_csv("hour.csv")

features = ["season","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","casual","registered"]
target = "cnt"

X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

knn_model = KNeighborsRegressor(n_neighbors=5)  
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"KNN Model Results:")
print(f"MAE: ", mae)
print(f"RMSE: ", rmse)
print(f"MAPE: ", mape)





