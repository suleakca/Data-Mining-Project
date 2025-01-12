from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.datasets import make_blobs
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from statistics import mean


data = pd.read_csv("hour.csv")

features = ["season", "mnth", "hr", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]  # Example features
target = "cnt"

X = data[features]
y = data[target]

def evaluate_model(model, X_test, y_test):
  y_pred = model.predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = mean_squared_error(y_test, y_pred, squared=False)
  mape = mean_absolute_percentage_error(y_test, y_pred) * 100 
  print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

kf = KFold(n_splits=10, shuffle=True, random_state=42) 

mae_scores, rmse_scores, mape_scores = [], [], []
for train_index, test_index in kf.split(X):
  X_train, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model = GradientBoostingRegressor(random_state=42)
  model.fit(X_train, y_train)

  evaluate_model(model, X_test, y_test)

  mae_scores.append(mean_absolute_error(y_test, model.predict(X_test)))
  rmse_scores.append(mean_squared_error(y_test, model.predict(X_test), squared=False))
  mape_scores.append(mean_absolute_percentage_error(y_test, model.predict(X_test)) * 100)

print("Average Scores:")
print(f"MAE: {sum(mae_scores) / len(mae_scores):.2f}")
print(f"RMSE: {sum(rmse_scores) / len(rmse_scores):.2f}")
print(f"MAPE: {sum(mape_scores) / len(mape_scores):.2f}%")
