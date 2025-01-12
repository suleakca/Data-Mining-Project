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

features = data.drop(['dteday', 'instant', 'cnt'], axis=1)
target = data["cnt"]


kf = KFold(n_splits=10, shuffle=True, random_state=42)

mae_scores = []
rmse_scores = []
mape_scores = []

for train_index, test_index in kf.split(features):

    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    mape_scores.append(mape)

print("MAE:", round(mean(mae_scores), 2))
print("RMSE:", round(mean(rmse_scores), 2))
print("MAPE:", round(mean(mape_scores), 2), "%")
