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


data = pd.read_csv("hour.csv")

features = ["season", "mnth", "hr", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]  # Example features
target = "cnt"

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)

dt_model.fit(X_train, y_train)

y_pred = dt_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)

print("Decision Tree Model Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Percentage Error (MAPE):", mape)

cv_scores = cross_val_score(dt_model, X, y, cv=10, scoring='neg_mean_absolute_error')
cv_mae = -cv_scores.mean()
cv_rmse = np.sqrt(-cv_scores).mean()

print("Cross-Validation Mean Absolute Error (CV-MAE):", cv_mae)
print("Cross-Validation Root Mean Squared Error (CV-RMSE):", cv_rmse)

