import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
data = pd.read_csv("traffic_green_time_dataset_varied.csv")

# 2. Create a test set 
def shuffle_and_split(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split(data, 0.2)

# Work on a copy of training data
data = train_set.copy()

# 3. Separate predictors and labels
green_signal_time = data["green_time"].copy()
green_time_features = data.drop("green_time", axis=1)

# 4. Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# 5. Transform the data
signal_time_prepared = num_pipeline.fit_transform(green_time_features)

# signal_time_prepared is now a NumPy array ready for training
print("Shape of training features:", signal_time_prepared.shape)

# 6. Train models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(signal_time_prepared, green_signal_time)

# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(signal_time_prepared, green_signal_time)

# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(signal_time_prepared, green_signal_time)

# 7. Predict using training data
lin_preds = lin_reg.predict(signal_time_prepared)
tree_preds = tree_reg.predict(signal_time_prepared)
forest_preds = forest_reg.predict(signal_time_prepared)

# 8. Calculate RMSE
#lin_rmse = root_mean_squared_error(green_signal_time, lin_preds)
lin_rmses = -cross_val_score(lin_reg,signal_time_prepared,green_signal_time,scoring="neg_root_mean_squared_error",cv=10)
#tree_rmse = root_mean_squared_error(green_signal_time, tree_preds)
tree_rmses = -cross_val_score(tree_reg,signal_time_prepared,green_signal_time,scoring="neg_root_mean_squared_error",cv=10)
#forest_rmse = root_mean_squared_error(green_signal_time, forest_preds)
forest_rmses= -cross_val_score(forest_reg,signal_time_prepared,green_signal_time,scoring="neg_root_mean_squared_error",cv=10)

#print("Linear Regression RMSE:", lin_rmse)
#print("Decision Tree RMSE:", tree_rmse)
#print("Random Forest RMSE:", forest_rmse)

print(pd.Series(lin_rmses).describe())
print(pd.Series(tree_rmses).describe())
print(pd.Series(forest_rmses).describe())
