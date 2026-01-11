import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"



if not os.path.exists(MODEL_FILE):
    
    data = pd.read_csv("traffic_green_time_dataset_varied.csv")

    def shuffle_and_split(data, test_ratio):
        np.random.seed(42)
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return data.iloc[train_indices], data.iloc[test_indices]

    train_set,test_set = shuffle_and_split(data, 0.2)
    
    test_set.drop("green_time",axis=1).to_csv("input.csv",index=False)

    green_signal_label = train_set["green_time"].copy()
    green_signal_features = train_set.drop("green_time", axis=1)

    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
    ])

    signal_time_prepared = num_pipeline.fit_transform(green_signal_features)
    print(signal_time_prepared)

    model=RandomForestRegressor(random_state=42)
    model.fit(signal_time_prepared,green_signal_label)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(num_pipeline, PIPELINE_FILE)
    print("Model is trained,Congrats!")
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transformed_input=pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    predictions = np.rint(predictions).astype(int)
    predictions = np.clip(predictions,10,120)
    input_data['green_time']=predictions

    input_data.to_csv("output.csv",index=False)
    print("Result save to output.csv")