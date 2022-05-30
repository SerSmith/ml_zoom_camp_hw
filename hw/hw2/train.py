import argparse
import os
import pickle
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    with mlflow.start_run():

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        mlflow.sklearn.autolog(rf)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)



        rmse = mean_squared_error(y_valid, y_pred, squared=False)



if __name__ == '__main__':

    mlflow.set_experiment("my_taxy")

    mlflow.set_tag("try", 'first_try')
    data_path = "/Users/sykuznetsov/Documents/GitHub/ml_zoom_camp_hw/data/data_hw2/data_output"


    run(data_path)