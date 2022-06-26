#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd
import numpy as np



def load_model(model_file='model.bin'):

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    
    return dv, lr



categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def predict(month: int, year: int):

    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{int(year):04d}-{int(month):02d}.parquet')

    print("read_data finished")
    dv, lr = load_model(model_file='model.bin')

    print("load_model finished")

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print(np.mean(y_pred))


    # df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # output_file = 'hw4_save.parquet'

    # df_result = df.copy()
    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("month")
    parser.add_argument("year")
    args = parser.parse_args()
 
    predict(args.month, args.year)