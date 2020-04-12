import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
from os import path
import os

def preprocess(positive, negative):

    non_target_ds = pd.read_csv(negative).assign(target=negative.split('.')[0].split('/')[1])
    target_ds = []
    for i, ds in enumerate(positive):
        target_ds.append(pd.read_csv(ds).assign(target=ds.split('.')[0].split('/')[1]))

    # target_ds = pd.read_csv('raise_up_both_hands_data.csv').assign(target=1)
    # non_target_ds = pd.read_csv('random_pose_data.csv').assign(target=0)

    target_ds = pd.concat(target_ds, ignore_index=True)
    ds = target_ds.append(non_target_ds, ignore_index=True)

    return ds


def model_building(ds):

    # Feature Engineering-----------------
    X = ds.iloc[:, 0:51].values
    y = ds.iloc[:, 51].values

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    RFC.fit(X, y)

    if path.isdir('saved_model'):
        pass
    else:
        os.mkdir('saved_model')
    with open('saved_model/model.pickle', 'wb') as f:
        pickle.dump(RFC, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive model building for classifying customised posture")
    parser.add_argument("--positive", nargs='+', help="Path(s) of target dataset", required=True)
    parser.add_argument("--negative",  help="Path of non-target dataset")
    args = parser.parse_args()

    target, non_target = args.positive, args.negative

    ds = preprocess(target, non_target)
    model_building(ds)

    if path.isfile('saved_model/model.pickle'):
        print('model is saved.')
