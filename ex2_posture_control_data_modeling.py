import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle
from os import path
import os

def preprocess(positive, negative):

    non_target_ds = pd.read_csv(negative).assign(target=0)
    target_ds = []
    for i, ds in enumerate(positive):
        target_ds.append(pd.read_csv(ds).assign(target=i+1))

    # target_ds = pd.read_csv('raise_up_both_hands_data.csv').assign(target=1)
    # non_target_ds = pd.read_csv('random_pose_data.csv').assign(target=0)

    target_ds = pd.concat(target_ds)
    ds = target_ds.append(non_target_ds, ignore_index=True)

    ds.to_csv('result.csv')

    return ds


def model_building(ds):

    # Feature Engineering-----------------
    X = ds.iloc[:, 0:51].values
    y = ds.iloc[:, 51].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    RFC.fit(X_train, y_train)

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
