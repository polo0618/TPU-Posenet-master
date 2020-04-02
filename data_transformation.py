import numpy as np
import pandas as pd
import pickle
import argparse


def open_data(data):
    #with open('data.txt', 'rb') as fp:
    #     l = pickle.load(fp)
    l = []
    with open(data, 'rb') as f:
        while 1:
            try:
                l.append(pickle.load(f))
            except EOFError:
                break
    #len(l)  # 33
    # l[0] #17
    # l[1][0] #3
    return l


def preprocess(data):
    res_data = []
    for pose in data:
        df = pd.DataFrame(pose, columns =['Position', 'Latlon', 'Accuracy Score']).T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        dfs = []
        for name, value in df.iteritems():
            dfs.append(
                pd.DataFrame(
                {'{} latitude'.format(name): value.values[0][0],
                 '{} longitude'.format(name): value.values[0][1],
                 '{} accuracy score'.format(name): value.values[1]
                }, index=[0])
            )
        row = pd.concat(dfs, axis=1)
        res_data.append(row)

    return pd.concat(res_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer data to training/testing dataset")
    parser.add_argument("target",  help="To know whether the data is the targeted posture")
    parser.add_argument("data",  help="data")

    args = parser.parse_args()
    data = preprocess(open_data(args.data))

    data.to_csv('{}_data.csv'.format(args.target), index=False)
    # data.to_csv('training_data.csv') if args.purpose == 'training' else data.to_csv('testing_data.csv')
    print('{} data saved.'.format(args.target))


    data = open_data('random_pose.txt')

    see = preprocess([data[0]])