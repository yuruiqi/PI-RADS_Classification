import os
import json
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

from Data.Preprocess import join_path
from Data.Split import Splitter
from Statistics.IndependenceTest import test_ind


def split():
    with open('/homes/rqyu/Data/PI-RADS/data.json') as f:
        all_data = json.load(f)

    casename = [x['casename'] for x in all_data]
    label = [int(x['PI-RADS']) for x in all_data]

    df = pd.DataFrame({'casename':casename, 'label':label})
    splitter = Splitter(df)

    seed = 0
    while True:
        seed += 1
        train, val, test = splitter.split_data(seed=seed)
        splitter.test_ind()

        train_casenames = train['casename'].tolist()
        val_casenames = val['casename'].tolist()
        test_casenames = test['casename'].tolist()

        train_pure_casenames = set([casename[:-2] for casename in train_casenames])
        val_pure_casenames = set([casename[:-2] for casename in val_casenames])
        test_pure_casenames = set([casename[:-2] for casename in test_casenames])

        if len(train_pure_casenames & val_pure_casenames) == 0 and len(test_pure_casenames & val_pure_casenames) == 0 and \
                len(train_pure_casenames & test_pure_casenames) == 0:
            break

    train_json = [all_data['casename'] for casename in train_casenames if (all_data['casename'] == casename)]
    val_json = [all_data['casename'] for casename in val_casenames if (all_data['casename'] == casename)]
    test_json = [all_data['casename'] for casename in test_casenames if (all_data['casename'] == casename)]

    save_dir = '/homes/rqyu/Data/PI-RADS'
    with open(join_path(save_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=1)
    with open(join_path(save_dir, 'val.json'), 'w') as f:
        json.dump(val_json, f, indent=1)
    with open(join_path(save_dir, 'test.json'), 'w') as f:
        json.dump(test_json, f, indent=1)


def test_ind_dict(data1, data2, key):
    data1 = [data[key] for data in data1]
    data2 = [data[key] for data in data2]

    print(key)

    test_ind(data1, data2)


def split_by_df():
    with open('/homes/rqyu/Data/PI-RADS/data.json') as f:
        all_data = json.load(f)

    train_df_case = pd.read_csv('/homes/rqyu/Data/PI-RADS/train_data.csv')['casename'].tolist()
    val_df_case = pd.read_csv('/homes/rqyu/Data/PI-RADS/val_data.csv')['casename'].tolist()
    test_df_case = pd.read_csv('/homes/rqyu/Data/PI-RADS/test_data.csv')['casename'].tolist()

    train_json = [case for case in all_data if case['casename'][:-2] in train_df_case]
    val_json = [case for case in all_data if case['casename'][:-2] in val_df_case]
    test_json = [case for case in all_data if case['casename'][:-2] in test_df_case]

    test_ind_dict(train_json, val_json, 'PI-RADS')
    test_ind_dict(train_json, val_json, 'age')
    test_ind_dict(train_json, val_json, 'D-max')
    test_ind_dict(train_json, val_json, 'psa')
    test_ind_dict(train_json, val_json, 'GS')

    test_ind_dict(train_json+val_json, test_json, 'PI-RADS')
    test_ind_dict(train_json+val_json, test_json, 'age')
    test_ind_dict(train_json+val_json, test_json, 'D-max')
    test_ind_dict(train_json+val_json, test_json, 'psa')
    test_ind_dict(train_json+val_json, test_json, 'GS')

    save_dir = '/homes/rqyu/Data/PI-RADS'
    with open(join_path(save_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=1)
    with open(join_path(save_dir, 'val.json'), 'w') as f:
        json.dump(val_json, f, indent=1)
    with open(join_path(save_dir, 'test.json'), 'w') as f:
        json.dump(test_json, f, indent=1)


if __name__ == '__main__':
    # split()
    split_by_df()
