import numpy as np
from monai.transforms import *
import json
import matplotlib.pyplot as plt
import pandas as pd

from ImageProcess.Operations import get_box
from Data.Preprocess import join_path
from Config import drop_case
from Statistics.IndependenceTest import test_ind


def intensity_stats(datalist):
    t2, dwi, adc = [], [], []
    transforms = [
        AddChanneld(keys=['t2', 'dwi', 'adc', 'roi', 'PI-RADS']),
        Spacingd(keys=['t2', 'dwi', 'adc', 'roi'], pixdim=[0.5, 0.5, 3.3], mode=['bilinear', 'bilinear', 'bilinear', 'nearest']),
        CropForegroundd(keys=['t2', 'dwi', 'adc', 'roi'], source_key='roi', margin=[10, 10, 2]),
        ConcatItemsd(keys=['t2', 'dwi', 'adc'], name='image'),
        Resized(keys=['image', 'roi'], spatial_size=[128, 128, 10]),
        NormalizeIntensityd(keys='image', channel_wise=True),
    ]
    for dict in datalist:
        data = LoadImaged(keys=['t2', 'dwi', 'adc', 'roi'])(dict)
        data['PI-RADS'] = np.array(data['PI-RADS'])

        for transform in transforms:
            data = transform(data)

        fg_index = np.where(data['roi'][0] > 0)
        t2_int = np.average([data['image'][0][fg_index[0][i], fg_index[1][i], fg_index[2][i]] for i in range(len(fg_index[0]))])
        dwi_int = np.average([data['image'][1][fg_index[0][i], fg_index[1][i], fg_index[2][i]] for i in range(len(fg_index[0]))])
        adc_int = np.average([data['image'][2][fg_index[0][i], fg_index[1][i], fg_index[2][i]] for i in range(len(fg_index[0]))])
        # t2.extend(t2_int)
        # dwi.extend(dwi_int)
        # adc.extend(adc_int)
        t2.append(t2_int)
        dwi.append(dwi_int)
        adc.append(adc_int)

    plt.hist(t2)
    plt.show()
    plt.hist(dwi)
    plt.show()
    plt.hist(adc)
    plt.show()
    print(np.median(t2), np.median(dwi), np.median(adc))


def reso_stats(datalist):
    reso0, reso1, reso2 = [], [], []
    for dict in datalist:
        data = LoadImaged(keys=['t2'])(dict)

        reso0.append(data['t2_meta_dict']['pixdim'][1])
        reso1.append(data['t2_meta_dict']['pixdim'][2])
        reso2.append(data['t2_meta_dict']['pixdim'][3])

    print(np.median(reso0), np.median(reso1), np.median(reso2))
    plt.hist(reso0)
    plt.show()
    plt.hist(reso1)
    plt.show()
    plt.hist(reso2)
    plt.show()


def label_stats(datalist):
    pirads = []
    loc = []
    for dict in datalist:
        pirads.append(dict['PI-RADS'])
        loc.append(dict['loc'])

    print(plt.hist(pirads))
    plt.show()
    print(plt.hist(loc))


def data_stats(data1, data2):
    for feature in ['PI-RADS', 'age', 'D-max', 'psa', 'GS', 'loc']:
        feature_data1 = [x[feature] for x in data1]
        feature_data2 = [x[feature] for x in data2]
        print(feature)
        test_ind(feature_data1, feature_data2)

        if feature == 'loc':
            test_ind(feature_data1, feature_data2, category=True)


if __name__ == '__main__':
    data_path = '/homes/rqyu/Data/PI-RADS/data.json'
    with open(data_path) as f:
        datalist = json.load(f)
    data_path = '/homes/rqyu/Data/PI-RADS/train.json'
    with open(data_path) as f:
        train_datalist = json.load(f)
    data_path = '/homes/rqyu/Data/PI-RADS/val.json'
    with open(data_path) as f:
        val_datalist = json.load(f)
    data_path = '/homes/rqyu/Data/PI-RADS/test.json'
    with open(data_path) as f:
        test_datalist = json.load(f)
    data_path = '/homes/rqyu/Data/PI-RADS/SUH/data.json'
    with open(data_path) as f:
        test2_datalist = json.load(f)

    for data in datalist:
        if data['casename'][:-2] in drop_case:
            datalist.remove(data)
    for data in train_datalist:
        if data['casename'][:-2] in drop_case:
            train_datalist.remove(data)
    for data in val_datalist:
        if data['casename'][:-2] in drop_case:
            val_datalist.remove(data)
    for data in test_datalist:
        if data['casename'][:-2] in drop_case:
            test_datalist.remove(data)

    # 1.
    # reso_stats(datalist)  # (0.5 0.5 3.3)
    intensity_stats(datalist)  # (-0.3503868 1.2224333 -0.14608796) (-0.26183647 1.3207121 -0.08422626)

    # label_stats(datalist)
    # (125,130,264,253)  772
    # (80,85,165,162)  492
    # (19,20,44,41)  124
    # (26,25,55,50)  156

    # data_stats(train_datalist+val_datalist, test_datalist)

    # 2.
    # reso_stats(test2_datalist)  # (0.52 0.52 3.6)
    # intensity_stats(test2_datalist)  # (-0.29335707 1.1033428 -0.64818645)
