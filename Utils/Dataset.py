import os
import math
from tqdm import tqdm
import random
import numpy as np
import json

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from monai.transforms import *
import pandas as pd
import matplotlib.pyplot as plt

from Data.Preprocess import join_path, get_filename_from_dir
from ImageProcess.Operations import seg_to_mask, get_box
from ImageProcess.Analysis import get_biggest_slice
from Data.Preprocess import join_path
from Visualization.Image import show_multi_images


def judge_roi(feature_json, i):
    return int(feature_json[f'roi{i}']['PI-RADS']) != 0


def get_data(dict, transforms, detail=False):
    """
    data:{'image': (3, *), 'roi': (3, *), 'label': (3, )}

    output:
        data:{'image': (3, *), 'roi': (*), 'label': (3, )}
    """
    data = LoadImaged(keys=['t2', 'dwi', 'adc', 'roi'])(dict)
    data['PI-RADS'] = np.array(data['PI-RADS'])
    data['GS'] = np.array(data['GS'])
    data['D-max'] = np.array(data['D-max'])
    data['loc'] = np.array(data['loc'])

    for transform in transforms:
        data = transform(data)

    del data['t2']
    del data['dwi']
    del data['adc']
    if not detail:
        del data['roi']

    return data

def read_dropcase(path):
    f = open(path)
    line = f.readline()
    drop_cases = []
    while line:
        drop_cases.append(line)
        line = f.readline().replace('\n','')
    f.close()
    return drop_cases


class RandContrastd:
    def __init__(self, keys, factors, prob):
        self.scale = RandScaleIntensityd(keys=keys, factors=factors, prob=prob)

    def __call__(self, data):
        v_min = np.min(data['image'])
        v_max = np.max(data['image'])
        data = self.scale(data)
        data['image'] = np.clip(data['image'], v_min, v_max)
        return data


class RandSimulationd:
    def __init__(self, keys, factors, prob):
        self.factors = factors
        self.prob = prob
        self.keys = keys

    def __call__(self, data):
        factor = random.uniform(self.factors[0], self.factors[1])

        shape = data[self.keys[0]].shape[1:]
        if random.uniform(0,1)<self.prob:
            down_shape = [int(round(x*factor)) for x in shape]
            data = Resized(keys=self.keys, spatial_size=down_shape, mode='nearest')(data)
            mode = 'trilinear' if len(data['image'].shape) == 4 else 'bilinear'
            data = Resized(keys=self.keys, spatial_size=shape, mode=mode, align_corners=True)(data)
            return data
        else:
            return data


class SelectSlice:
    def __init__(self, keys, roi_key):
        self.keys = keys
        self.roi_key = roi_key

    def __call__(self, data):
        slice = get_biggest_slice(data[self.roi_key], axis=2)
        for key in self.keys:
            data[key] = data[key][...,slice]
        data[self.roi_key] = data[self.roi_key][...,slice]
        return data


class MyDataset(Dataset):
    def __init__(self, data_path, config, preload=False, augment=False, detail=False):
        self.shape = config['SHAPE']
        if len(self.shape) == 3:
            threed = True
        else:
            threed = False
        self.augment = augment
        self.detail = detail

        # 1. get data
        pixdim = [0.5, 0.5, 3.3] if threed else [0.5, 0.5]

        if threed:
            self.transforms = [
                AddChanneld(keys=['t2', 'dwi', 'adc', 'roi', 'PI-RADS', 'loc']),
                Spacingd(keys=['t2', 'dwi', 'adc', 'roi'], pixdim=pixdim,
                         mode=['bilinear', 'bilinear', 'bilinear', 'nearest']),
                CropForegroundd(keys=['t2', 'dwi', 'adc', 'roi'], source_key='roi', margin=[10, 10, 2]),
                ConcatItemsd(keys=['t2', 'dwi', 'adc'], name='image'),
                # ResizeWithPadOrCropd(keys=['t2', 'dwi', 'adc', 'roi'], spatial_size=self.shape),
                Resized(keys=['image', 'roi'], spatial_size=self.shape, mode=['trilinear', 'nearest'],
                        align_corners=[True, None]),
                NormalizeIntensityd(keys='image', channel_wise=True),
            ]
        else:
            self.transforms = [
                SelectSlice(keys=['t2', 'dwi', 'adc'], roi_key='roi'),
                AddChanneld(keys=['t2', 'dwi', 'adc', 'roi','PI-RADS', 'loc']),
                Spacingd(keys=['t2', 'dwi', 'adc', 'roi'], pixdim=pixdim, mode=['bilinear','bilinear','bilinear', 'nearest']),
                CropForegroundd(keys=['t2', 'dwi', 'adc', 'roi'], source_key='roi', margin=[10, 10]),
                ConcatItemsd(keys=['t2', 'dwi', 'adc'], name='image'),
                # ResizeWithPadOrCropd(keys=['t2', 'dwi', 'adc', 'roi'], spatial_size=self.shape),
                Resized(keys=['image', 'roi'], spatial_size=self.shape, mode=['bilinear', 'nearest']),
                NormalizeIntensityd(keys='image', channel_wise=True),
            ]

        with open(data_path, 'r') as file:
            self.datalist = json.load(file)

        if config['DROP CASE'] is not None:
            drop_cases = read_dropcase(config['DROP CASE'])
            for data in self.datalist:
                if data['casename'][:-2] in drop_cases:
                    self.datalist.remove(data)

        if preload:
            self.datas = [get_data(d, self.transforms, self.detail) for d in tqdm(self.datalist)]
        else:
            self.datas = None

        # 2. augment
        self.deformation_aug = [
            RandRotated(keys=['image'], range_x=0.0 if len(self.shape) == 3 else 30,
                        range_z=15 if len(self.shape) == 3 else 0.0, prob=0.5),
            RandFlipd(keys=['image'], spatial_axis=2 if len(self.shape)==3 else None, prob=0.5),
        ]

        self.scale_augments = [
            RandGaussianNoised(keys=['image'], mean=0.1, prob=0.5),
            RandGaussianSmoothd(keys=['image'], sigma_x=(0.5, 1.5), prob=0.5),  # blur
            # RandScaleIntensityd(keys=['image'], factors=(-0.3, 0.3), prob=0.15),  # brightness
            # RandContrastd(keys=['image'], factors=(-0.35, 0.5), prob=0.15),  # contrast
            RandSimulationd(keys=['image'], factors=(0.5, 1), prob=0.5),  # simulation
            # RandAdjustContrastd(keys=['image'], gamma=(0.7, 1.5), prob=0.15)  # gamma
        ]

    def __getitem__(self, index):
        if self.datas is None:
            data = self.datalist[index]
            data = get_data(data, self.transforms, self.detail)
        else:
            data = self.datas[index]

        if self.augment:
            for augment in self.deformation_aug:
                data = augment(data)
            for augment in self.scale_augments:
                data = augment(data)

        img = data['image'].astype(np.float32)
        label = data['PI-RADS'].astype(np.float32)
        loc = data['loc'].astype(np.float32)
        if self.detail:
            gs = data['GS'].astype(np.float32)
            dmax = data['D-max'].astype(np.float32)
            roi = data['roi'].astype(np.float32)
            return img, label, gs, dmax, roi, loc
        else:
            return img, label, loc

    def __len__(self):
        return len(self.datalist)


if __name__ == '__main__':
    from Config import configs

    config = configs['Base']
    config['SHAPE'] = [128, 128, 10]

    # data_path = '/homes/rqyu/Data/PI-RADS/data.json'
    data_path = '/homes/rqyu/Data/PI-RADS/SUH/data.json'

    dataset = MyDataset(data_path, config, augment=False, preload=False, detail=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i, (img, label, gs, dmax, roi, loc) in enumerate(dataloader):
        # print(img.shape)

        label = label[0]
        gs = gs[0]
        dmax = dmax[0]
        loc = loc[0]
        roi = roi[0,0]
        slice = get_biggest_slice(roi)

        img = img[0,:,:,:,slice].transpose(0,1).transpose(1,2)  # (h,w,c)
        img = (img-img.min())/(img.max()-img.min())
        roi = roi[:,:,slice].transpose(0,1)

        img = img.transpose(0,1)  # (h,w,c)
        casename = dataset.datalist[i]["casename"]

        # channel_dict = np.array([-0.26183647, 1.3207121, -0.08422626])
        # distance = (img - channel_dict).square().sum(axis=2).sqrt()  # (h,w)
        # map = torch.exp(-distance)
        # map = (map-torch.min(map))/(torch.max(map) - torch.min(map))
        # print(map.min(), map.max())

        plt.figure()
        show_multi_images([{'name':'t2', 'img':img[:,:,0], 'roi':roi},
                           {'name':'dwi', 'img':img[:,:,1], 'roi':roi},
                           {'name':'adc', 'img':img[:,:,2], 'roi':roi},
                           {'name':'all', 'img':img, 'cmap':None, 'roi':roi},
                           # {'name':'attention', 'img':map}
                           ],
                          arrangement=[1,4],
                          title=f'{casename} {label} {gs} {dmax} {loc}',
                          # save_path=join_path(r'/homes/rqyu/Projects/PI-RADS_Classification/dataset view', f'{casename}.png'),
                          save_path=join_path(r'/homes/rqyu/Projects/PI-RADS_Classification/dataset view SUH', f'{casename}.png'),
                          )
