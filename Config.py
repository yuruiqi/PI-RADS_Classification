import torch
import os
from TrainUtils.Coder import Coder
import sys


configs = {
    'Base': {
        'TRAIN': '/homes/rqyu/Data/PI-RADS/train.json',
        'VAL': '/homes/rqyu/Data/PI-RADS/val.json',
        'TEST': '/homes/rqyu/Data/PI-RADS/test.json',
        'TEST2': '/homes/rqyu/Data/PI-RADS/SUH/data.json',
        'TEST INDEX': 1,

        'DROP CASE': '/homes/rqyu/Data/PI-RADS/dropcases.txt',

        'SAVE': r'/homes/rqyu/Projects/PI-RADS_Classification/Results',

        # Train
        'EPOCH': 1000,
        'PATIENCE': 100,
        'BATCH': 32,
        'OPTIM': 'sgd',
        'LR': 0.001,
        'LR GAMMA': 0.99,
        'STEP SIZE': 50,
        'DROPOUT': False,

        # Model
        'SHAPE': [128, 128, 10],
        'PROCESS MODE': 'UDM',
        'E BOOL': False,
        'INPLANE': [64, 128, 256, 512],
        'GROUP': 1,
        'ATTENTION': False,
        'LOC':None,
        'SEP INPUT': False,
    },

    'ce': {
        'NAME': 'CE',
        'PROCESS MODE': 'Cross Entropy',
    },

    'udm': {
        'NAME': 'UDM',
    },

    'sr': {
        'NAME': 'Soft Regression',
        'PROCESS MODE': 'Soft Regression',
        },

    'encode': {
        'NAME': 'Encode',
        'PROCESS MODE': 'Encode',
        },

    'udm_attention': {
        'NAME': 'UDM ATTENTION',
        'ATTENTION': True,
    },

    'ce_2d': {
        'NAME': 'CE 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Cross Entropy',
    },

    'udm_2d': {
        'NAME': 'UDM 2d',
        'SHAPE': [128, 128],
    },

    'sr_2d': {
        'NAME': 'Soft Regression 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Soft Regression',
    },

    'encode_2d': {
        'NAME': 'Encode 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Encode',
    },

    'udm_loc_mc': {
        'NAME': 'UDM loc multi-class',
        'LOC':'multi class'
    },

    'udm_loc_sl': {
        'NAME': 'UDM loc select',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT':True
    },

    'ce_loc_sl': {
        'NAME': 'CE loc select',
        'PROCESS MODE': 'Cross Entropy',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT':True
    },

    'sr_loc_sl': {
        'NAME': 'SR loc select',
        'PROCESS MODE': 'Soft Regression',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT':True
    },

    'encode_loc_sl': {
        'NAME': 'Encode loc select',
        'PROCESS MODE': 'Encode',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT':True
    },


    'udm_loc_sl_2d': {
        'NAME': 'UDM loc select 2d',
        'SHAPE': [128, 128],
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT': True
    },

    'ce_loc_sl_2d': {
        'NAME': 'CE loc select 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Cross Entropy',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT': True
    },

    'sr_loc_sl_2d': {
        'NAME': 'SR loc select 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Soft Regression',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT': True
    },

    'encode_loc_sl_2d': {
        'NAME': 'Encode loc select 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'Encode',
        'SEP INPUT': True,
        'LOC': 'select',
        'GROUP': 2,
        'DROPOUT': True
    },


    'udm_loc_mt': {
        'NAME': 'UDM loc mt',
        'PROCESS MODE': 'UDM',
        'LOC': 'multi task',
        'DROPOUT': True
    },

    'udm_loc_mt_2d': {
        'NAME': 'UDM loc mt 2d',
        'SHAPE': [128, 128],
        'PROCESS MODE': 'UDM',
        'LOC': 'multi task',
        'DROPOUT': True
    },

    'ce_loc_mt': {
        'NAME': 'CE loc mt',
        'PROCESS MODE': 'Cross Entropy',
        'LOC': 'multi task',
        'DROPOUT': True
    },

    'udm_nosep_sl': {
        'NAME': 'UDM nosep select',
        'LOC': 'select',
        'DROPOUT': True
    },

    'udm_nosep_sl1': {
        'NAME': 'UDM nosep select1',
        'LOC': 'select',
        'DROPOUT': True
    },

    'udm_nosep_sl2': {
        'NAME': 'UDM nosep select2',
        'LOC': 'select',
        'DROPOUT': True
    },

    'udm_nosep_sl3': {
        'NAME': 'UDM nosep select3',
        'LOC': 'select',
        'DROPOUT': True
    },

    'udm_nosep_sl_2d': {
        'NAME': 'UDM nosep select 2d',
        'SHAPE': [128, 128],
        'LOC': 'select',
        'DROPOUT': True
    },

    'udm_sep1': {
        'NAME': 'UDM sep1',
        'SEP INPUT': True,
        'DROPOUT': True
    },

    'udm_sep2': {
        'NAME': 'UDM sep2',
        'SEP INPUT': True,
        'DROPOUT': True
    },

    'udm_sep3': {
        'NAME': 'UDM sep3',
        'SEP INPUT': True,
        'DROPOUT': True
    },

    'udm_sep4': {
        'NAME': 'UDM sep4',
        'SEP INPUT': True,
        'DROPOUT': True
    },

}

