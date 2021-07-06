import os
import json
import pandas as pd
from Data.Preprocess import join_path, get_filename_from_dir
import numpy as np


def get_feature_json_case(dir, case, df):
    case_dir = join_path(dir, case)
    case_df = df[df['PINYIN']==case]

    psa = case_df['PSA'].item()
    if psa == psa:
        if case == 'LIN XIAO MAO':
            psa = 14.50
        elif case == 'DONG BAO SHENG':
            psa = 5.16
        elif psa.isdigit():
            psa = float(psa)
        else:
            psa = float(psa[1:])
    else:
        print(f'{case} no psa')
        return None

    gs = int(case_df['ISUP(All)'].item())

    loc = case_df['position'].item()
    if '弥漫' in loc:
        loc = 1
    elif 'PZ' in loc:
        loc = 1
    else:
        loc = 0

    roi_path = get_filename_from_dir(case_dir, ['ROI', 'CK', 'nii'], return_path=True)[0]

    data = {'casename':case,
            't2': join_path(case_dir, 't2.nii'),
            'dwi': join_path(case_dir, 'dwi_Reg.nii'),
            'adc': join_path(case_dir, 'adc_Reg.nii'),
            'roi': roi_path,
            'PI-RADS': int(case_df['PI-RADS （expert）'].item()), 'age': int(case_df['age'].item()),
            'D-max': float(case_df['D-max'].item()), 'psa': float(psa),
            'GS': gs,
            'loc': loc}

    with open(join_path(case_dir, 'feature.json'), 'w') as file:
        json.dump(data, file, indent=1)

    if data['PI-RADS'] == 1:
        return None
    else:
        return data


def get_feature_json(data_dir, save_path):
    dir = data_dir
    data = []

    df = pd.read_csv(f'/homes/rqyu/Data/PI-RADS/SUH/SUH_PIRADS.csv')

    for case in os.listdir(dir):
        # print(case)
        case_data = get_feature_json_case(dir, case, df)
        if case_data is not None:
            data.append(case_data)

    with open(save_path, 'w') as file:
        json.dump(data, file, indent=1)


if __name__ == '__main__':
    get_feature_json(r'/homes/rqyu/Data/PI-RADS/SUH/data', r'/homes/rqyu/Data/PI-RADS/SUH/data.json')
