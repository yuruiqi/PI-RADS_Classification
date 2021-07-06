import os
import json
import pandas as pd
from Data.Preprocess import join_path
import numpy as np


# 1.
def get_1_roi_feature(case_dir, n):
    roi_path = join_path(case_dir, f'roi{n}_info.csv')

    if os.path.exists(roi_path):
        df = pd.read_csv(roi_path, header=None)

        if df.shape[0] > df.shape[1]:
            df = pd.DataFrame(df.values.T[1:], columns=df[0])
        else:
            df = pd.DataFrame(df.values[1:], columns=df.iloc[0])

        if 'GS' in df.columns:
            gs = float(df['GS'].item())
            if np.isnan(gs):
                gs = 0
            else:
                gs = int(round(gs))
        else:
            gs = 0

        psa = df['psa'].item()
        if '>' in psa:
            psa = float(psa[1:])
        else:
            psa = float(psa)

        loc = int(df['loc'].item())
        loc_df = pd.read_csv(r'/homes/rqyu/Data/PI-RADS/loc.csv')
        # TODO
        if loc == 2:
            casename = case_dir.replace('/homes/rqyu/Data/PI-RADS/original_data/', '')
            loc = loc_df[loc_df['casename']==casename][loc_df['roi']==n]['zone'].item()
            if loc == 'pz':
                loc = 1
            elif loc == 'tz':
                loc = 0
            else:
                raise ValueError

        dict = {'t2': join_path(case_dir, 't2_Resize.nii'),
                'dwi': join_path(case_dir, 'dwi_Reg_Resize.nii'),
                'adc': join_path(case_dir, 'adc_Reg_Resize.nii'),
                'roi': join_path(case_dir, f'roi{n}_Resize.nii'),
                'roi_info': roi_path,
                'PI-RADS': int(df['PI-RADS'].item()), 'age': int(df['age'].item()),
                'D-max': float(df['D-max'].item()), 'psa': float(psa),
                'GS': gs,
                'loc':loc}
    else:
        dict = None
    return dict


def get_feature_json_case(case_dir):
    data = []
    for i in range(3):
        roi_data = get_1_roi_feature(case_dir, i)
        if roi_data is not None:
            data.append(roi_data)

    with open(join_path(case_dir, 'feature.json'), 'w') as file:
        json.dump(data, file, indent=1)
    return data


def get_feature_json(data_dir, save_path):
    dir = data_dir
    data = []
    for case in os.listdir(dir):
        # print(case)
        case_dir = join_path(dir, case)
        case_datas = get_feature_json_case(case_dir)
        for i, case_data in enumerate(case_datas):
            case_data['casename'] = f'{case}_{i}'
            data.append(case_data)

            if case_data['loc'] >1:
                print(case)
                raise ValueError

    with open(save_path, 'w') as file:
        json.dump(data, file, indent=1)


if __name__ == '__main__':
    get_feature_json(r'/homes/rqyu/Data/PI-RADS/original_data', join_path(r'/homes/rqyu/Data/PI-RADS', 'data.json'))
