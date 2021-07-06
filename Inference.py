import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import sys
import json
from Analysis.Measurement import classification_stat
import numpy as np
import pandas as pd

sys.path.append(r'/homes/rqyu/PycharmProjects/MyUtils')

from Utils.Dataset import MyDataset
from Config import configs
from Network.UnsureDataLoss import UnsureDataLoss
from Run import run

from TrainUtils.Recorder import LossRecorder, ClassifierRecorder
from Network.Utils import load_state_dict
from Data.Preprocess import join_path
# from Network.ResNet3d import generate_model
from Network.Backbone import ResNet
from TrainUtils.Sampler import get_sampler_weight
from monai.visualize.class_activation_maps import GradCAM


def inference(config, grad_cam=False):
    device = config['DEVICE']

    model_save_dir = join_path(config['SAVE'], config['NAME'])
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    batch_size = config['BATCH']
    patience = config['PATIENCE']
    process_mode = config['PROCESS MODE']

    ##########
    # Prepare
    ##########
    if config['TEST INDEX'] == 1:
        test_path = config['TEST']
    else:
        test_path = config['TEST2']

    test_dataset = MyDataset(test_path, config, preload=False, detail=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    evaluator = ResNet(config, grad_cam).to(device)
    evaluator.load_state_dict(torch.load(join_path(model_save_dir, 'Resnet.pkl')))

    if process_mode == 'UDM':
        criterion = UnsureDataLoss(config)
        criterion.load_state_dict(torch.load(join_path(model_save_dir, 'Udm.pkl')))
        criterion.to(device)
    elif process_mode == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()
    elif process_mode == 'Soft Regression':
        criterion = nn.BCELoss()
    elif process_mode == 'Encode':
        criterion = nn.BCELoss()

    # Recorder
    test_loss_recorder = LossRecorder('test', patience=patience, save_dir=model_save_dir)
    test_acc_recorder = ClassifierRecorder('test acc', n_classes=4, save_dir=model_save_dir)
    if config['LOC']:
        test_loc_recorder = ClassifierRecorder('test loc', n_classes=2, save_dir=model_save_dir)
    else:
        test_loc_recorder = None

    if config['LOC'] == 'select':
        print('tz, t2',evaluator.fc.weight[0,:256].abs().mean().item(),
              'tz, dwi adc',evaluator.fc.weight[0,256:].abs().mean().item(),
              'pz, t2',evaluator.fc.weight[1,:256].abs().mean().item(),
              'pz, dwi adc',evaluator.fc.weight[1,256:].abs().mean().item())

    if grad_cam:
        run(test_loader, evaluator, criterion, None, 'inference',
            test_loss_recorder, test_acc_recorder, test_loc_recorder, config, grad_cam)
    else:
        with torch.no_grad():
            run(test_loader, evaluator, criterion, None, 'inference',
                test_loss_recorder, test_acc_recorder, test_loc_recorder, config)

    # Recorder
    test_loss_recorder.new_epoch()
    if config['LOC']:
        test_loss_recorder.print_result(keys=['loss', 'loc'])
        test_loc_recorder.new_epoch()
        test_loc_recorder.print_result()
    else:
        test_loss_recorder.print_result(keys=['loss'])
    test_acc_recorder.new_epoch()
    test_acc_recorder.print_result(show_metrix=True, show_class_num=True)

    pred = np.concatenate(test_acc_recorder.data_all[0]['pred']) + 2
    radiologist = np.concatenate(test_acc_recorder.data_all[0]['label']) + 2
    gs = np.concatenate(test_acc_recorder.data_all[0]['gs'])
    dmax = np.concatenate(test_acc_recorder.data_all[0]['dmax'])

    if config['LOC']:
        loc_pred = np.concatenate(test_loc_recorder.data_all[0]['pred'])
        loc = np.concatenate(test_loc_recorder.data_all[0]['label'])

    classification_stat(gs, radiologist, pred, dmax)

    casenames = [x['casename'] for x in test_dataset.datalist]
    df = pd.DataFrame({'casename':casenames, 'radiologist':radiologist, 'pred':pred, 'gs':gs})
    df.to_csv(join_path(model_save_dir, 'test result.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=0, help='config key')
    parser.add_argument('--preload', type=int, default=0, help='0:all, 1:train, 2:none')
    parser.add_argument('--device', type=int, default=2, help='cuda id')
    parser.add_argument('--batch', type=int, default=32, help='batch size')

    # config_key = parser.parse_args().config
    # config = configs['Base'].copy()
    # config.update(configs[config_key])
    #
    # config['PRELOAD'] = parser.parse_args().preload
    # config['DEVICE'] = torch.device(f'cuda:{parser.parse_args().device}')
    # config['BATCH'] = parser.parse_args().batch

    config_key = 'udm_loc_sl_2d'
    config = configs['Base'].copy()
    config.update(configs[config_key])
    config['PRELOAD'] = 2
    config['DEVICE'] = 2

    print(config_key)
    grad_cam = False
    test2 = True

    if test2:
        config['TEST INDEX'] = 2

    if grad_cam:
        config['BATCH'] = 1
        inference(config, grad_cam=True)
    else:
        config['BATCH'] = 32
        inference(config, grad_cam=False)
