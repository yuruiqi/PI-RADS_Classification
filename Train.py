import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import shutil
import os
import argparse
import sys
import json

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


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    # optimizer.param_groups[0]['lr'] = lr
    # if epoch<100:
    #     optimizer.param_groups[1]['lr'] = 0
    # else:
    #     optimizer.param_groups[1]['lr'] = base_lr * (1-(epoch-100)/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(config):
    device = config['DEVICE']
    if config['PRELOAD'] == 0:
        train_preload = True
        val_preload = True
    elif config['PRELOAD'] == 1:
        train_preload = True
        val_preload = False
    else:
        train_preload = False
        val_preload = False

    model_save_dir = join_path(config['SAVE'], config['NAME'])
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    batch_size = config['BATCH']
    e_bool = config['E BOOL']
    lr = config['LR']
    lr_gamma = config['LR GAMMA']
    patience = config['PATIENCE']
    step_size = config['STEP SIZE']
    n_epoch = config['EPOCH']
    process_mode = config['PROCESS MODE']

    ##########
    # Prepare
    ##########
    # Get train data
    train_dataset = MyDataset(config['TRAIN'], config, preload=train_preload, augment=True)
    train_label = [data['PI-RADS'] for data in train_dataset.datalist]
    weight = get_sampler_weight(train_label)
    sampler = WeightedRandomSampler(weight, num_samples=len(weight))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=5)

    # Get val data
    val_dataset = MyDataset(config['VAL'], config, preload=val_preload)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    evaluator = ResNet(config).to(device)

    if process_mode == 'UDM':
        criterion = UnsureDataLoss(config)
        criterion.to(device)
    elif process_mode == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()
    elif process_mode == 'Soft Regression':
        criterion = nn.BCELoss()
    elif process_mode == 'Encode':
        criterion = nn.BCELoss()

    if config['LOC'] is None:
        optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, evaluator.parameters()), 'lr':lr},
                                     {'params': filter(lambda p: p.requires_grad, criterion.parameters()), 'lr':lr*5}])
        # optimizer = torch.optim.SGD([{'params': evaluator.parameters(), 'lr':lr},
        #                              {'params': criterion.parameters(), 'lr':lr*5}])
    else:
        optimizer = torch.optim.SGD([
            {'params': filter(lambda p: p.requires_grad, evaluator.parameters()), 'lr': lr},
            # {'params': evaluator.fc_loc.parameters(), 'lr': lr},
            {'params': filter(lambda p: p.requires_grad, criterion.parameters()),'lr': lr * 5}])

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)

    # Recorder
    train_loss_recorder = LossRecorder('train', save_dir=model_save_dir)
    val_loss_recorder = LossRecorder('val', patience=patience, save_dir=model_save_dir)
    train_acc_recorder = ClassifierRecorder('train acc', n_classes=4, save_dir=model_save_dir)
    val_acc_recorder = ClassifierRecorder('val acc', n_classes=4, save_dir=model_save_dir)
    if config['LOC']:
        train_loc_recorder = ClassifierRecorder('train loc', n_classes=2, save_dir=model_save_dir)
        val_loc_recorder = ClassifierRecorder('val loc', n_classes=2, save_dir=model_save_dir, patience=20)
    else:
        train_loc_recorder = None
        val_loc_recorder = None

    ##########
    # TrainUtils
    ##########
    best_epoch = 0
    frozen = False
    finish = False
    for epoch in range(n_epoch):
        print('Epoch {}'.format(epoch))

        run(train_loader, evaluator, criterion, optimizer, 'train',
            train_loss_recorder, train_acc_recorder, train_loc_recorder, config)

        # adjust_learning_rate_poly(optimizer, epoch, config['EPOCH'], config['LR'], 0.9)  # 1

        with torch.no_grad():
            run(val_loader, evaluator, criterion, optimizer, 'inference',
                val_loss_recorder, val_acc_recorder, val_loc_recorder, config)

        scheduler.step()  # 2

        # Recorder
        train_loss_recorder.new_epoch()
        if config['LOC']:
            train_loss_recorder.print_result(keys=['loss', 'loc'])
            train_loc_recorder.new_epoch()
            train_loc_recorder.print_result(show_metrix=True, show_class_num=True)
        else:
            train_loss_recorder.print_result(keys=['loss'])
        train_acc_recorder.new_epoch()
        train_acc_recorder.print_result(show_metrix=True, show_class_num=True)

        val_loss_recorder.new_epoch()
        if config['LOC']:
            val_loss_recorder.print_result(keys=['loss', 'loc'])
            val_loc_recorder.new_epoch()
            val_loc_recorder.print_result(show_metrix=True, show_class_num=True)
        else:
            val_loss_recorder.print_result(keys=['loss'])
        val_acc_recorder.new_epoch()
        val_acc_recorder.print_result(show_metrix=True, show_class_num=True)

        # freeze layer
        # if config['LOC'] == 'select':
        #     if frozen:
        #         save, finish = val_loss_recorder.judge(key='loss')
        #     else:
        #         save, freeze = val_loc_recorder.judge(key='acc', lower=False)
        #         if freeze:
        #             evaluator.load_state_dict(torch.load(join_path(model_save_dir, 'Resnet.pkl')))
        #             criterion.load_state_dict(torch.load(join_path(model_save_dir, 'Udm.pkl')))
        #             for name, param in evaluator.named_parameters():
        #                 if name in ['fc.weight', 'fc.bias', 'fc_t2.weight', 'fc_t2.bias',
        #                             'fc_dwi_adc.weight', 'fc_dwi_adc.bias']:
        #                     param.requires_grad = True
        #                 else:
        #                     param.requires_grad = False
        #             optimizer = torch.optim.SGD([
        #                 {'params': filter(lambda p: p.requires_grad, evaluator.parameters()), 'lr': 0.001},
        #                 {'params': filter(lambda p: p.requires_grad, criterion.parameters()), 'lr': 0.005}])
        #             frozen = True
        # else:
        #     # save
        #     save, finish = val_loss_recorder.judge(key='loss')
        #     # save, finish = val_acc_recorder.judge(key='acc')

        # save, finish = val_loss_recorder.judge(key='loss')
        save, finish = val_loss_recorder.judge(key='classification')

        if save:
            print('Saving.')
            best_epoch = epoch
            torch.save(evaluator.state_dict(), join_path(model_save_dir, 'Resnet.pkl'))

            if process_mode == 'UDM':
                torch.save(criterion.state_dict(), join_path(model_save_dir, 'Udm.pkl'))
                t = [criterion.t23.item(), criterion.t34.item(), criterion.t45.item()]
                if e_bool:
                    e = [criterion.e23.item(), criterion.e34.item(), criterion.e45.item()]
        if finish:
            break

    print('******** Best Epoch ********')
    # best_epoch = val_acc_recorder.print_best(show_metrix=True)
    train_acc_recorder.print_result(best_epoch, show_metrix=True)
    val_acc_recorder.print_result(best_epoch, show_metrix=True)
    print('Best Epoch: {}'.format(best_epoch))

    # best_epoch = val_loss_recorder.print_best()
    # train_acc_recorder.print_result(best_epoch, show_metrix=True)
    # val_acc_recorder.print_result(best_epoch, show_metrix=True)
    # print('Best Epoch: {}'.format(best_epoch))

    if process_mode == 'UDM':
        print('t:', t)
        if e_bool:
            print('e:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config key')
    parser.add_argument('--preload', type=int, default=0, help='0:all, 1:train, 2:none')
    parser.add_argument('--device', type=int, default=2, help='cuda id')
    parser.add_argument('--batch', type=int, default=32, help='batch size')

    debug = False
    debug_key = 'udm_sep1'

    config = configs['Base'].copy()
    if debug:
        config_key = debug_key
        config.update(configs[config_key])

        config['PRELOAD'] = 2
        config['DEVICE'] = 2
        config['BATCH'] = 32
    else:
        config_key = parser.parse_args().config
        config.update(configs[config_key])

        config['PRELOAD'] = parser.parse_args().preload
        config['DEVICE'] = torch.device(f'cuda:{parser.parse_args().device}')
        config['BATCH'] = parser.parse_args().batch

    print(config_key)
    # set_seed()

    train(config)
