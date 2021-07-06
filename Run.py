import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from Utils.Encode import encode, decode
from ImageProcess.Analysis import get_biggest_slice
from Visualization.Image import show_multi_images
from Data.Preprocess import join_path
import os
import numpy as np


def run(data_loader, model, criterion, optimizer, mode, loss_recorder, acc_recorder, loc_recorder, config, gradcam=False):
    device = config['DEVICE']

    if mode == 'train':
        model.train()
    elif mode == 'inference':
        model.eval()

    for i, data in enumerate(data_loader):
        if len(data) == 3:
            img, label, loc = data
            gs = None
            dmax = None
        else:
            img, label, gs, dmax, roi, loc = data

        label = label.to(torch.float32).to(device)  # (batch, 1) from 2 to 5
        img = img.to(torch.float32).to(device)  # (batch, 3, *)
        loc = loc.to(torch.float32).to(device)  # (batch, 1)

        prediction = model(img)  # (batch, outnode) or (batch, outnode+1)

        # Optimize
        if config['LOC'] == 'multi task':
            # (batch, out_node+1) before sigmoid
            loc_pred = torch.sigmoid(prediction[:, -1:])  # (batch, 1)
            loc_loss = nn.BCELoss()(loc_pred, loc)
            loss_recorder.record(loc_loss.detach().item(), 'loc')

            loc_recorder.record(loc.cpu().numpy(), 'label')
            loc_recorder.record(loc_pred.detach().cpu().numpy(), 'pred')

            prediction = prediction[:, :-1]
        elif config['LOC'] == 'select':
            # (batch, out_node+1). loc after sigmoid
            loc_pred = prediction[:, -1:]  # (batch, 1)
            loc_loss = nn.BCELoss()(loc_pred, loc)
            loss_recorder.record(loc_loss.detach().item(), 'loc')

            loc_recorder.record(loc.cpu().numpy(), 'label')
            loc_recorder.record(loc_pred.detach().cpu().numpy(), 'pred')

            prediction = prediction[:, :-1]
        else:
            loc_loss = 0

        if config['PROCESS MODE'] == 'UDM':
            loss = criterion(prediction, label)
            prediction_r = criterion.inference(prediction).detach().cpu().numpy() - 2  # (batch, 1)
        elif config['PROCESS MODE'] == 'Cross Entropy':
            # (batch, 4) before softmax
            prediction = torch.softmax(prediction, dim=1)
            loss = criterion(prediction, (label-2).to(torch.long).squeeze(dim=1))
            prediction_r = torch.argmax(prediction, dim=1).unsqueeze(dim=1).cpu().numpy()  # (batch, 1)
        elif config['PROCESS MODE'] == 'Soft Regression':
            # (batch, 1) before sigmoid
            prediction = torch.sigmoid(prediction)
            label_target = (label-2)/3  # (batch, 1)
            loss = criterion(prediction, label_target)

            prediciton_target = torch.tensor([[0,0.33,0.67,1]], device=prediction.device, dtype=prediction.dtype)\
                .repeat([prediction.shape[0],1])  # (batch, 4)
            prediction = prediction.repeat([1, 4])  # (batch, 4)
            prediction_r = torch.argmin(torch.abs(prediction-prediciton_target), dim=1)\
                .unsqueeze(dim=1).cpu().numpy()  # (batch, 1)
        elif config['PROCESS MODE'] == 'Encode':
            # (batch, 3) before sigmoid
            prediction = torch.sigmoid(prediction)
            label_code = encode(label)
            loss = criterion(prediction, label_code)

            prediction = decode(prediction)
            prediction_r = prediction.cpu().numpy()  # (batch, 1)
        else:
            loss = None
        loss_recorder.record(loss.detach().item(), 'classification')
        loss = loss + loc_loss

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if gradcam:
            loss.backward()

        loss_r = loss.detach().item()
        loss_recorder.record(loss_r, 'loss')

        label_r = label.detach().cpu().numpy() - 2
        acc_recorder.record(label_r, 'label')
        acc_recorder.record(prediction_r, 'pred')
        if gs is not None:
            acc_recorder.record(gs, 'gs')
            acc_recorder.record(dmax, 'dmax')
            # acc_recorder.record(loc, 'loc')

        casenames = [x['casename'] for x in data_loader.dataset.datalist]
        if gradcam:
            casename = casenames[i]
            grad_cam = model.gc_saver.print('layer4', img.shape[-3:])

            roi = roi[0, 0]  # (h, w, d)
            slice = get_biggest_slice(roi)

            grad_cam = grad_cam[0, 0, :, :, slice].cpu().detach()  # (h, w)
            img = img[0, :, :, :, slice].cpu().detach()  # (3, h, w)
            roi = roi[:, :, slice].cpu().detach()

            pic_save_dir = join_path(config['SAVE'], config['NAME'], f"test{config['TEST INDEX']}")
            if not os.path.exists(pic_save_dir):
                os.mkdir(pic_save_dir)

            show_multi_images([{'name':'t2', 'img':img[0], 'roi':roi},
                               {'name':'dwi', 'img':img[1], 'roi':roi},
                               {'name':'adc', 'img':img[2], 'roi':roi},
                               {'name':'gradcam', 'img':grad_cam, 'roi':roi}],
                              arrangement=[1,4],
                              title=f'{casename} label:{int(label_r.item()+2)} pirads:{int(prediction_r.item()+2)} '
                                    f'gs:{int(gs.item())}',
                              save_path=join_path(pic_save_dir, f'{casename}.png'))
