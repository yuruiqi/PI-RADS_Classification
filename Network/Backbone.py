import torch
import torch.nn as nn
import torch.functional as F
from functools import partial
from Visualization.Interpretation import GradCamSaver
import matplotlib.pyplot as plt
from Network.Utils import batch_slice


def conv3x3x3(in_planes, out_planes, stride=1, threed=True, group=1):
    if threed:
        return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=group)


def conv1x1x1(in_planes, out_planes, stride=1, threed=True, group=1):
    if threed:
        return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, groups=group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, threed=True, group=1):
        super().__init__()
        if threed:
            bn = nn.BatchNorm3d
        else:
            bn = nn.BatchNorm2d

        self.conv1 = conv3x3x3(in_planes, planes, stride, threed, group=group)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, threed, group=group)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, attention=False, threed=True, group=1):
        super().__init__()

        if threed:
            bn = nn.BatchNorm3d
        else:
            bn = nn.BatchNorm2d

        self.conv1 = conv1x1x1(in_planes, planes, threed=threed, group=group)
        self.bn1 = bn(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, threed=threed, group=group)
        self.bn2 = bn(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion, threed=threed, group=group)
        self.bn3 = bn(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.attention = attention

    def forward(self, x):
        if self.attention:
            x, map = x
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.attention:
            # map = torch.nn.functional.interpolate(map, size=x.shape[2:])
            if len(x.shape)==5:
                map = torch.nn.AdaptiveAvgPool3d(residual.shape[2:])(map)
            else:
                map = torch.nn.AdaptiveAvgPool2d(residual.shape[2:])(map)
            out += residual*map
        else:
            out += residual

        out = self.relu(out)

        if self.attention:
            return [out, map]
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, config, gradcam=False, show_sl=False):
        super().__init__()

        block = Bottleneck

        layers = [3, 4, 6, 3]
        block_inplanes = config['INPLANE']
        n_input_channels = 3
        conv1_t_size = 7
        conv1_t_stride = 1
        no_max_pool = False
        shortcut_type = 'B'


        if len(config['SHAPE']) == 2:
            threed = False
            self.conv = nn.Conv2d
            self.bn = nn.BatchNorm2d
            self.maxpool = nn.MaxPool2d
            dropout = torch.nn.Dropout2d
        else:
            threed = True
            self.conv = nn.Conv3d
            self.bn = nn.BatchNorm3d
            self.maxpool = nn.MaxPool3d
            dropout = torch.nn.Dropout3d
        if config['DROPOUT']:
            self.dropout = True
            self.drop1 = dropout(p=0.1)
            self.drop2 = dropout(p=0.1)
            self.drop3 = dropout(p=0.1)
            self.drop4 = dropout(p=0.1)
        else:
            self.dropout = False

        self.threed = threed

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.group = config['GROUP']
        self.sep_input = config['SEP INPUT']

        self.loc = config['LOC']
        # if self.loc == 'select':
        if self.sep_input:
            self.conv1_t2 = self.conv(1,
                                   self.in_planes//2,
                                   kernel_size=(7, 7, conv1_t_size) if threed else 7,
                                   stride=(2, 2, conv1_t_stride) if threed else 2,
                                   padding=(3, 3, conv1_t_size // 2) if threed else 3,
                                   bias=False,
                                   groups=1)
            self.conv1_dwi_adc = self.conv(2,
                                   self.in_planes//2,
                                   kernel_size=(7, 7, conv1_t_size) if threed else 7,
                                   stride=(2, 2, conv1_t_stride) if threed else 2,
                                   padding=(3, 3, conv1_t_size // 2) if threed else 3,
                                   bias=False,
                                   groups=1)
        else:
            self.conv1 = self.conv(n_input_channels,
                                   self.in_planes,
                                   kernel_size=(7, 7, conv1_t_size) if threed else 7,
                                   stride=(2, 2, conv1_t_stride) if threed else 2,
                                   padding=(3, 3, conv1_t_size // 2) if threed else 3,
                                   bias=False,
                                   groups=self.group)

        self.bn1 = self.bn(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = self.maxpool(kernel_size=3, stride=2, padding=1)

        self.attention = config['ATTENTION']
        self.channel_dict = torch.tensor([-0.3503868, 1.2224333, -0.14608796], device=config['DEVICE'])

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, threed=threed)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2,
                                       threed=threed)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2,
                                       threed=threed)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2,
                                       threed=threed)
        if threed:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # fc layer node
        process_mode = config['PROCESS MODE']
        if process_mode == 'UDM' or process_mode == 'Soft Regression':
            out_node = 1
        elif process_mode == 'Cross Entropy':
            out_node = 4
        elif process_mode == 'Encode':
            out_node = 3
        else:
            raise ValueError

        # fc layer
        if self.loc == 'multi task':
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, out_node+1)
        elif self.loc == 'select':
            self.fc_loc = nn.Linear(block_inplanes[3] * block.expansion, 1)
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, out_node*2)
        else:
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, out_node)

        # init
        # self.init_weight()
        self.pretrain()

        # grad cam
        if gradcam:
            self.gc_saver = GradCamSaver()
            self.layer4.register_forward_hook(self.gc_saver.save_activation('layer4'))
        else:
            self.gc_saver = None

    def forward(self, x):
        if self.attention:
            channel_dict = self.channel_dict.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)\
                .repeat([x.shape[0], 1]+list(x.shape[2:]))  # (batch, 3, *)
            distance = (x - channel_dict).square().sum(dim=1, keepdim=True).sqrt()  # (batch, 1, *)
            map = torch.exp(-distance)
            map = (map-torch.min(map))/(torch.max(map) - torch.min(map))

        # if self.loc == 'select':
        if self.sep_input:
            x = torch.cat([self.conv1_t2(x[:, 0:1]), self.conv1_dwi_adc(x[:, 1:])], dim=1)
        else:
            x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        if self.attention:
            x, map = self.layer1([x, map])
            x, map = self.layer2([x, map])
            x, map = self.layer3([x, map])
            x, map = self.layer4([x, map])
        else:
            x = self.layer1(x)
            if self.dropout:
                x = self.drop1(x)
            x = self.layer2(x)
            if self.dropout:
                x = self.drop2(x)
            x = self.layer3(x)
            if self.dropout:
                x = self.drop3(x)
            x = self.layer4(x)  # (batch, c, *)
            if self.dropout:
                x = self.drop4(x)

        if self.gc_saver is not None:
            x.register_hook(self.gc_saver.save_grad('layer4'))

        x = self.avgpool(x)  # (batch, c, 1,1,1)
        x = x.view(x.size(0), -1)  # (batch, inplane[3])

        if self.loc == 'select':
            loc_pred = torch.sigmoid(self.fc_loc(x))  # (batch, 1)
            x = self.fc(x).reshape([x.shape[0], 2, -1])  # (batch, 2, outnode)
            x = x[:, 0] * (1 - loc_pred) + x[:, 1] * loc_pred  # (batch, outnode)
            x = torch.cat([x, loc_pred], dim=1)  # (batch, out_node+1)
        else:
            x = self.fc(x)
        return x

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, threed=True):
        if threed:
            bn = nn.BatchNorm3d
        else:
            bn = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride, threed=threed),
                    bn(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample,
                  attention=self.attention,
                  threed=threed,
                  group=self.group))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, attention=self.attention, threed=threed, group=self.group))

        return nn.Sequential(*layers)

    def pretrain(self):
        if self.threed:
            state_dict = torch.load('/homes/rqyu/Data/pretrained_models/resnet50_3d.pth')['state_dict']
        else:
            state_dict = torch.load('/homes/rqyu/Data/pretrained_models/resnet50-19c8e357.pth')
        model_dict = self.state_dict()
        # 筛除不加载的层结构
        if self.sep_input:
            for k,v in list(state_dict.items()):
                if k == 'conv1.weight':
                    shape = model_dict['conv1_t2.weight'].shape
                    state_dict['conv1_t2.weight'] = state_dict[k][:shape[0], :shape[1]]
                    shape = model_dict['conv1_dwi_adc.weight'].shape
                    state_dict['conv1_dwi_adc.weight'] = state_dict[k][:shape[0], :shape[1]]

                if k in model_dict:
                    model_v_shape = model_dict[k].shape
                    if model_v_shape != state_dict[k].shape:
                        if len(model_v_shape)>2:
                            state_dict[k] = state_dict[k][:model_v_shape[0], :model_v_shape[1]]
                        else:
                            state_dict[k] = state_dict[k][:model_v_shape[0]]

        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        state_dict['fc.weight'] = state_dict['fc.weight'][:self.fc.weight.shape[0], :self.fc.weight.shape[1]]
        state_dict['fc.bias'] = state_dict['fc.bias'][:self.fc.bias.shape[0]]
        # 更新当前网络的结构字典
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, self.conv):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, self.bn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

