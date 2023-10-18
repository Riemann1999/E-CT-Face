import torch
import torch.nn as nn
import torch.nn.functional as F
from model.net import FEC as FEC
from model.net import Neck as Neck
from model.net import MFA as MFA
import torchvision.models._utils as _utils


class ClassHead(nn.Module):
    def __init__(self, input_channels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(input_channels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, input_channels=512, num_anchors=3):
        super(BboxHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(input_channels, self.num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3,
                          1).contiguous()
        return out.view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, input_channels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(input_channels, self.num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0)

    def forward(self, x):
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)


class ect(nn.Module):
    def __init__(self, cfg=None, phase='train', label='det'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(ect, self).__init__()
        self.phase = phase
        self.label = label
        if cfg['name'] == 'Hyb_no_stride':
            from mobilevit.Backbone_ARCHI import mobile_vit_small as create_model
            backbone = create_model()
            if cfg['pretrain']:
                log_dir1 = './model/Pretrain_weight.pth'
                checkpoint = torch.load(log_dir1)
                backbone.load_state_dict(checkpoint)
            self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        """
        B C 1/8W  1/8H
        B C 1/16W 1/16H
        B C 1/32W 1/32H 
        """
        if self.label == 'det':
            in_channels_list = cfg['in_channel_list']
            out_channels = cfg['out_channel']
            trans_channel_list = cfg['Trans_channel_list']

            self.fec = FEC(in_channels_list, trans_channel_list)
            self.neck = Neck(in_channels_list, out_channels)

            self.MFA1 = MFA(out_channels, out_channels)
            self.MFA2 = MFA(out_channels, out_channels)
            self.MFA3 = MFA(out_channels, out_channels)
            self.MFA4 = MFA(out_channels, out_channels)

            self.ClassHead = self._make_class_head(neck_num=4, input_channels=cfg['out_channel'])
            self.BboxHead = self._make_bbox_head(neck_num=4, input_channels=cfg['out_channel'])
            self.LandmarkHead = self._make_landmark_head(neck_num=4, input_channels=cfg['out_channel'])
        elif self.label == 'cls':
            """
            body's output:
            384
            288
            192
            96
            """
            exp_channels = 384
            self.classifier = nn.Sequential()
            self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
            self.classifier.add_module(name="flatten", module=nn.Flatten())
            if 0.0 < 0.1 < 1.0:
                self.classifier.add_module(name="dropout", module=nn.Dropout(p=0.1))
            self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=1000))

    @staticmethod
    def _make_class_head(neck_num=3, input_channels=64, anchor_num=2):
        class_head = nn.ModuleList()
        for i in range(neck_num):
            class_head.append(ClassHead(input_channels, anchor_num))
        return class_head

    @staticmethod
    def _make_bbox_head(neck_num=3, input_channels=64, anchor_num=2):
        bbox_head = nn.ModuleList()
        for i in range(neck_num):
            bbox_head.append(BboxHead(input_channels, anchor_num))
        return bbox_head

    @staticmethod
    def _make_landmark_head(neck_num=3, input_channels=64, anchor_num=2):
        landmark_head = nn.ModuleList()
        for i in range(neck_num):
            landmark_head.append(LandmarkHead(input_channels, anchor_num))
        return landmark_head

    def forward(self, inputs):
        if self.label == 'det':
            out = self.body(inputs)
            """
            out :
            B 48 1/4H
            B 64 1/8H
            B 80 1/16H
            B 96 1/32H       
            """
            fec = self.fec(out)
            neck = self.neck(fec)

            feature1 = self.MFA1(neck[0])
            feature2 = self.MFA2(neck[1])
            feature3 = self.MFA3(neck[2])
            feature4 = self.MFA4(neck[3])
            features = [feature1, feature2, feature3, feature4]

            bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)],
                                         dim=1)
            classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],
                                        dim=1)
            ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)],
                                        dim=1)
            if self.phase == 'train':
                output = (bbox_regressions, classifications, ldm_regressions)
            else:
                output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
            return output
        elif self.label == 'cls':
            out = self.body(inputs)
            """
            96  56 56 
            192 28 28
            288 14 14
            384 7  7
            """
            out = list(out.values())
            output = out[3]
            output = self.classifier(output)
            return output
