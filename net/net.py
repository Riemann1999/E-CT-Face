import torch.nn as nn
import torch.nn.functional as F
from layer.Position_encoding import build_position_encoding
import torch


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn_1x1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class MFA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MFA, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)

        self.cp33 = Channel_Part(out_channel // 2, reduction=4, tactics='Avg')

        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.cp55 = Channel_Part(out_channel // 4, reduction=4, tactics='Avg')

        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

        self.cp77 = Channel_Part(out_channel // 4, reduction=4, tactics='Max')

        self.cp = Channel_Part(out_channel, reduction=4, tactics='Nan')

    def forward(self, input_neck):
        conv_3x3 = self.conv3X3(input_neck)

        res33 = conv_3x3
        conv_3x3 = self.cp33(conv_3x3)

        conv_5x5_1 = self.conv5X5_1(input_neck)
        conv_5x5 = self.conv5X5_2(conv_5x5_1)

        res55 = conv_5x5
        conv_5x5 = self.cp55(conv_5x5)

        conv_7x7_2 = self.conv7X7_2(conv_5x5_1)
        conv_7x7 = self.conv7x7_3(conv_7x7_2)

        res77 = conv_7x7
        conv_7x7 = self.cp77(conv_7x7)

        out = torch.cat([conv_3x3, conv_5x5, conv_7x7], dim=1)
        res = torch.cat([res33, res55, res77], dim=1)
        out = res + res * self.cp(out).expand_as(res)

        out = F.relu(out)
        return out


class Channel_Part(nn.Module):
    def __init__(self, channel, reduction=16, tactics='Avg'):
        super(Channel_Part, self).__init__()
        self.tactics = tactics
        if self.tactics == 'Avg':
            self.gap = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(channel // reduction),
                nn.Hardswish(),
            )
        elif self.tactics == 'Max':
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(channel // reduction),
                nn.Hardswish(),
            )
        elif self.tactics == "Nan":
            self.fc = nn.Sequential(
                nn.Conv2d(channel // reduction, channel, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(channel),
                nn.Hardsigmoid()
            )
        elif self.tactics == "Avg&Max":
            self.gap1 = nn.AdaptiveMaxPool2d(1)
            self.gap2 = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(channel // reduction),
                nn.Hardswish(),
            )

    def forward(self, x):
        if self.tactics == 'Avg' or self.tactics == 'Max':
            y = self.fc(self.gap(x))
            return y
        elif self.tactics == 'Nan':
            y = self.fc(x)
            return y
        elif self.tactics == 'Avg&Max':
            y = self.fc(self.gap1(x)) + self.fc(self.gap2(x))
            return y


class Detail_Preservation(nn.Module):
    def __init__(self, dimension=1):
        super(Detail_Preservation, self).__init__()
        self.d = dimension

    def forward(self, x):
        # x => B C H W
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], self.d)


class FEC(nn.Module):
    def __init__(self, in_channels_list, trans_channels_list):
        super(FEC, self).__init__()

        self.conv11_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_list[0], out_channels=trans_channels_list[0], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(trans_channels_list[0]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=trans_channels_list[0], out_channels=trans_channels_list[0], kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=trans_channels_list[0]),
            Detail_Preservation(),
            nn.BatchNorm2d(4 * trans_channels_list[0]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=4 * trans_channels_list[0], out_channels=4 * trans_channels_list[0],
                      kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=4 * trans_channels_list[0]),
            nn.BatchNorm2d(4 * trans_channels_list[0]),
            nn.Hardswish(),

        )
        self.conv12_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_list[1], out_channels=trans_channels_list[1], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(trans_channels_list[1]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=trans_channels_list[1], out_channels=trans_channels_list[1], kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=trans_channels_list[1]),
            Detail_Preservation(),
            nn.BatchNorm2d(4 * trans_channels_list[1]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=4 * trans_channels_list[1], out_channels=4 * trans_channels_list[1],
                      kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=4 * trans_channels_list[1]),
            nn.BatchNorm2d(4 * trans_channels_list[1]),
            nn.Hardswish(),

        )
        self.conv13_in = nn.Sequential(
            nn.Conv2d(in_channels=in_channels_list[2], out_channels=trans_channels_list[2], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(trans_channels_list[2]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=trans_channels_list[2], out_channels=trans_channels_list[2], kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=trans_channels_list[2]),
            Detail_Preservation(),
            nn.BatchNorm2d(4 * trans_channels_list[2]),
            nn.Hardswish(),

            nn.Conv2d(in_channels=4 * trans_channels_list[2], out_channels=4 * trans_channels_list[2],
                      kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), groups=4 * trans_channels_list[2]),
            nn.BatchNorm2d(4 * trans_channels_list[2]),
            nn.Hardswish(),

        )

        self.conv11_out = nn.Sequential(
            nn.Conv2d(in_channels=4 * trans_channels_list[0], out_channels=in_channels_list[1], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(in_channels_list[1]),
        )
        self.conv12_out = nn.Sequential(
            nn.Conv2d(in_channels=4 * trans_channels_list[1], out_channels=in_channels_list[2], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(in_channels_list[2]),
        )
        self.conv13_out = nn.Sequential(
            nn.Conv2d(in_channels=4 * trans_channels_list[2], out_channels=in_channels_list[3], kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(in_channels_list[3]),
        )
        self.post1 = build_position_encoding(hidden_dim=in_channels_list[0], dep=800)
        self.post2 = build_position_encoding(hidden_dim=in_channels_list[1], dep=400)
        self.post3 = build_position_encoding(hidden_dim=in_channels_list[2], dep=200)

    def forward(self, input_trans):
        input_trans = list(input_trans.values())
        """
        input_channel_list
        48 64 80
        trans_list
        96 120 144
        """
        output1 = input_trans[0]
        x = self.post1(output1)
        x = self.conv11_in(x)
        output2 = self.conv11_out(x) + input_trans[1]

        x2 = self.post2(output2)
        x2 = self.conv12_in(x2)
        output3 = self.conv12_out(x2) + input_trans[2]

        x3 = self.post3(output3)
        x3 = self.conv13_in(x3)
        output4 = self.conv13_out(x3) + input_trans[3]
        out = [output1, output2, output3, output4]

        return out


class Neck(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(Neck, self).__init__()
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = conv_bn_1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn_1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn_1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.output4 = conv_bn_1x1(in_channels_list[3], out_channels, stride=1, leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge3 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input_backbone):
        # input = list(input.values())

        output1 = self.output1(input_backbone[0])
        output2 = self.output2(input_backbone[1])
        output3 = self.output3(input_backbone[2])
        output4 = self.output4(input_backbone[3])

        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="bilinear", align_corners=True)
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="bilinear", align_corners=True)
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3, output4]
        return out
