from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from collections import OrderedDict


class ConvAttention(nn.Module):
    # 构建前向传播的空间注意力型卷积
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(ConvAttention, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(mid_channels),
                                     nn.ReLU(inplace=True), )
        self.branch2 = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm2d(mid_channels),
                                     nn.ReLU(inplace=True), )

    def forward(self, x):
        x = self.branch1(x)
        out = self.branch2(x)

        return out


# DA模块
class DualAttention(nn.Module):
    # 构建特征融合的通道注意力卷积
    def __init__(self, LowChannels, HighChannels):
        super(DualAttention, self).__init__()  # 此处LowChannels代表着低语义信息的特征图（即为高分辨率图像），HighChannels则代表低分辨率图像
        self.add_high = nn.Sequential(nn.Conv2d(HighChannels, LowChannels, 1),
                                      )
        self.space_attention = nn.Sequential(nn.Conv2d(HighChannels, 1, 9, padding=4),
                                             nn.BatchNorm2d(1),
                                             nn.Sigmoid(), )
        self.channel_attention = nn.Sequential(nn.Conv2d(HighChannels, LowChannels, 1),
                                               nn.BatchNorm2d(LowChannels),
                                               nn.Sigmoid())

    def forward(self, LowFeatures, HighFeatures):  # 依次为高分辨率、低分辨率图像
        Residual = LowFeatures
        _, _, h, w = LowFeatures.shape
        LowFeatures = F.interpolate(self.add_high(HighFeatures), size=[h, w], mode='bilinear',
                                    align_corners=False) + LowFeatures
        attention = F.interpolate(self.space_attention(HighFeatures), size=[h, w], mode='bilinear',
                                  align_corners=False) * self.channel_attention(
            F.interpolate(HighFeatures, size=[1, 1], mode='bilinear', align_corners=False))
        LowFeatures = attention * LowFeatures
        output = Residual + LowFeatures

        return output


# BR模块
class BoundaryRefine(nn.Module):
    # 构建边界修正模块
    def __init__(self, in_channels, num_class):
        super(BoundaryRefine, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, num_class, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_class),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, num_class, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(num_class),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, num_class, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(num_class),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, num_class, kernel_size=1),
            nn.BatchNorm2d(num_class),
        )
        self.out = nn.Sequential(
            nn.Conv2d(num_class * 4, num_class, kernel_size=1),
            nn.BatchNorm2d(num_class),
        )

    def forward(self, x):
        muti_out = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x), self.residual(x)), dim=1)
        result = self.out(muti_out)
        return result


class Up(nn.Module):
    # 将低分辨率特征图映射到高分辨率特征尺度上
    def __init__(self, input_channels, output_channels):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.up(x)


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2, 2), )

    def forward(self, x):
        return self.down(x)


# U-MPSC网络模型
class CrackDetectModel(nn.Module):
    def __init__(self, in_channels=3, num_class=2, base_c=64):
        super(CrackDetectModel, self).__init__()
        channel_list = [base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16]
        # 特征提取分支
        self.DownConv1 = ConvAttention(in_channels, channel_list[0])
        self.Down_1to2 = Down()
        self.DownConv2 = ConvAttention(channel_list[0], channel_list[1])
        self.Down_2to3 = Down()
        self.DownConv3 = ConvAttention(channel_list[1], channel_list[2])
        self.Down_3to4 = Down()
        self.DownConv4 = ConvAttention(channel_list[2], channel_list[3])
        self.Down_4to5 = Down()
        self.DownConv5 = ConvAttention(channel_list[3], channel_list[4])
        # 特征聚合分支
        self.UpAttention4 = DualAttention(channel_list[3], channel_list[4])
        self.Up_5to4 = Up(channel_list[4], channel_list[3])
        self.UpConv4 = ConvAttention(channel_list[4], channel_list[3])
        self.UpAttention3 = DualAttention(channel_list[2], channel_list[3])
        self.Up_4to3 = Up(channel_list[3], channel_list[2])
        self.UpConv3 = ConvAttention(channel_list[3], channel_list[2])
        self.UpAttention2 = DualAttention(channel_list[1], channel_list[2])
        self.Up_3to2 = Up(channel_list[2], channel_list[1])
        self.UpConv2 = ConvAttention(channel_list[2], channel_list[1])
        self.UpAttention1 = DualAttention(channel_list[0], channel_list[1])
        self.Up_2to1 = Up(channel_list[1], channel_list[0])
        self.UpConv1 = ConvAttention(channel_list[1], channel_list[0])
        # 边界修正模块
        self.BoundaryRefine = BoundaryRefine(channel_list[0], num_class)

    def forward(self, x):
        # 特征提取环节
        down1 = self.DownConv1(x)
        down2 = self.DownConv2(self.Down_1to2(down1))
        down3 = self.DownConv3(self.Down_2to3(down2))
        down4 = self.DownConv4(self.Down_3to4(down3))
        down5 = self.DownConv5(self.Down_4to5(down4))
        # 特征聚合环节
        up4 = self.UpConv4(torch.cat((self.UpAttention4(down4, down5), self.Up_5to4(down5)), dim=1))
        up3 = self.UpConv3(torch.cat((self.UpAttention3(down3, up4), self.Up_4to3(up4)), dim=1))
        up2 = self.UpConv2(torch.cat((self.UpAttention2(down2, up3), self.Up_3to2(up3)), dim=1))
        output = self.UpConv1(torch.cat((self.UpAttention1(down1, up2), self.Up_2to1(up2)), dim=1))
        output = self.BoundaryRefine(output)
        return {"out": output}


if __name__ == "__main__":
    model = CrackDetectModel(in_channels=3, num_class=2).eval()
    rand_x = torch.randn(1, 3, 256, 256)
    print(model(rand_x)['out'].shape)
