import torch
import torch.nn as nn
from .swin import (swin_t,swin_b)
torch.cuda.empty_cache()
import torch.nn.functional as F
from thop import profile
from .MIIM import feature_fusion_block

class LDFormer(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1024]):
        super(LDFormer, self).__init__()
        # 动态创建多个阶段
        self.stages = nn.ModuleList([
            self._make_stage(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

    def _make_stage(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs

class image_base(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(image_base, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.pool(x)
        return x
class LIFormer(nn.Module):
    def __init__(self, channels=[3, 64, 128, 256, 512, 1024]):
        super(LIFormer, self).__init__()
        self.stages = nn.ModuleList([
            image_base(channels[i], channels[i+1]) for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


def _transform_outputs(outputs):
    upsampled_outputs = [
        nn.functional.interpolate(
            input=x,
            size=outputs[0].shape[2:],
            mode='bilinear',
            align_corners=False) for x in outputs
    ]
    outputs = torch.cat(upsampled_outputs, dim=1)
    return outputs

class ChannelReducer(nn.Module):
    def __init__(self, in_channels):
        super(ChannelReducer, self).__init__()
        self.conv_reduce = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_reduce(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class total_model(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super().__init__()
        self.encoder2 = swin_b(pretrained=True)
        self.LDFormer = LDFormer()
        self.LIFormer = LIFormer()
        self.final = nn.Conv2d(dim * 23, n_class, kernel_size=1, stride=1, padding=0)
        self.ffb1 = feature_fusion_block(dim=dim*1)
        self.ffb2 = feature_fusion_block(dim=dim*2)
        self.ffb3 = feature_fusion_block(dim=dim*4)
        self.ffb4 = feature_fusion_block(dim=dim*8)
        self.num_attentions = 2
        self.ChannelReducer1 = ChannelReducer(256)
        self.ChannelReducer2 = ChannelReducer(512)
        self.ChannelReducer3 = ChannelReducer(1024)
        self.ChannelReducer4 = ChannelReducer(2048)


    def forward(self, x ,xx):
        out = self.encoder2(x)
        swin_b1, swin_b2, swin_b3, swin_b4 = out[0], out[1], out[2], out[3]
        _, LD1, LD2, LD3, LD4 = self.LDFormer(xx)
        _,LI1, LI2, LI3, LI4 = self.LIFormer(x)

        swin = [swin_b1, swin_b2, swin_b3, swin_b4]
        LI = [LI1, LI2, LI3, LI4]

        for i in range(len(swin)):
            add = swin[i] + LI[i]
            mlt = swin[i] * LI[i]
            swin[i] = torch.cat([add, mlt], dim=1)
        swin_b1, swin_b2, swin_b3, swin_b4 = swin

        swin_b1 = self.ChannelReducer1(swin_b1)
        swin_b2 = self.ChannelReducer2(swin_b2)
        swin_b3 = self.ChannelReducer3(swin_b3)
        swin_b4 = self.ChannelReducer4(swin_b4)

        for _ in range(self.num_attentions):
            swin_b1,LD1 = self.ffb1(swin_b1,LD1)
            swin_b2,LD2 = self.ffb2(swin_b2,LD2)
            swin_b3,LD3 = self.ffb3(swin_b3,LD3)
            swin_b4,LD4 = self.ffb4(swin_b4,LD4)

        outputs = [swin_b1,swin_b2,swin_b3,swin_b4,LI4]
        outputs = _transform_outputs(outputs)
        output = self.final(outputs)
        return output

class EncoderDecoder(nn.Module):
    def __init__(self, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255)):
        super(EncoderDecoder, self).__init__()
        # self.backbone = total_model(128, 40)
        self.backbone = total_model(128, 40)
        self.criterion = criterion

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        out = self.backbone(rgb, modal_x)
        out = F.interpolate(out, size=orisize[-2:], mode='bilinear', align_corners=False)
        return out

    def forward(self, rgb, modal_x=None, label=None):
        out = self.encode_decode(rgb, modal_x)
        if label is not None:
            loss = self.criterion(out, label.long())
            return loss
        return out


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    rgb = Variable(torch.rand(1,3,480,640)).cuda()
    modal_x = Variable(torch.rand(1,3,480,640)).cuda()
    model = EncoderDecoder().cuda()



