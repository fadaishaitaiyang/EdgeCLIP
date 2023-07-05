import torch
import torch.nn as nn


# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Block, self).__init__()

#         self.relu = nn.ReLU()

#         self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), padding=(3, 0), bias=False)
#         self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
#         self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3), bias=False)
#         self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

#         self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False)
#         self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
#         self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False)
#         self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

#         self.last = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
#         self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

#     def forward(self, x):
#         x_3 = self.relu(self.bn_m_3(self.m_3(x)))
#         x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

#         x_5 = self.relu(self.bn_m_5(self.m_5(x)))
#         x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

#         x = x_3 + x_5
#         x = self.relu(self.bn_last(self.last(x)))

#         return x
class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=7, stride=1, padding=3, dilation=1, groups=inplanes, bias=bias)
        self.bn = nn.BatchNorm2d(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
        self.activation = nn.GELU()
        self.pointwise1 = nn.Conv2d(planes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bn(x_1)
        x_2 = self.conv2(x)
        x_2 = self.bn(x_2)
        x = x_1 + x_2
        # x = self.pointwise(x)
        x = self.activation(x)
        x = self.pointwise(x)
        return x
# class af(nn.Module):
#     def __init__(self,dim,norm_layer=nn.BatchNorm2d):
#         super(af, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=5,padding=2)
#         self.dilation1 = nn.Sequential(
#             SeparableConv2d(dim, dim, kernel_size=3, padding=1, dilation=1, bias=False),
#             norm_layer(dim),
#             nn.ReLU(inplace=True))
#         self.dilation2 = nn.Sequential(
#             SeparableConv2d(dim, dim, kernel_size=3, padding=2, dilation=2, bias=False),
#             norm_layer(dim),
#             nn.ReLU(inplace=True))
#         self.dilation3 = nn.Sequential(
#             SeparableConv2d(dim, dim, kernel_size=3, padding=4, dilation=4, bias=False),
#             norm_layer(dim),
#             nn.ReLU(inplace=True))
#         self.dilation4 = nn.Sequential(
#             SeparableConv2d(dim, dim, kernel_size=3, padding=8, dilation=8, bias=False),
#             norm_layer(dim),
#             nn.ReLU(inplace=True))
#         self.conv2 = nn.Conv2d(dim, dim, 1)
#     def forward(self,x):
#         u=x.clone()
#         attn= self.conv1(x)
#         attn_1 = self.dilation1(attn)
#         attn_2 = self.dilation2(attn)
#         attn_3 = self.dilation3(attn)
#         attn_4 = self.dilation4(attn)

#         attn = attn+attn_1+attn_2+attn_3+attn_4

#         attn = self.conv2(attn)

#         return attn * u
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

#
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class encoder(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.block1 = SeparableConv2d(64, 512)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.block2 = SeparableConv2d(512, 1024)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.block3 = SeparableConv2d(1024, 2048)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.__init_weight()
        self.layer5_scene = self._make_layer_face(Bottleneck, 256, 2, stride=1)
        self.layer5_face = self._make_layer_face(Bottleneck, 256, 2, stride=1)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        x=nn.Sequential(*layers)
        return nn.Sequential(*layers)

    def _make_layer_face(self, block, planes, blocks, stride=1):
        downsample = None
        self.inplanes = 2048
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim=2, keepdim=True)
        x = x.squeeze(2).transpose(1, 2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
