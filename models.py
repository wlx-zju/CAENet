import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ******************************** the MobileNetV2 backbone ******************************** #
def conv3x3(in_planes, out_planes, stride=1, bias=False, dilation=1, groups=1):
    "3x3 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=bias, groups=groups)


def conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias, groups=groups)


def batchnorm(in_planes):
    "batch norm 2d"
    return nn.BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)


def convbnrelu(in_planes, out_planes, kernel_size, stride=1, groups=1, act=True):
    "conv-batchnorm-relu"
    if act:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes),
            nn.ReLU6(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=int(kernel_size / 2.), groups=groups,
                      bias=False),
            batchnorm(out_planes))


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        intermed_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(convbnrelu(in_planes, intermed_planes, 1),
                                    convbnrelu(intermed_planes, intermed_planes, 3, stride=stride,
                                               groups=intermed_planes),
                                    convbnrelu(intermed_planes, out_planes, 1, act=False))

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return (out + residual)
        else:
            return out


class mbv2(nn.Module):
    def __init__(self):
        super(mbv2, self).__init__()
        self.mobilenet_config = [[1, 16, 1, 1],
                                 [6, 24, 2, 2],
                                 [6, 32, 3, 2],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1]]
        self.in_planes = 32  # number of input channels
        self.num_layers = len(self.mobilenet_config)

        self.layer1 = convbnrelu(3, self.in_planes, kernel_size=3, stride=2)
        c_layer = 2
        for t, c, n, s in self.mobilenet_config:
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_planes, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_planes = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers))
            c_layer += 1

    def forward(self, x):
        x_2 = self.layer1(x)  # x / 2
        x_4 = self.layer3(self.layer2(x_2))  # 24, x / 4
        x_8 = self.layer4(x_4)  # 32, x / 8
        x_16_1 = self.layer5(x_8)  # 64, x / 16
        x_16_2 = self.layer6(x_16_1)  # 96, x / 16
        x_32_1 = self.layer7(x_16_2)  # 160, x / 32
        x_32_2 = self.layer8(x_32_1)  # 320, x / 32

        return x_2, x_4, x_8, x_16_1, x_16_2, x_32_1, x_32_2


def mobilenetv2(pretrained=False):
    model = mbv2()
    if pretrained:
        model_dir = './pretrained_weights/mobilenetv2-e6e8dd43.pth'
        model.load_state_dict(torch.load(model_dir))
        print('Load from pretrained weights', model_dir)
    return model


# ******************************** the ResNet18 backbone ******************************** #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x_2 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x_4 = self.layer1(x_2)
        x_8 = self.layer2(x_4)
        x_16 = self.layer3(x_8)
        x_32 = self.layer4(x_16)

        return x_2, x_4, x_8, x_16, x_32


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model_dir = './pretrained_weights/resnet18-5c106cde.pth'
        model.load_state_dict(torch.load(model_dir), strict=False)
        print('Load from pretrained weights', model_dir)
    return model


# ******************************** the STDCNet backbone ******************************** #
class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, num_classes=1000, dropout=0.20,
                 pretrained=True, use_conv_last=False):
        super(STDCNet1446, self).__init__()
        block = CatBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        if pretrained:
            pretrain_model = './pretrained_weights/STDCNet1446_76.47.tar'
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)

    def init_weight(self, pretrain_model):
        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        x_2 = self.x2(x)
        x_4 = self.x4(x_2)
        x_8 = self.x8(x_4)
        x_16 = self.x16(x_8)
        x_32 = self.x32(x_16)
        if self.use_conv_last:
            x_32 = self.conv_last(x_32)

        return x_2, x_4, x_8, x_16, x_32

# ******************************** the proposed IRP and LAF modules ******************************** #
class IRP(nn.Module):
    def __init__(self, in_planes):
        super(IRP, self).__init__()
        self.max3 = True
        self.branch1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=(3 - 1) // 2),
                                     conv1x1(in_planes, in_planes // 4, stride=1, bias=False, groups=1))
        self.branch2 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=(3 - 1) // 2),
                                     conv1x1(in_planes // 4, in_planes // 4, stride=1, bias=False, groups=1))
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=(5 - 1) // 2),
                                     conv1x1(in_planes, in_planes // 4, stride=1, bias=False, groups=1))
        self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=(5 - 1) // 2),
                                     conv1x1(in_planes // 4, in_planes // 4, stride=1, bias=False, groups=1))
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x1)
        x3 = self.branch3(x)
        x4 = self.branch4(x3)
        x_out = torch.cat((x1, x2, x3, x4), dim=1)
        x_out = x_out + x

        return x_out


class LAF(nn.Module):
    def __init__(self, feature_num, in_planes, relu):
        super(LAF, self).__init__()
        self.feature_num = feature_num

        # Spatial Attention
        self.spatial_conv = nn.Sequential(nn.Conv2d(2, feature_num, kernel_size=7, stride=1, padding=3, bias=False),
                                          nn.BatchNorm2d(feature_num, eps=1e-5, momentum=0.01, affine=True))

        # Channel Attention
        self.fc1 = nn.Linear(in_planes, in_planes // 8)
        self.relu = relu(inplace=True)
        self.fc2 = nn.Linear(in_planes // 8, in_planes)

    def forward(self, x0, x1, x2=None):
        if self.feature_num == 2:
            gather = x0 + x1
        elif self.feature_num == 3:
            gather = x0 + x1 + x2

        b, c, h, w = gather.shape

        # Spatial Attention
        s = torch.cat((torch.max(gather, 1)[0].unsqueeze(1), torch.mean(gather, 1).unsqueeze(1)), dim=1)
        s = self.spatial_conv(s).unsqueeze(dim=2)

        # Channel Attetion
        x_flatten1 = (gather.sum(dim=2).sum(dim=2) / (h * w))
        x_out1 = self.fc2(self.relu(self.fc1(x_flatten1)))
        x_flatten2 = (gather.max(dim=2)[0]).max(dim=2)[0]
        x_out2 = self.fc2(self.relu(self.fc1(x_flatten2)))
        d = x_out1 + x_out2
        d = d.view(b, 1, c, 1, 1)

        f = F.softmax(d * s, dim=1)  # softmax along the feature level axis

        if self.feature_num == 2:
            output = x0 * f[:, 0, ...] + x1 * f[:, 1, ...]
        elif self.feature_num == 3:
            output = x0 * f[:, 0, ...] + x1 * f[:, 1, ...] + x2 * f[:, 2, ...]

        return output


# ******************************** the Context-aware Attentive Enrichment Network ******************************** #
class CAENet1(nn.Module):
    def __init__(self, tasks=['semantic', 'depth'], class_num=13, pretrained=True):
        super(CAENet1, self).__init__()
        self.tasks = tasks
        self.backbone = mobilenetv2(pretrained)

        self.convfeat32_2 = conv1x1(320, 256, bias=False)
        self.convfeat32_1 = conv1x1(160, 256, bias=False)
        self.convfeat16_2 = conv1x1(96, 256, bias=False)
        self.convfeat16_1 = conv1x1(64, 256, bias=False)
        self.convfeat8 = conv1x1(32, 256, bias=False)
        self.convfeat4 = conv1x1(24, 256, bias=False)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.laf4 = LAF(feature_num=2, in_planes=256, relu=nn.ReLU6)
        self.laf3 = LAF(feature_num=3, in_planes=256, relu=nn.ReLU6)
        self.laf2 = LAF(feature_num=2, in_planes=256, relu=nn.ReLU6)
        self.laf1 = LAF(feature_num=2, in_planes=256, relu=nn.ReLU6)

        self.irp4 = IRP(256)
        self.irp3 = IRP(256)
        self.irp2 = IRP(256)
        self.irp1 = IRP(256)

        if 'semantic' in self.tasks:
            self.task_conv1 = conv1x1(256, 64, groups=1, bias=False)
            self.task_out1 = conv3x3(64, class_num, bias=True)
        if 'depth' in self.tasks:
            self.task_conv2 = conv1x1(256, 64, groups=1, bias=False)
            self.task_out2 = conv3x3(64, 1, bias=True)

        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x_2, x_4, x_8, x_16_1, x_16_2, x_32_1, x_32_2 = self.backbone(x)

        feat32_2 = self.convfeat32_2(x_32_2)
        feat32_1 = self.convfeat32_1(x_32_1)
        feat32 = self.laf4(feat32_1, feat32_2) + feat32_2
        feat32 = self.conv_adapt4(self.irp4(self.relu(feat32)))
        feat32 = F.interpolate(feat32, size=x_16_2.shape[-2:], mode='bilinear', align_corners=True)

        feat16_2 = self.convfeat16_2(x_16_2)
        feat16_1 = self.convfeat16_1(x_16_1)
        feat16 = self.laf3(feat16_1, feat16_2, feat32) + feat32
        feat16 = self.conv_adapt3(self.irp3(self.relu(feat16)))
        feat16 = F.interpolate(feat16, size=x_8.shape[-2:], mode='bilinear', align_corners=True)

        feat8 = self.convfeat8(x_8)
        feat8 = self.laf2(feat8, feat16) + feat16
        feat8 = self.conv_adapt2(self.irp2(self.relu(feat8)))
        feat8 = F.interpolate(feat8, size=x_4.shape[-2:], mode='bilinear', align_corners=True)

        feat4 = self.convfeat4(x_4)
        feat4 = self.laf1(feat4, feat8) + feat8
        feat4 = self.irp1(self.relu(feat4))

        pred = {}
        if 'semantic' in self.tasks:
            pred['semantic'] = self.task_out1(self.relu(self.task_conv1(feat4)))
            pred['semantic'] = F.interpolate(pred['semantic'], size=x.shape[-2:], mode='bilinear', align_corners=True)
        if 'depth' in self.tasks:
            pred['depth'] = self.task_out2(self.relu(self.task_conv2(feat4)))
            pred['depth'] = F.interpolate(pred['depth'], size=x.shape[-2:], mode='bilinear', align_corners=True)

        return pred


class CAENet2(nn.Module):
    def __init__(self, backbone='resnet18', tasks=['semantic', 'depth'], class_num=13, pretrained=True):
        super(CAENet2, self).__init__()
        self.tasks = tasks

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=pretrained)
            self.convfeat32 = conv1x1(512, 256, bias=False)
            self.convfeat16 = conv1x1(256, 256, bias=False)
            self.convfeat8 = conv1x1(128, 128, bias=False)
            self.convfeat4 = conv1x1(64, 128, bias=False)
        elif backbone == 'stdcnet':
            self.backbone = STDCNet1446(pretrained=pretrained, use_conv_last=False)
            self.convfeat32 = conv1x1(1024, 256, bias=False)
            self.convfeat16 = conv1x1(512, 256, bias=False)
            self.convfeat8 = conv1x1(256, 128, bias=False)
            self.convfeat4 = conv1x1(64, 128, bias=False)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 128, bias=False)
        self.conv_adapt2 = conv1x1(128, 128, bias=False)

        self.laf3 = LAF(feature_num=2, in_planes=256, relu=nn.ReLU)
        self.laf2 = LAF(feature_num=2, in_planes=128, relu=nn.ReLU)
        self.laf1 = LAF(feature_num=2, in_planes=128, relu=nn.ReLU)

        self.irp4 = IRP(256)
        self.irp3 = IRP(256)
        self.irp2 = IRP(128)
        self.irp1 = IRP(128)

        if 'semantic' in self.tasks:
            self.task_conv1 = conv1x1(128, 128, groups=1, bias=False)
            self.task_out1 = conv3x3(128, class_num, bias=True)
        if 'depth' in self.tasks:
            self.task_conv2 = conv1x1(128, 128, groups=1, bias=False)
            self.task_out2 = conv3x3(128, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_2, x_4, x_8, x_16, x_32 = self.backbone(x)

        feat32 = self.conv_adapt4(self.irp4(self.relu(self.convfeat32(x_32))))
        feat32 = F.interpolate(feat32, size=x_16.shape[-2:], mode='bilinear', align_corners=True)

        feat16 = self.convfeat16(x_16)
        feat16 = self.laf3(feat16, feat32) + feat32
        feat16 = self.conv_adapt3(self.irp3(self.relu(feat16)))
        feat16 = F.interpolate(feat16, size=x_8.shape[-2:], mode='bilinear', align_corners=True)

        feat8 = self.convfeat8(x_8)
        feat8 = self.laf2(feat8, feat16) + feat16
        feat8 = self.conv_adapt2(self.irp2(self.relu(feat8)))
        feat8 = F.interpolate(feat8, size=x_4.shape[-2:], mode='bilinear', align_corners=True)

        feat4 = self.convfeat4(x_4)
        feat4 = self.laf1(feat4, feat8) + feat8
        feat4 = self.irp1(self.relu(feat4))

        pred = {}
        if 'semantic' in self.tasks:
            pred['semantic'] = self.task_out1(self.relu(self.task_conv1(feat4)))
            pred['semantic'] = F.interpolate(pred['semantic'], size=x.shape[-2:], mode='bilinear', align_corners=True)
        if 'depth' in self.tasks:
            pred['depth'] = self.task_out2(self.relu(self.task_conv2(feat4)))
            pred['depth'] = F.interpolate(pred['depth'], size=x.shape[-2:], mode='bilinear', align_corners=True)

        return pred


def CAENet(tasks=['semantic', 'depth'],
           class_num=13,
           pretrained=False,
           backbone='mobilenetv2'):
    if backbone == 'mobilenetv2':
        model = CAENet1(tasks, class_num, pretrained)
    else:
        model = CAENet2(backbone, tasks, class_num, pretrained)
    return model
