import torch
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import models
from Dataset.ImageNet_LT import LT_Dataset, LT_CLIP_Dataset
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']


def Conv1(in_planes, out_places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=out_places, kernel_size=7, stride=stride, padding=3, bias=False),
        nn.BatchNorm2d(out_places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, out_places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = x
        y = self.classifier(x)
        return feature, y


class ResNet_ImageNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet_ImageNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_planes=3, out_places=64)
        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)

        # add and test
        self.add_mlp = nn.Linear(in_features=2048, out_features=512)
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.xavier_uniform_(m.weight, )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places, stride, downsampling=True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # add
        x = self.add_mlp(x)
        feature = x
        y = self.classifier(x)
        return feature, y


def ResNet18():
    return ResNet([2, 2, 2, 2])


def ResNet50():
    return ResNet_ImageNet([3, 4, 6, 3])
    # return models.resnet50(pretrained=True)


def ResNet101():
    return ResNet([3, 4, 23, 3])


def ResNet152():
    return ResNet([3, 8, 36, 3])

# class ModifiedResNet50(nn.Module):
#
#     def __init__(self, original_model, num_classes):
#         super(ModifiedResNet50, self).__init__()
#         self.resnet = original_model
#         # 添加新的线性层
#         self.new_fc = nn.Linear(512, num_classes)
#         # 替换原有的分类层
#         self.resnet.fc = nn.Sequential(
#             nn.Linear(original_model.fc.in_features, 512),  # 添加一个新的 Linear 层
#             nn.ReLU(),  # 可以添加激活函数
#             self.new_fc  # 最后的分类层
#         )
#
#     def forward(self, x):
#         return self.resnet(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50().to(device)
    # model = models.resnet50(pretrained=False).to(device)
    # model.
    img_root = 'E:\\Datasets\\ImageNet'
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_txt = 'E:\\PycharmProjects\\CLIP2MIX\\ImageNet_LT_test.txt'
    test_data = LT_Dataset(root=img_root, txt=test_txt, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=0)
    total_corrects = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, outputs = model(images)
            # outputs = model(images)
            # print(outputs.shape)
            # _, preds = torch.nn.functional.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            print(preds)
            acc = (preds == labels).sum().item()
            # print(acc)
            total_corrects += acc
            # print(preds)
            # break
    print(total_corrects/len(test_data))
