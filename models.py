import torch
import torch.nn as nn
import torchvision


class AlexNetCustom(nn.Module):

    def __init__(self, n_classes, dropout=0.5, save_hm=False) -> None:

        super().__init__()
        self.save_hm = save_hm
        m = torchvision.models.alexnet(pretrained=True)
        self.features = m.features
        self.avgPool = m.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=n_classes)
        )
        self.heat_maps = []
    
    def freezeFeatures(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if self.save_hm and layer.__class__.__name__ == "Conv2d":
                self.heat_maps.append(x)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG19Custom(nn.Module):

    def __init__(self, n_classes, dropout=0.5, save_hm=False) -> None:
        super().__init__()
        self.save_hm = save_hm
        m = torchvision.models.vgg19(pretrained=True)
        self.features = m.features
        self.avgPool = m.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=n_classes)
        )
        self.heat_maps = []
    
    def freezeFeatures(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if self.save_hm and layer.__class__.__name__ == "Conv2d":
                self.heat_maps.append(x)

        x_v = torch.clone(x)
        self.heat_maps.append(x_v)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Inception(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def forward(self, data):
        return data