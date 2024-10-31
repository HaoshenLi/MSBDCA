from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from model.cross_attention import *


__all__ = [
    "VGG",
    "VGG6",
    "MSBDCA_Image",
    "MSBDCA_Image_Clinic",
    "vgg6_3d",
    "vgg6_bn_3d",
    "vgg11_3d",
    "vgg11_bn_3d",
    "vgg13_3d",
    "vgg13_bn_3d",
    "vgg16_3d",
    "vgg16_bn_3d",
    "vgg19_3d",
    "vgg19_bn_3d",
]


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))
        self.avgpool = nn.AdaptiveAvgPool3d((1, 3, 3))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(
            # nn.Linear(64 * 7 * 7, 128),
            nn.Linear(64 * 3 * 3, 128),
            # nn.Linear(512 * 3 * 3, 128),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
  
  
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False, in_channels: int = 1) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = in_channels
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool3d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG6(nn.Module):
    def __init__(
        self, num_classes: int = 1000, init_weights: bool = True, in_channels=2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down1 = nn.MaxPool3d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 3, 3))
        self.flatten = nn.Flatten(1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (16, 144, 144)
        x = self.layer1(x) #(16, 16, 144, 144)
        x = self.down1(x) #(16, 4, 36, 36)
        x = self.layer2(x) #(32, 4, 36, 36)
        x = self.down2(x) #(32, 2, 18, 18)
        x = self.layer3(x) #(64, 2, 18, 18)

        x = self.down3(x) #(64, 1, 9, 9)
        x = self.avgpool(x) #(64, 1, 3, 3)
        x = self.flatten(x)
        return x
    

class MSBDCA_Image(nn.Module):
    def __init__(
        self, num_classes: int = 1000, init_weights: bool = True, in_channels=2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down1 = nn.MaxPool3d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.image_embs = nn.Parameter(torch.zeros(1, 1, 576).float())
        self.local2global_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=20736)
        self.global2local_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=20736,
                                               key_embed_size=576)
        
        self.local2global_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=5184)
        self.global2local_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=5184,
                                               key_embed_size=576)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 3, 3))
        self.flatten = nn.Flatten(1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    # nn.init.constant_(m.bias, 0)
            nn.init.normal_(self.image_embs, mean=0, std=1)

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:
        #x: (16, 144, 144)
        image_embs = self.image_embs.expand(x.shape[0] // 5, 1, -1)
        x = self.layer1(x) #(16, 16, 144, 144)
        x = self.down1(x) #(16, 4, 36, 36)
        x = self.layer2(x) #(32, 4, 36, 36)
        x = self.down2(x) #(32, 2, 18, 18)
        
        c, d, h, w = x.shape[1:]
        x = rearrange(x, '(b l) c d h w -> b l (c d h w)', l=5)
        local2global1, global2local1 = self.local2global_ca1(query=image_embs, key=x, value=x, key_mask=mask), self.global2local_ca1(query=x, key=image_embs, value=image_embs, query_mask=mask)
        x = x + global2local1
        image_embs = image_embs + local2global1
        x = rearrange(x, 'b l (c d h w) -> (b l) c d h w', c=c, d=d, h=h, w=w)

        x = self.layer3(x) #(64, 2, 18, 18)
        x = self.down3(x) #(64, 1, 9, 9)

        c, d, h, w = x.shape[1:]
        x = rearrange(x, '(b l) c d h w -> b l (c d h w)', l=5)
        local2global2, global2local2 = self.local2global_ca2(query=image_embs, key=x, value=x, key_mask=mask), self.global2local_ca2(query=x, key=image_embs, value=image_embs, query_mask=mask)
        x = x + global2local2
        image_embs = image_embs + local2global2
        x = rearrange(x, 'b l (c d h w) -> (b l) c d h w', c=c, d=d, h=h, w=w)

        x = self.avgpool(x) #(64, 1, 3, 3)
        x = self.flatten(x)
        return x, image_embs
    

class MSBDCA_Image_Clinic(nn.Module):
    def __init__(
        self, num_classes: int = 1000, init_weights: bool = True, in_channels=2, dropout: float = 0.1) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels, 8, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down1 = nn.MaxPool3d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False)
        self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down2 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.layer3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                                    nn.BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.ReLU(inplace=True))
        self.down3 = nn.MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.image_embs = nn.Parameter(torch.zeros(1, 1, 576).float())
        self.local2global_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=20736)
        self.global2local_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=20736,
                                               key_embed_size=576)
        self.local2global_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=5184)
        self.global2local_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=5184,
                                               key_embed_size=576)
        
        self.local2pathology_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=20736)
        self.pathology2local_ca1 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=20736,
                                               key_embed_size=576)
        self.local2pathology_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=576,
                                               key_embed_size=5184)
        self.pathology2local_ca2 = CrossAttention(embed_size=576, 
                                               heads=1,
                                               query_embed_size=5184,
                                               key_embed_size=576)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 3, 3))
        self.flatten = nn.Flatten(1)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    # nn.init.constant_(m.bias, 0)
            nn.init.normal_(self.image_embs, mean=0, std=1)

    def forward(self, x: torch.Tensor, mask, clicinfo) -> torch.Tensor:
        #x: (16, 144, 144)
        image_embs = self.image_embs.expand(x.shape[0] // 5, 1, -1)
        x = self.layer1(x) #(16, 16, 144, 144)
        x = self.down1(x) #(16, 4, 36, 36)
        x = self.layer2(x) #(32, 4, 36, 36)
        x = self.down2(x) #(32, 2, 18, 18)
        
        c, d, h, w = x.shape[1:]
        x = rearrange(x, '(b l) c d h w -> b l (c d h w)', l=5)
        local2global1, global2local1 = self.local2global_ca1(query=image_embs, key=x, value=x, key_mask=mask), self.global2local_ca1(query=x, key=image_embs, value=image_embs, query_mask=mask)
        x = x + global2local1
        image_embs = image_embs + local2global1
        local2pathology1, pathology2local1 = self.local2pathology_ca1(query=clicinfo, key=x, value=x, key_mask=mask), self.pathology2local_ca1(query=x, key=clicinfo, value=clicinfo, query_mask=mask)
        x = x + pathology2local1
        clicinfo = clicinfo + local2pathology1
        x = rearrange(x, 'b l (c d h w) -> (b l) c d h w', c=c, d=d, h=h, w=w)

        x = self.layer3(x) #(64, 2, 18, 18)
        x = self.down3(x) #(64, 1, 9, 9)

        c, d, h, w = x.shape[1:]
        x = rearrange(x, '(b l) c d h w -> b l (c d h w)', l=5)
        local2global2, global2local2 = self.local2global_ca2(query=image_embs, key=x, value=x, key_mask=mask), self.global2local_ca2(query=x, key=image_embs, value=image_embs, query_mask=mask)
        x = x + global2local2
        image_embs = image_embs + local2global2
        local2pathology2, pathology2local2 = self.local2pathology_ca2(query=clicinfo, key=x, value=x, key_mask=mask), self.pathology2local_ca2(query=x, key=clicinfo, value=clicinfo, query_mask=mask)
        x = x + pathology2local2
        clicinfo = clicinfo + local2pathology2
        x = rearrange(x, 'b l (c d h w) -> (b l) c d h w', c=c, d=d, h=h, w=w)

        x = self.avgpool(x) #(64, 1, 3, 3)
        x = self.flatten(x)
        return x, image_embs, clicinfo
    

cfgs: Dict[str, List[Union[str, int]]] = {
    "S1": [8, 16, "M", 32, 32, "M", 64, 64, "M"],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, num_classes: int = 1, in_channels: int = 1, **kwargs: Any) -> VGG:
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, in_channels=in_channels), num_classes, **kwargs)
    return model


def vgg6_3d(**kwargs: Any) -> VGG:
    return _vgg("S1", False, **kwargs)


def vgg6_bn_3d(**kwargs: Any) -> VGG:
    return _vgg("S1", True, **kwargs)


def vgg11_3d(**kwargs: Any) -> VGG:
    return _vgg("A", False, **kwargs)


def vgg11_bn_3d(**kwargs: Any) -> VGG:
    return _vgg("A", True, **kwargs)


def vgg13_3d(**kwargs: Any) -> VGG:
    return _vgg("B", False, **kwargs)


def vgg13_bn_3d(**kwargs: Any) -> VGG:
    return _vgg("B", True, **kwargs)


def vgg16_3d(**kwargs: Any) -> VGG:
    return _vgg("D", False, **kwargs)


def vgg16_bn_3d(**kwargs: Any) -> VGG:
    return _vgg("D", True, **kwargs)


def vgg19_3d(**kwargs: Any) -> VGG:
    return _vgg("E", False, **kwargs)


def vgg19_bn_3d(**kwargs: Any) -> VGG:
    return _vgg("E", True, **kwargs)
