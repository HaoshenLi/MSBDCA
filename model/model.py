import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vgg_3d import *
from einops import rearrange
    

class Model_MSBDCA_Image(nn.Module):
    def __init__(self, 
                **kwargs):
        super().__init__()
        self.backbone = MSBDCA_Image(in_channels=1)
            
        self.classifier = nn.Sequential(
            nn.Linear(1152, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2, bias=True)
        )  
        
    def forward(self, x, mask, batch):
        batch_size, instance_num = x.shape[0], x.shape[1]
        x = rearrange(x, 'b i c d h w -> (b i) c d h w')
        x, image_embs = self.backbone(x, mask)
        image_embs = rearrange(image_embs, 'b i c -> (b i) c')
        x = rearrange(x, '(b i) c -> b i c', b=batch_size)
        x = torch.mean(x, dim=1)
        fuse_feature = torch.concat((image_embs, x), dim=1)
        pred = self.classifier(fuse_feature)
        return {
            "pred": pred
        }


class Model_MSBDCA_Image_Clinic(nn.Module):
    def __init__(self, 
                **kwargs):
        super().__init__()
        self.backbone = MSBDCA_Image_Clinic(in_channels=1)
        self.infoembs = nn.ModuleList([
            nn.Embedding(num_embeddings=5, embedding_dim=576) for _ in range(5)])
        self.classifier = nn.Sequential(
            nn.Linear(1728, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2, bias=True)
        )  
        
    def forward(self, x, mask, batch):
        batch_size, instance_num = x.shape[0], x.shape[1]
        clicinfo = batch['clicinfo'].to(x.device) #(B, )
        clicinfo = torch.stack([self.infoembs[i](clicinfo[:, i]) for i in range(len(self.infoembs))], dim=1) # (B, 5, L)
        x = rearrange(x, 'b i c d h w -> (b i) c d h w')
        x, image_embs, clicinfo = self.backbone(x, mask, clicinfo)
        image_embs = rearrange(image_embs, 'b i c -> (b i) c')
        clicinfo = torch.mean(clicinfo, dim=1)
        x = rearrange(x, '(b i) c -> b i c', b=batch_size)
        x = torch.mean(x, dim=1)
        fuse_feature = torch.concat((clicinfo, image_embs, x), dim=1)
        pred = self.classifier(fuse_feature)
        return {
            "pred": pred
        }
    
class Model_Clinic(nn.Module):
    def __init__(self, 
                **kwargs):
        super().__init__()
        features = 576
        
        self.infoembs = nn.ModuleList([
            nn.Embedding(num_embeddings=5, embedding_dim=features) for _ in range(5)])

        self.classifier = nn.Sequential(
            nn.Linear(features, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 2, bias=True)
        )  
        
    def forward(self, X, M, batch):
        clicinfo = batch['clicinfo'].to(X.device) #(B, )
        clicinfo = torch.stack([self.infoembs[i](clicinfo[:, i]) for i in range(len(self.infoembs))], dim=1) # (B, 5, L)
        clicinfo = torch.mean(clicinfo, dim=1)
        pred = self.classifier(clicinfo)

        return {
            "pred": pred
        }

    