import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data.mixup import one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features,
                 s=30.0,
                 m=0.50,
                 easy_margin=False,
                 ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.float()  # for fp16
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if label.dim() == 1:
            one_hot = torch.zeros(cosine.size(), device=device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            if self.ls_eps > 0:
                one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        else:
            one_hot = label
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

# https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place/blob/034a7d8665bb4696981698348c9370f2d4e61e35/models/ch_mdl_dolg_efficientnet.py
class DenseCrossEntropy(nn.Module):
    def forward(self, x, t):
        x = x.float()
        t = t.float()

        y = torch.nn.functional.log_softmax(x, dim=-1)

        t = one_hot(t, y.shape[1], device=device).float()

        loss = -y * t
        loss = loss.sum(-1)
        return loss.mean()
