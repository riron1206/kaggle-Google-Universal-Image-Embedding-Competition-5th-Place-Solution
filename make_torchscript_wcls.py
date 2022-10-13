import gc
import os
import sys
import torch
import torch.nn as nn
from zipfile import ZipFile
from functools import partial
from torchvision import models
from torchvision import transforms
import open_clip
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path


####################
# Config
####################
conf_dict = {'ckpt_path': './models/kqi_3090_ex072_multiinput_emb_arcface_sam_gpGf_ViT-H_fold0_seed0_ep650_bestlb.pth',
             'ckpt_cls_path': './models/cls_last.ckpt',
             'pt_path': './submissions/yokoi_best_clstta.pt'
             }
conf_base = OmegaConf.create(conf_dict)

def load_pytorch_model_y(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    print(res)
    return model

def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    print(res)
    return model
    
def get_encoder():
    model_name, param_name = 'ViT-H-14', 'laion2b_s32b_b79k'
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=param_name)
    model = model.visual
    return model


class CLIPMODEL(nn.Module):
    def __init__(self, encoder, head, classifier):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.classifier = classifier

    def normalize(self,x):
        return transforms.functional.normalize(x/255, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    def ccrop_resize(self,x,cratio: float):
        B,C,H,W = x.shape
        x = transforms.functional.center_crop(x, [int(H * cratio), int(W *cratio)])
        x = transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return self.normalize(x)

    def ccrop_short_resize(self,x):
        B,C,H,W = x.shape
        x = transforms.functional.center_crop(x, [min(H,W), min(H,W)])
        x = transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return self.normalize(x)

    def pad_resize(self,x):
        B,C,H,W = x.shape
        if W >H:
            h_ = W - H
            h1 = h_ // 2
            h2 = h_ - h1
            x = torch.nn.functional.pad(x, (0, 0, h1, h2), mode='constant', value=0.0)
        elif W < H:
            w_ = H - W
            w1 = w_ // 2
            w2 = w_ - w1
            x = torch.nn.functional.pad(x, (w1, w2), mode='constant', value=0.0)
        x = transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return self.normalize(x)

    def pad_crop_resize(self,x):
        B,C,H,W = x.shape
        if W >H:
            h_ = W - H
            h1 = h_ // 2
            h2 = h_ - h1
            x = torch.nn.functional.pad(x, (0, 0, h1, h2), mode='constant', value=0.0)
        elif W < H:
            w_ = H - W
            w1 = w_ // 2
            w2 = w_ - w1
            x = torch.nn.functional.pad(x, (w1, w2), mode='constant', value=0.0)
        x = transforms.functional.center_crop(x, [int((H+W)/2), int((H+W)/2)])
        x = transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return self.normalize(x)

    def pad_ccrop_resize(self,x,cratio: float):
        B,C,H,W = x.shape
        if W >H:
            h_ = W - H
            h1 = h_ // 2
            h2 = h_ - h1
            x = torch.nn.functional.pad(x, (0, 0, h1, h2), mode='constant', value=0.0)
        elif W < H:
            w_ = H - W
            w1 = w_ // 2
            w2 = w_ - w1
            x = torch.nn.functional.pad(x, (w1, w2), mode='constant', value=0.0)
        x = transforms.functional.center_crop(x, [int(max(H,W)*cratio), int(max(H,W)*cratio)])
        x = transforms.functional.resize(x, [224, 224],interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
        return self.normalize(x)

    def concat_image_info(self, emb, H: int, W: int):
        subfeature = torch.tensor([[
            float(H/W),
            float(H/224.0), 
            float(W/224.0),
        ]], device=emb.device).float()
        return torch.cat([emb, subfeature], dim=1)

    def forward(self, x):
        B,C,H,W = x.shape
        ratio = W/H

        if ratio==1.0:
            x_resize = self.ccrop_resize(x,cratio=0.9)

            feat = self.encoder(x_resize)
            feat = self.concat_image_info(feat, H, W)
            feat = self.head(feat)
            return torch.nn.functional.normalize(feat)

        else:
            x_resize = self.ccrop_resize(x,cratio=1.0)
            feat1 = self.encoder(x_resize)
            cls = torch.argmax(self.classifier(feat1))
            feat1 = self.concat_image_info(feat1, H, W)
            feat1 = self.head(feat1)
            if cls == 0 or cls == 8 or cls == 10:
                x_pad = self.pad_ccrop_resize(x,cratio=0.9)
                feat2 = self.encoder(x_pad)
                feat2 = self.concat_image_info(feat2, H, W)
                feat2 = self.head(feat2)

                x_pad = self.pad_ccrop_resize(x,cratio=1.0)
                feat3 = self.encoder(x_pad)
                feat3 = self.concat_image_info(feat3, H, W)
                feat3 = self.head(feat3)
                return torch.nn.functional.normalize((feat1+feat2+feat3)/3.0)   
            else:
                x_pad = self.pad_ccrop_resize(x,cratio=0.9)
                feat2 = self.encoder(x_pad)
                feat2 = self.concat_image_info(feat2, H, W)
                feat2 = self.head(feat2)

                return torch.nn.functional.normalize((feat1+feat2)/2.0)            


def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    
    os.makedirs(Path(conf.pt_path).parent, exist_ok=True)

    encoder = get_encoder().cuda()
    encoder.eval()

    head = torch.nn.Sequential(
            nn.BatchNorm1d(1024+3, affine=False, eps=1e-6), 
            nn.Linear(1024+3, 64)
        )
    head = load_pytorch_model_y(conf.ckpt_path, head, ignore_suffix='neck').cuda()
    head.eval()

    classifier = nn.Sequential(nn.Linear(1024,11))
    classifier = load_pytorch_model(conf.ckpt_cls_path, classifier, ignore_suffix='model.classifier').cuda()
    classifier.eval()

    encoder = torch.jit.trace(encoder, torch.rand(1, 3, 224, 224).cuda())

    model = CLIPMODEL(encoder, head, classifier).cuda()
    model.eval()
    saved_model = torch.jit.script(model)
    saved_model.save(conf.pt_path)

    saved_model = torch.jit.load(conf.pt_path)
    print(saved_model(torch.rand(1,3,300,1200).cuda()))

if __name__ == "__main__":
    main()