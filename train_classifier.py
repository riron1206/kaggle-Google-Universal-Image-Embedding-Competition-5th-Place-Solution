import os
import sys
from PIL import Image
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torchmetrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import model_selection
from sklearn import metrics
import albumentations as A
from PIL import Image
from torchvision import transforms
import timm
from omegaconf import OmegaConf

import open_clip
from pytorch_metric_learning import losses

from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from tqdm import tqdm
import glob
import random

from src.arcface_models import DenseCrossEntropy

####################
# Config
####################
conf_dict = {'data_dir': './data/130k',
             'num_samples': 50,
             'output_dir': './output/classfier_130k',
             'emb_dir': './output/classfier_130k/embs/',
             'seed': 2021,
             'batch_size': 64*8,
             'epoch': 30,
             'n_classes': None,
             'mkemb': True,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)

####################
# Util
####################
def make_embs(model, dataloader, dist_path, suffix=''):
    os.makedirs(dist_path, exist_ok=True)
    emb_buffer = []
    label_buffer = []
    emb_length = 0
    count_npy = 0
    with torch.no_grad():
        for d in tqdm(dataloader):
            images = d['images']
            targets = d['targets']

            batch_size = images.shape[0]
            emb_length += batch_size
            images = images.cuda()
            emb = model(images)
            emb_buffer.append(emb.cpu().numpy())
            label_buffer.append(targets.cpu().numpy())
            if emb_length>150000:
                emb_buffer = np.concatenate(emb_buffer)
                np.save(os.path.join(dist_path, f'emb_{count_npy}_{suffix}.npy'), emb_buffer)
                label_buffer = np.concatenate(label_buffer)
                np.save(os.path.join(dist_path, f'label_{count_npy}_{suffix}.npy'), label_buffer)
                emb_buffer = []
                label_buffer = []
                emb_length = 0
                count_npy += 1

        emb_buffer = np.concatenate(emb_buffer)
        np.save(os.path.join(dist_path, f'emb_{count_npy}_{suffix}.npy'), emb_buffer)
        label_buffer = np.concatenate(label_buffer)
        np.save(os.path.join(dist_path, f'label_{count_npy}_{suffix}.npy'), label_buffer)
        emb_buffer = []
        label_buffer = []
        emb_length = 0
        count_npy += 1

####################
# Dataset
####################
class ImageDataset(Dataset):
    def __init__(self, df, transform=None, conf=None):

        self.df = df.reset_index(drop=True)
        self.img_path = df['path'].values
        self.label = df['class'].values
        self.transform = transform
        self.conf = conf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_path[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = self.label[idx]

        return {
            "images": image,
            "targets": torch.tensor(label, dtype=torch.long)
        }

class EMBDataset(Dataset):
    def __init__(self, emb_npy_list, label_npy_list):
        self.emb_npy = np.concatenate([np.load(pth) for pth in emb_npy_list])
        self.label_npy = np.concatenate([np.load(pth) for pth in label_npy_list])

    def __len__(self):
        return self.emb_npy.shape[0]

    def __getitem__(self, idx):
        return self.emb_npy[idx], self.label_npy[idx]

####################
# Image Dataset
####################
def get_image_dataloaders(conf):
    # For Products10k
    df = pd.read_csv(os.path.join(conf.data_dir,'train.csv'))
    cls_map = {'apparel':0,  'artwork':1,  'cars':2,  'dishes':3,  'furniture':4,  'illustrations':5,  'landmark':6,  'meme':7,  'packaged':8,  'storefronts':9,  'toys':10}
    df['class'] = df['label'].map(cls_map)
    df['path'] = conf.data_dir + '/' + df['label'] + '/' + df['image_name']

    # cv split
    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=conf.seed)
    for n, (train_index, val_index) in enumerate(skf.split(df, df["label"])):
        df.loc[val_index, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)

    train_df = df[df['fold'] != 0]
    valid_df = df[df['fold'] == 0]

    train_transform = transforms.Compose([#transforms.PILToTensor(),
                                            transforms.RandomResizedCrop((224,224), scale=(1.0, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    valid_transform = transforms.Compose([#transforms.PILToTensor(),
                                            transforms.RandomResizedCrop((224,224), scale=(1.0, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    train_dataset = ImageDataset(train_df, transform=train_transform, conf=conf)
    valid_dataset = ImageDataset(valid_df, transform=valid_transform, conf=conf)

    return DataLoader(train_dataset, batch_size=conf.batch_size, num_workers=2, shuffle=False, pin_memory=True, drop_last=False), \
        DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=2, shuffle=False, pin_memory=True, drop_last=False), \
            11



####################
# Data Module
####################

class CustomDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None):
        if stage == 'fit':
            train_embs = sorted(glob.glob(os.path.join(self.conf.emb_dir, 'train/','emb_*.npy')))
            train_labels = sorted(glob.glob(os.path.join(self.conf.emb_dir, 'train/','label_*.npy')))
            valid_embs = sorted(glob.glob(os.path.join(self.conf.emb_dir, 'valid/','emb_*.npy')))
            valid_labels = sorted(glob.glob(os.path.join(self.conf.emb_dir, 'valid/','label_*.npy')))

            self.train_dataset = EMBDataset(emb_npy_list=train_embs, label_npy_list=train_labels)
            self.valid_dataset = EMBDataset(emb_npy_list=valid_embs, label_npy_list=valid_labels)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=16, shuffle=True, pin_memory=True, drop_last=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=16, shuffle=False, pin_memory=True, drop_last=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.conf.batch_size, num_workers=16, shuffle=False, pin_memory=True, drop_last=False, persistent_workers=True)

####################
# Custom Model
####################
def get_encoder():
    model_name, param_name = 'ViT-H-14', 'laion2b_s32b_b79k'
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=param_name)
    model = model.visual
    return model

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(1024,11))

    def forward(self, x):
        return self.classifier(x)


####################
# Lightning Module
####################

class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        self.model = CustomModel()
        self.criteria = DenseCrossEntropy()
        self.acc_top1 = Accuracy(top_k=1)


    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-2, weight_decay=0.1)
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=self.hparams.epoch,
                                          cycle_mult=1.0,
                                          max_lr=1e-2,
                                          min_lr=1e-6,
                                          warmup_steps=3,
                                          gamma=1.0)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criteria(y_hat, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.model(x)

        loss = self.criteria(y_hat, y)
        self.log('val_loss', loss)

        self.acc_top1(y_hat, y)


        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu().detach().numpy()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu().detach().numpy()
        self.log('val_top1_acc', self.acc_top1, on_step=False, on_epoch=True)
        self.log('avg_val_loss', avg_val_loss)


####################
# Train
####################
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(conf.seed)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='avg_val_loss',
                                            save_last=True, save_top_k=1, mode='min',
                                            save_weights_only=True, filename='{step}-{avg_val_loss:.4f}')


    train_dataloader, valid_dataloader, num_classes = get_image_dataloaders(conf)

    encoder = get_encoder().cuda().eval()
    if conf.mkemb:
        for t in range(1):
            make_embs(encoder, train_dataloader, os.path.join(conf.emb_dir, 'train/'), suffix=str(t))
        make_embs(encoder, valid_dataloader, os.path.join(conf.emb_dir, 'valid/'), suffix='0')
        exit()


    conf.n_classes = num_classes
    lit_model = LitSystem(conf)
    data_module = CustomDataModule(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,#0.2,
        check_val_every_n_epoch=20,
        accumulate_grad_batches=1,
        gradient_clip_val=1000,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()