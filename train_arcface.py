#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ******************************************************************
# Parameters
# ******************************************************************
BATCH_SIZE = 64*4
MODEL_NAME = "openclip-ViT-H_laion2b"
SIZE = 224
add_name = "_emb_arcface_sam_gpGf_ViT-H"
cfg_m = 0.5
cfg_optimizer = "sam"
cfg_s = 30.0
cv_fn = "cv_split"
device_ids = [0]
emb_model_cls = "EmbModel"
emb_size = 64
epochs = 1000
gradient_accumulation_steps = 1
is_add_feat_img_ratio = True
is_syncBN = False
load_model_path = "none"
loss_mixup_a = -1
lr = 1e-2
min_lr_ratio = 1e-4
model_cls = "CustomModelMAE"
n_fold = 20
select_data_name = "glr2021|products10k|GPR1200_cutv2|food101"
trn_fold = [0]
warmup_t = 3
weight_decay = 0.1

#is_matplotlib_agg = False
is_matplotlib_agg = True

EMB_DIR = ""
#EMB_DIR = "./output/kqi_3090_ex072_multiinput_emb_arcface_sam_gpGf_ViT-H/embs"
is_make_embs = True

DEBUG = False
#DEBUG = True

if DEBUG:
    NAME = f"tmp_notta2"
else:
    NAME = f"kqi_3090_ex072_multiinput{add_name}"


# In[2]:


# ******************************************************************
# Data
# ******************************************************************
import os, sys
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

INPUT_DIR = f"./data"
OUTPUT_DIR = f"./output/{NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR

if is_matplotlib_agg:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
else:
    from matplotlib import pyplot as plt

# glr2021, products10k
df = pd.read_csv(f"{INPUT_DIR}/preprocess_glr2021_products10k.csv")

# add GPR1200
if "GPR1200_cutv2" in select_data_name:
    df_GPR1200 = pd.read_csv(f"{INPUT_DIR}/preprocess_GPR1200.csv")
    df_GPR1200["label"] = df_GPR1200["label"] + df["label"].max() + 1
    df_GPR1200["category"] = "GPR1200_" + df_GPR1200["category"]
    df_GPR1200 = df_GPR1200[["path", "label", "category"]]
    df = pd.concat([df, df_GPR1200]).reset_index(drop=True)

# add food101
if "food101" in select_data_name:
    df_f101 = pd.read_csv(f"{INPUT_DIR}/preprocess_food101.csv")
    df_f101["label"] = df_f101["label"] + df["label"].max() + 1
    df_f101["category"] = "food101"
    df_f101 = df_f101[["path", "label", "category"]]
    df = pd.concat([df, df_f101]).reset_index(drop=True)

# reset label_id
df["label_orig"] = df["label"].to_numpy()
df["label"] = LabelEncoder().fit_transform(df["label"].to_numpy())


# In[3]:


if DEBUG:
    # Create debug.csv
    if os.path.isfile(f"{OUTPUT_DIR}/train_debug_5fold.csv") == False:
        df_debug = pd.DataFrame()
        np.random.seed(9)
        for i in tqdm( np.random.choice( df.label.unique(), 500) ):
            _df = df[df.label == i].iloc[:n_fold]
            df_debug = df_debug.append(_df, ignore_index=True)
        print("debug df.shape:", df_debug.shape)
        df_debug["label"] = LabelEncoder().fit_transform(df_debug["label"].to_numpy())
        df_debug.to_csv(f"{OUTPUT_DIR}/train_debug_5fold.csv", index=False)


# In[4]:


# ******************************************************************
# Quick EDA
# ******************************************************************
print("label.max:", df.label.max())
print("len(label.unique):", len(df.label.unique()))
print(df.category.value_counts())

if DEBUG:
    # Image Visualization
    n = 30
    _df = df.copy()
    _df = _df.sample(n).reset_index(drop=True)
    fig = plt.figure(figsize=(15,85))
    x=1
    for i in range(n):
        fig.add_subplot(n, 6, x)
        plt.title(_df.loc[i,'label_orig'],fontsize=12)
        plt.imshow(cv2.imread(_df.loc[i,'path'])[:, :, ::-1])  # to RGB
        x+=1
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close()

    # Check for missing labels
    pd.DataFrame(sorted(df['label'].unique())).plot.line(marker="+")
    plt.title("label")
    plt.show()

    v_counts = df["category"].value_counts()
    plt.pie(v_counts.to_numpy(), labels=v_counts.index, autopct="%1.1f%%", pctdistance=0.7)
    plt.title("all category")
    plt.show()


# In[5]:


# ******************************************************************
# CFG
# ******************************************************************
import os, yaml, shutil

class Config:
    def __init__(self):
        self.name=NAME
        self.train=True
        self.debug=DEBUG
        self.print_freq=1000
        self.is_save_per_steps=False
        self.target_col="label"
        self.num_classes=len(df.label.unique())
        self.loss = "DenseCrossEntropy"

        self.emb_model_cls=emb_model_cls
        self.model_cls=model_cls
        self.apex=True
        self.load_model_path=load_model_path
        self.is_load_opt=False

        # ============== model params ==============
        self.model_name=MODEL_NAME
        self.size=SIZE
        self.batch_size=BATCH_SIZE
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.is_syncBN = is_syncBN

        # ArcFace
        self.s=cfg_s
        self.m=cfg_m
        self.easy_margin=False
        self.ls_eps=0.0
        self.emb_size = emb_size
        # ==========================================

        self.num_workers=8
        self.epochs=epochs
        self.optimizer=cfg_optimizer
        self.lr=lr
        self.warmup_lr_init=self.lr*min_lr_ratio
        self.min_lr=self.lr*min_lr_ratio
        self.weight_decay=weight_decay
        self.scheduler='CosineLRScheduler'  # warmup
        self.warmup_t=warmup_t
        self.T_max=epochs
        self.max_grad_norm=1000

        self.cv_fn=cv_fn
        self.cv_col="label"
        self.seeds=[0]
        self.n_fold=n_fold
        self.trn_fold=trn_fold

        self.is_wandb=False
        self.device_ids=device_ids

        # Alpha of mixup with loss. if -1, do not execute loss mixup
        self.loss_mixup_a = loss_mixup_a
        if loss_mixup_a == -1:
            self.mixup_off_epoch = 0
        else:
            self.mixup_off_epoch = self.epochs - 100

        # make embs
        self.is_make_embs=is_make_embs

        if self.model_cls == "CustomModel_LPFT":
            # LP-FT
            self.load_emb_model_path=load_emb_model_path
            self.is_emb_train=False
            self.n_freeze=n_freeze
        else:
            # If you don't use LP-FT load_emb_model_path="", is_emb_train=True
            self.load_emb_model_path=""
            self.is_emb_train=True

        if self.model_name == f"openclip-ViT-L_laion2b":
            # openclip-ViT-L_laion2b only mean, std different
            # https://github.com/mlfoundations/open_clip/blob/4caf23e71c12b54d4e9fb8bf0410e08eb75fe1f6/src/open_clip/pretrained.py#L127
            self.data_mean = [0.5, 0.5, 0.5]
            self.data_std = [0.5, 0.5, 0.5]
        else:
            # openclip mean, std
            # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/constants.py
            self.data_mean = [0.48145466, 0.4578275, 0.40821073]
            self.data_std = [0.26862954, 0.26130258, 0.27577711]

        # Save the model in this epoch unit (used for evaluation of zeroshot after training) If 0, do not save
        if self.model_cls == "CustomModel_LPFT":
            self.save_ep = self.epochs // 5  # Model heavy, so only 5
        else:
            self.save_ep = self.epochs // 100  # Leave 100 for NECK alone, because it's light

        # Also add image width, height, img_ratio features to the input emb
        self.is_add_feat_img_ratio = is_add_feat_img_ratio


CFG = Config()

if CFG.debug:
    #if CFG.is_make_embs:
    #    EMB_DIR == ""
    #else:
    #    EMB_DIR = f"{OUTPUT_DIR}/embs"
    EMB_DIR = f"{OUTPUT_DIR}/embs"
    CFG.is_wandb=False
    CFG.print_freq=10
    CFG.epochs=5
    CFG.save_ep=3
    CFG.mixup_off_epoch=3
    CFG.seeds=[0]
    CFG.trn_fold=[0]
    df = pd.read_csv(f"{OUTPUT_DIR}/train_debug_5fold.csv")
    print("debug df.shape:", df.shape)

with open(OUTPUT_DIR + "/cfg.yaml", "w") as wf:
    yaml.dump(CFG.__dict__, wf)

if len(CFG.device_ids) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(_) for _ in CFG.device_ids])
    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ["CUDA_VISIBLE_DEVICES"])
    CFG.batch_size = CFG.batch_size*len(CFG.device_ids)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CFG.device_ids[0])
    print("os.environ['CUDA_VISIBLE_DEVICES']:", os.environ["CUDA_VISIBLE_DEVICES"])
    # For descriptive error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

print(CFG.__dict__)


# In[6]:


# ******************************************************************
# train-valid, test split
# ******************************************************************
#%reload_ext autoreload
#%autoreload 2
from src.cv_split import cv_split, cv_split_group, cv_split_stratified_group

# Prepare a test set with unknown classes uniformly in every category with StratifiedGroupKFold
_folds = cv_split_stratified_group(df, CFG.seeds[0], CFG.n_fold)

fold = CFG.trn_fold[0]
train_valid_df = _folds[_folds['fold'] != fold].reset_index(drop=True)
test_df        = _folds[_folds['fold'] == fold].reset_index(drop=True)

train_valid_df["label"] = LabelEncoder().fit_transform(train_valid_df["label"].to_numpy())
test_df["label"]        = LabelEncoder().fit_transform(test_df["label"].to_numpy())

test_df.to_csv(OUTPUT_DIR + f'/test_seed{CFG.seeds[0]}.csv', index=False)

# train-valid set
df = train_valid_df.drop(['fold'], axis=1)

# The number of classes used for learning also changes
CFG.num_classes = len(df.label.unique())
with open(OUTPUT_DIR + "/cfg.yaml", "w") as wf:
    yaml.dump(CFG.__dict__, wf)

print("num_classes:", len(df.label.unique()), CFG.num_classes)
df


# In[8]:


# ******************************************************************
# Libraries
# ******************************************************************
import os
import gc
import cv2
import math
import copy
import time
import random
import shutil
import yaml
import glob
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter, OrderedDict
import traceback
import imagesize

# For data manipulation
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

# Utils
import pickle, joblib
from tqdm import tqdm

# Sklearn Imports
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, StratifiedGroupKFold

# For Image Models
import timm
print("timm:", timm.__version__)
from timm.data import ImageDataset
from timm.data.mixup import Mixup, one_hot
from timm.data.auto_augment import rand_augment_transform
from timm.loss.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler  # warmupつきCosineAnnealingLR

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CLIP
import open_clip

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

TORCH_VERSION = '.'.join(torch.__version__.split('.')[:3])
print('torch:', TORCH_VERSION)

#%reload_ext autoreload
#%autoreload 2
from src.sam import SAM, ASAM
from src.arcface_models import ArcMarginProduct, DenseCrossEntropy
from src.knn_faiss import knn_faiss, get_embs, map_per_image, map_per_set


# In[9]:


# ******************************************************************
# Utils
# ******************************************************************
def init_logger(log_file=OUTPUT_DIR + '/train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('> SEEDING DONE')


# In[10]:


# ******************************************************************
# Wandb
# ******************************************************************
if CFG.is_wandb:
    import wandb
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        api_key = user_secrets.get_secret("WANDB")
        wandb.login(key=api_key)
        anonymous = None
    except:
        wandb.login(key="****")  # https://wandb.ai/settings
        wandb.init(project="****", entity='****', name=NAME)
        wandb.config.update(CFG.__dict__)


# In[11]:


# ******************************************************************
# Dataset
# ******************************************************************
class TrainDataset(Dataset):
    def __init__(self, train, transforms=None):
        self.train = train
        self.file_names = train['path'].values
        self.labels = train['label'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.train)

    def __getitem__(self, index):
        img_path = self.file_names[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]

        # subfeature is image information
        H,W,C = img.shape
        subfeature = torch.tensor([
            float(H/W),
            float(H/224.0),
            float(W/224.0),
        ]).float()

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, torch.tensor(label, dtype=torch.long), subfeature


# In[12]:


# ******************************************************************
# Transform
# ******************************************************************
def get_transforms(*, data):
    if data == "train":
        return A.Compose([
            # openclip augment: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transform.py#L59
            A.RandomResizedCrop(height=CFG.size, width=CFG.size,
                                scale=(0.9, 1.0), # openclip scale
                                ratio=(1.0, 1.0), # Avoid destroying the aspect ratio of the original image
                               ),
            A.Normalize(mean=CFG.data_mean, std=CFG.data_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])

    elif data == "valid":
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(mean=CFG.data_mean, std=CFG.data_std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ])


# In[14]:


# ******************************************************************
# CV
# ******************************************************************
def save_folds_csv(df, seed):
    if CFG.cv_fn == "cv_split_stratified_group":
        folds = cv_split_stratified_group(df, seed, n_splits=CFG.n_fold)
    elif CFG.cv_fn == "cv_split":
        folds = cv_split(df, seed, n_splits=CFG.n_fold, cv_col=CFG.cv_col)
    folds.to_csv(OUTPUT_DIR + f'/folds_seed{seed}.csv', index=False)
    return folds


# In[15]:


# ******************************************************************
# Make embs
# ******************************************************************
class EmbModel(nn.Module):
    def __init__(self, is_bk_freeze: bool = True):
        super().__init__()

        if CFG.model_name == f"openclip-ViT-L_laion2b":
            # openclip ViT-L-14-laion2b =======================================
            model_name, param_name = 'ViT-L-14', 'laion2b_s32b_b82k'
            clip_vis_L14_laion2b, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=param_name)
            clip_vis_L14_laion2b = clip_vis_L14_laion2b.visual

            self.model = clip_vis_L14_laion2b
            self.n_feat = self.model.output_dim  # 768

        elif CFG.model_name == f"openclip-ViT-H_laion2b":
            # openclip ViT-H-14-laion2b =======================================
            model_name, param_name = 'ViT-H-14', 'laion2b_s32b_b79k'
            clip_vis_H14_laion2b, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=param_name)
            clip_vis_H14_laion2b = clip_vis_H14_laion2b.visual

            self.model = clip_vis_H14_laion2b
            self.n_feat = self.model.output_dim  # 1024

        elif CFG.model_name == f"openclip-ViT-g_laion2b":
            # openclip ViT-g-14-laion2b =======================================
            model_name, param_name = 'ViT-g-14', 'laion2b_s12b_b42k'
            clip_vis_g14_laion2b, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=param_name)
            clip_vis_g14_laion2b = clip_vis_g14_laion2b.visual

            self.model = clip_vis_g14_laion2b
            self.n_feat = self.model.output_dim  # 1024

        # backborn freeze ==============
        if is_bk_freeze:
            for i, param in enumerate(self.model.parameters()):
                param.requires_grad = False

    def l2_norm(self,x):
        return x / x.norm(p=2, dim=1)[:,None]

    def forward(self, x):
        return self.model(x)


# Export n_feat to CFG for use in training
CFG.n_feat = eval(CFG.emb_model_cls)().n_feat
# Add image information to emb
if CFG.is_add_feat_img_ratio:
    CFG.n_feat = CFG.n_feat + 3
print("n_feat:", CFG.n_feat)


# In[16]:


def make_embs(model, dataloader, dist_path: str):
    """
    model inference and write out the emb to np file
    Args:
        model: pytorch model object
        dataloader: DataLoader
        dist_path: directory path where emb np files are saved
    """
    os.makedirs(dist_path, exist_ok=True)
    emb_buffer = []
    label_buffer = []
    emb_length = 0
    count_npy = 0
    with torch.no_grad():
        for images, targets, asps in tqdm(dataloader):  # for multiinput
            batch_size = images.shape[0]
            emb_length += batch_size
            images = images.to(device)

            emb = model(images)
            # Add image information to emb
            if CFG.is_add_feat_img_ratio:
                asps = asps.to(device)
                emb = torch.cat([emb, asps], dim=1)

            emb_buffer.append(emb.cpu().numpy())
            label_buffer.append(targets.cpu().numpy())
            if emb_length>250_000:
                emb_buffer = np.concatenate(emb_buffer)
                np.save(os.path.join(dist_path, f'emb_{count_npy}.npy'), emb_buffer)
                label_buffer = np.concatenate(label_buffer)
                np.save(os.path.join(dist_path, f'label_{count_npy}.npy'), label_buffer)
                emb_buffer = []
                label_buffer = []
                emb_length = 0
                count_npy += 1

        emb_buffer = np.concatenate(emb_buffer)
        np.save(os.path.join(dist_path, f'emb_{count_npy}.npy'), emb_buffer)
        label_buffer = np.concatenate(label_buffer)
        np.save(os.path.join(dist_path, f'label_{count_npy}.npy'), label_buffer)
        emb_buffer = []
        label_buffer = []
        emb_length = 0
        count_npy += 1


# In[17]:


def embs_main(seed: int = CFG.seeds[0], fold: int = CFG.trn_fold[0]):
    """
    1.model load
    2.CV split
    3.infer and create emb file
    Args:
        seed: random seed for CV split
        fold: valid fold number
    """
    seed_torch(seed)
    # =============================
    # encoder
    # =============================
    m = eval(CFG.emb_model_cls)().to(device)
    if len(CFG.device_ids) > 1:
        m = nn.DataParallel(m, device_ids=CFG.device_ids)  # Data Parallel
    m.eval()

    # =============================
    # CV
    # =============================
    folds = save_folds_csv(df, seed)
    train_df = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_df = folds[folds['fold'] == fold].reset_index(drop=True)
    print(train_df.shape, valid_df.shape)

    # =============================
    # make train emb
    # =============================
    _dataset = TrainDataset(train_df, transforms=get_transforms(data='train'))
    _loader = DataLoader(
        _dataset,
        batch_size=CFG.batch_size*2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    make_embs(m, _loader, dist_path=f"{OUTPUT_DIR}/embs/train")

    # =============================
    # make valid emb
    # =============================
    _dataset = TrainDataset(valid_df, transforms=get_transforms(data='valid'))
    _loader = DataLoader(
        _dataset,
        batch_size=CFG.batch_size*2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    make_embs(m, _loader, dist_path=f"{OUTPUT_DIR}/embs/valid")


# In[18]:


if CFG.is_make_embs and EMB_DIR == "":
    embs_main()
    EMB_DIR = f"{OUTPUT_DIR}/embs"
else:
    # Since save_folds_csv() is executed in embs_main()
    _ = save_folds_csv(df, CFG.seeds[0])


# In[19]:


class EMBDataset(Dataset):
    def __init__(self, emb_npy_list, label_npy_list):
        self.emb_npy = np.concatenate([np.load(pth) for pth in emb_npy_list])
        self.label_npy = np.concatenate([np.load(pth) for pth in label_npy_list])

    def __len__(self):
        return self.emb_npy.shape[0]

    def __getitem__(self, idx):
        return self.emb_npy[idx], self.label_npy[idx]


# In[20]:


# ******************************************************************
# Model
# ******************************************************************
class CustomModelMAE(nn.Module):
    """
    Model with the same neck as MAE's Linear probe
    """
    def __init__(self):
        super().__init__()

        n_feat = CFG.n_feat
        emb_size = CFG.emb_size

        # neck ==============
        # https://github.com/facebookresearch/mae/blob/6a2ba402291005b003a70e99f7c87d1a2c376b0d/main_linprobe.py#L222
        self.neck = torch.nn.Sequential(
            torch.nn.BatchNorm1d(n_feat, affine=False, eps=1e-6),
            nn.Linear(n_feat, emb_size)
        )

        # ArcFace ==============
        self.arc = ArcMarginProduct(emb_size,
                                    CFG.num_classes,
                                    s=CFG.s,
                                    m=CFG.m,
                                    easy_margin=CFG.easy_margin,
                                    ls_eps=CFG.ls_eps)

    def feature(self, x):
        x = self.neck(x)
        return x

    def forward(self, x, labels=None):
        embs = self.feature(x)
        output = self.arc(embs, labels)
        return output


# In[21]:


# https://github.com/sinpcw/kaggle-whale2/blob/master/models.py
def loadpth(pth: str, map_location=None) -> OrderedDict:
    """
    Helper functions for parameter loading
    DataParallel model are saved as module.xxxx, so if the file starts with "module." when loaded, remove it
    """
    ostate = torch.load(pth, map_location=map_location)['model']
    nstate = OrderedDict()
    for k, v in ostate.items():
        if k.startswith('module.'):
            nstate[k[len('module.'):]] = v
        else:
            nstate[k] = v
    return nstate


# In[22]:


# ******************************************************************
# Metric
# ******************************************************************
def get_score(labels: np.ndarray, predictions: np.ndarray):
    return accuracy_score(labels, predictions)


# In[23]:


# ******************************************************************
# Helper
# ******************************************************************
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# In[25]:


def x_mixup(x, y, a: float = 1.0, enable: bool = True):
    a = np.clip(a, 0.0, 1.0)
    if enable and np.random.rand() >= 0.5:
        j = torch.randperm(x.size(0))
        u = x[j]
        z = y[j]
        a = np.random.beta(a, a)
        w = a * x + (1.0 - a) * u
        return w, y, z, a, True
    return x, y, y, 1.0, False

def forward_step(step, losses, batch_size, model, criterion, images, labels, loss_mixup_a, optimizer, scaler, optimizer_step):
    """
    forward processing (loss calculation + backward)
    SAM needs two forwards, so split function
    """
    # SAM 1st/2nd Path
    optimizer.step = optimizer_step

    # ===========================================
    # model forward for loss mixup
    # ===========================================
    if loss_mixup_a != -1:
        fx, t1, t2, a, usemix = x_mixup(images, labels, a=loss_mixup_a, enable=True)
        if CFG.apex:
            with autocast():
                y1 = model(fx, t1)
                y2 = model(fx, t2)
                loss = a * criterion(y1, t1) + (1.0 - a) * criterion(y2, t2)
        else:
            y1 = model(fx, t1)
            y2 = model(fx, t2)
            loss = a * criterion(y1, t1) + (1.0 - a) * criterion(y2, t2)
    else:
        if CFG.apex:
            with autocast():
                outputs = model(images, labels)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images, labels)
            loss = criterion(outputs, labels)

    # ===========================================
    # record loss
    # ===========================================
    losses.update(loss.item(), batch_size)
    if CFG.gradient_accumulation_steps > 1:
        loss = loss / CFG.gradient_accumulation_steps
    if CFG.apex:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

    # ===========================================
    # optimizer step
    # ===========================================
    if (step + 1) % CFG.gradient_accumulation_steps == 0:
        if CFG.apex:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return optimizer, scaler, losses, grad_norm


# In[26]:


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device,
             mixup_fn=None, loss_mixup_a=-1):
    model.train()
    if CFG.apex:
        scaler = GradScaler()
    losses = AverageMeter()

    start = end = time.time()

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        if isinstance(optimizer, SAM) or isinstance(optimizer, ASAM):
            optimizer, scaler, _, _ = forward_step(step, losses, batch_size, model, criterion, images, labels,
                                                   loss_mixup_a, optimizer, scaler,
                                                   optimizer.first_step)
            optimizer, scaler, losses, grad_norm = forward_step(step, losses, batch_size, model, criterion, images, labels,
                                                                loss_mixup_a, optimizer, scaler,
                                                                optimizer.second_step)
        else:
            optimizer, scaler, losses, grad_norm = forward_step(step, losses, batch_size, model, criterion, images, labels,
                                                                loss_mixup_a, optimizer, scaler,
                                                                optimizer.step)

        end = time.time()

        # ===========================================
        # logging
        # ===========================================
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.4e}  '
                  .format(epoch+1, step, len(train_loader),
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=optimizer.param_groups[0]["lr"]))

        if CFG.is_save_per_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                OUTPUT_DIR + "/per_steps.pth",
            )

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg


# In[27]:


@torch.inference_mode()  # pytorch >= 1.9
def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []

    start = end = time.time()

    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # ===========================================
        # compute loss
        # ===========================================
        with torch.no_grad():
            outputs = model(images, labels)  # logit
        loss = criterion(outputs, labels)
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps

        # ===========================================
        # record topk index
        # ===========================================
        _, tk = torch.topk(outputs, 5, dim=1)
        topk_indexs = tk.to('cpu').detach().numpy()
        preds.append(topk_indexs[:,0])  # top1

        end = time.time()

        # ===========================================
        # logging
        # ===========================================
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))

    predictions = np.concatenate(preds)

    torch.cuda.empty_cache()
    gc.collect()

    return losses.avg, predictions


# In[28]:


# ******************************************************************
# Train loop
# ******************************************************************
def train_loop(folds, fold, seed):

    LOGGER.info(f"========== fold: {fold}, seed: {seed} training ==========")

    # ====================================================
    # loader
    # ====================================================
    if CFG.is_emb_train:
        val_idx = folds[folds['fold'] == fold].index

        valid_folds = folds.loc[val_idx].reset_index(drop=True)

        train_embs_npy = sorted(glob.glob(os.path.join(f'{EMB_DIR}/train/','emb_*.npy')))
        train_labels_npy = sorted(glob.glob(os.path.join(f'{EMB_DIR}/train/','label_*.npy')))
        valid_embs_npy = sorted(glob.glob(os.path.join(f'{EMB_DIR}/valid/','emb_*.npy')))
        valid_labels_npy = sorted(glob.glob(os.path.join(f'{EMB_DIR}/valid/','label_*.npy')))

        train_dataset = EMBDataset(emb_npy_list=train_embs_npy, label_npy_list=train_labels_npy)
        valid_dataset = EMBDataset(emb_npy_list=valid_embs_npy, label_npy_list=valid_labels_npy)
        valid_labels = valid_dataset.label_npy  # np.array([10,201,123,...])
    else:
        trn_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[trn_idx].reset_index(drop=True)
        valid_folds = folds.loc[val_idx].reset_index(drop=True)
        valid_labels = valid_folds[CFG.target_col].values

        train_dataset = TrainDataset(train_folds, transforms=get_transforms(data='train'))
        valid_dataset = TrainDataset(valid_folds, transforms=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineLRScheduler':
            # https://blog.shikoan.com/timm-cosine-lr-scheduler/
            scheduler = CosineLRScheduler(optimizer, t_initial=CFG.epochs, lr_min=CFG.min_lr,
                                          warmup_t=CFG.warmup_t, warmup_lr_init=CFG.warmup_lr_init, warmup_prefix=True)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    if CFG.model_cls == "CustomModel_LPFT":
        model = eval(CFG.model_cls)(p=CFG.load_emb_model_path,
                                    n_freeze=CFG.n_freeze)
    else:
        model = eval(CFG.model_cls)()

    if os.path.exists(CFG.load_model_path):
        LOGGER.info("=> loading checkpoint '{}'".format(CFG.load_model_path))
        states = torch.load(CFG.load_model_path, map_location=torch.device("cpu"))
        if len(CFG.device_ids) > 1:
            model.load_state_dict( loadpth(CFG.load_model_path, map_location=torch.device("cpu")) )
        else:
            model.load_state_dict(states["model"])

    if len(CFG.device_ids) > 1:
        if CFG.is_syncBN:
            model = convert_model(model).to(device) # Convert Batch Norm to Sync Batch Norm
            model = DataParallelWithCallback(model, device_ids=CFG.device_ids) # Data Parallel
            LOGGER.info(f"=> train Data Parallel {CFG.device_ids} use SyncBN")
        else:
            model.to(device)
            model = nn.DataParallel(model, device_ids=CFG.device_ids)  # Data Parallel
            LOGGER.info(f"=> train Data Parallel {CFG.device_ids}")
        cudnn.benchmark = True
    else:
        model.to(device)

    if CFG.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=CFG.lr, amsgrad=False, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=CFG.weight_decay)
    elif CFG.optimizer == 'nesterov':
        optimizer = optim.SGD(model.parameters(), lr=CFG.lr, momentum=0.9, weight_decay=CFG.weight_decay, nesterov=True)
    elif CFG.optimizer == 'sam':
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=CFG.lr, weight_decay=CFG.weight_decay, rho=0.05)

    scheduler = get_scheduler(optimizer)

    if os.path.exists(CFG.load_model_path):
        if CFG.is_load_opt:
            LOGGER.info("=> loading optimizer and scheduler")
            optimizer.load_state_dict(states["optimizer"])
            scheduler.load_state_dict(states["scheduler"])

    if CFG.is_wandb:
        wandb.watch(model)

    # ====================================================
    # loop
    # ====================================================
    train_criterion = eval(CFG.loss)().to(device)
    valid_criterion = nn.CrossEntropyLoss().to(device)

    best_score = -np.inf
    best_loss = np.inf

    mixup_fn = None
    loss_mixup_a = CFG.loss_mixup_a

    for epoch in range(CFG.epochs):

        # mixup_off_epoch or higher without mixup
        if epoch >= CFG.mixup_off_epoch:
            mixup_fn = None
            loss_mixup_a = -1
            LOGGER.info(f'Epoch {epoch+1} - loss mixup off')

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model, train_criterion, optimizer, epoch, scheduler, device,
                            mixup_fn, loss_mixup_a)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, valid_criterion, device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, CosineLRScheduler):
            # https://blog.shikoan.com/timm-cosine-lr-scheduler/
            scheduler.step(epoch+1)

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if CFG.is_wandb:
            # save log wandb
            wandb.log({f"[fold{fold}] epoch": epoch+1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score,
                       f"[fold{fold}] lr": optimizer.param_groups[0]["lr"]})

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            best_pth = OUTPUT_DIR + f'/{CFG.name}_fold{fold}_seed{seed}_best_score.pth'
            torch.save({'model': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'preds': preds},
                        best_pth)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            best_pth = OUTPUT_DIR + f'/{CFG.name}_fold{fold}_seed{seed}_best_loss.pth'
            torch.save({'model': model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        'preds': preds},
                        best_pth)
        if ( CFG.save_ep > 0 ) and ( (epoch+1) % CFG.save_ep == 0 ):
            LOGGER.info(f'Epoch {epoch+1} - Save ep{epoch+1}')
            _pth = OUTPUT_DIR + f'/{CFG.name}_fold{fold}_seed{seed}_ep{str(epoch+1)}.pth'
            torch.save({'model': model.state_dict()}, _pth)

    valid_folds['preds'] = torch.load(OUTPUT_DIR + f'/{CFG.name}_fold{fold}_seed{seed}_best_loss.pth',
                                      map_location=torch.device('cpu'))['preds'].tolist()

    return valid_folds


# In[29]:


# ******************************************************************
# Main
# ******************************************************************
def get_result(result_df, preds_col="preds", add_info=""):
    labels = result_df[CFG.target_col].values
    preds = result_df[preds_col].values
    preds = np.array([np.array(p) for p in preds])
    score = get_score(labels, preds)
    LOGGER.info(f'{add_info}Score: {score:<.4f}')
    return score


def main():
    if CFG.train:
        for seed in CFG.seeds:
            seed_torch(seed)

            # CV
            folds = pd.read_csv(OUTPUT_DIR + f'/folds_seed{seed}.csv')
            LOGGER.info(f"folds.shape: {str(folds.shape)}")

            # train fold
            oof_df = pd.DataFrame()
            for fold in range(CFG.n_fold):
                if fold in CFG.trn_fold:
                    _oof_df = train_loop(folds, fold, seed)
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} seed: {seed} result ==========")
                    _ = get_result(_oof_df, add_info=f"fold{fold} ")

                    gc.collect()
                    torch.cuda.empty_cache()

            # CV result
            LOGGER.info(f"========== CV ==========")
            oof_score = get_result(oof_df, add_info=f"oof ")
            # save result
            oof_df.to_csv(OUTPUT_DIR + f'/{CFG.name}_oof_df_seed{seed}.csv', index=False)


# In[30]:


if __name__ == '__main__':
    main()

#if CFG.epochs > 10:
#    from IPython.display import clear_output
#    clear_output()


# In[31]:


# ******************************************************************
# knn test
# ******************************************************************
def get_embs_multiinput(p, loader, device, n_limit: int = -1, is_tf: bool = False):
    """
    Vertical stacking of emb and label by inferring loader with model
    Args:
        loader: DataLoader
        device: pytorch device（"cuda"/"cuda:0"/"cpu"）
        n_limit: number for censoring inference up to specific data in loader. If -1, no termination
        is_tf: flag to indicate whether the loader is TF data or not, must be True when using tfrecord's loader
    """
    emb_m = eval(CFG.emb_model_cls)().to(device)
    emb_m.eval()
    m = CustomModelMAE()
    if len(CFG.device_ids) > 1:
        m.load_state_dict( loadpth(p, map_location=torch.device("cpu")) )
    else:
        states = torch.load(p, map_location=torch.device("cpu"))
        m.load_state_dict(states["model"])
    m = m.to(device)
    m.eval()

    embs = None
    targets = []
    for step, (images, labels, asps) in tqdm(enumerate(loader)):
        targets.extend(labels.reshape(-1).tolist())

        if is_tf:
            # TFRecordDataLoader
            images = torch.from_numpy(images.transpose(0, 3, 1, 2)).to(device)
        else:
            # pytorch DataLoader
            images = images.to(device)

        with torch.no_grad():
            if is_tf:
                # TFRecordDataLoader
                emb = model.feature(images)
            else:
                # pytorch DataLoader
                emb = emb_m(images)

            # Add image information to emb
            asps = asps.to(device)
            emb = torch.cat([emb, asps], dim=1)

            emb = m.feature(emb)
            emb = emb.to('cpu')

        if embs is None:
            embs = emb
        else:
            embs = torch.cat((embs, emb), 0)
        if (n_limit != -1) and (len(targets) >= n_limit):
            print(f"{len(targets)} >= {n_limit}")
            break
    print("embs.shape:", embs.shape)
    return embs, targets


# In[32]:


def knn_test_main(model, device,
                  query_csv: str = OUTPUT_DIR + f'/test_seed{CFG.seeds[0]}.csv',
                  db_csv: str = OUTPUT_DIR + f'/folds_seed{CFG.seeds[0]}.csv',
                  is_show: bool = True,
                 ):
    """
    1.Prepare db and query data from test set
    2.Calculate db and query emb with the loaded model
    3.Search for db data close to query with knn from calculated emb
    Args:
        model: weight-loaded pytorch model
        device: pytorch device（"cuda"/"cuda:0"/"cpu"）
        query_csv: csv for knn query
        db_csv: csv for knn db
        is_show: visualization flag for the percentage of categories in db,query
    """
    # ====================================================
    # Data
    # ====================================================
    # use unknown classes not used in train
    db_query_df = pd.read_csv(query_csv)
    print("db_query_df.shape:", db_query_df.shape)

    # ===============================
    # Make db:query = 1+3:1, where 3 in db is a class unrelated to query
    # ===============================
    # query without duplicate elements in the label (so that each class has one data item)
    query_df = db_query_df[~db_query_df["label"].duplicated()].reset_index(drop=True)

    # db is created by taking duplicate elements of label and removing duplicate elements from it (so that each class has one data item from each class except query)
    db_df = db_query_df[db_query_df["label"].duplicated()].reset_index(drop=True)
    db_df = db_df[~db_df["label"].duplicated()].reset_index(drop=True)

    # The rest of the db should be train-valid class data
    folds = pd.read_csv(db_csv)
    folds["category"] = "train_" + folds["category"]
    _df = folds[~folds["label"].duplicated()].reset_index(drop=True)
    _df = _df.sample(n=len(db_df)*3, random_state=0)
    _df['label'] = _df['label'] + query_df["label"].max() + 1
    db_df = pd.concat([db_df, _df], ignore_index=True)

    # ===============================
    # upper limit of db,query
    # ===============================
    n_db     = 20_000  # Number of db for GUIE, upper limit of db
    n_query  = 5_000   # Number of query for GUIE, upper limit of query
    db_df    = db_df.sort_values(by='label').iloc[0:n_db,:].reset_index(drop=True)
    query_df = query_df.sort_values(by='label').iloc[0:n_query,:].reset_index(drop=True)
    print("db_df, query_df:", db_df.shape, query_df.shape)

    # ===============================
    # percentage visualization of categories in db,query
    # ===============================
    if is_show:
        db_v_counts = pd.DataFrame(db_df["category"].value_counts()).reset_index()
        qu_v_counts = pd.DataFrame(query_df["category"].value_counts()).reset_index()
        v_counts = pd.merge(db_v_counts, qu_v_counts, on="index", how="outer").fillna(0)
        v_counts.columns = ["category", "db_count", "qu_count"]
        cmap = plt.get_cmap("Paired")
        v_counts["color"] = [cmap(i) for i in range(len(v_counts))]
        plt.figure(figsize=(10,10))
        plt.subplot(1,2,1)
        plt.pie(v_counts["qu_count"].to_numpy(), labels=v_counts["category"].to_numpy(),
                autopct="%1.1f%%", pctdistance=0.7, colors=v_counts["color"].to_numpy())
        plt.title("query category")
        plt.subplot(1,2,2)
        plt.pie(v_counts["db_count"].to_numpy(), labels=v_counts["category"].to_numpy(),
                autopct="%1.1f%%", pctdistance=0.7, colors=v_counts["color"].to_numpy())
        plt.title("db category")
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close()

    # ====================================================
    # loader
    # ====================================================
    db_dataset = TrainDataset(db_df, transforms=get_transforms(data='valid'))
    db_loader = DataLoader(
        db_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
    )

    query_dataset = TrainDataset(query_df, transforms=get_transforms(data='valid'))
    query_loader = DataLoader(
        query_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
    )

    # ====================================================
    # emb
    # ====================================================
    if CFG.is_add_feat_img_ratio:
        db_embs, db_labels = get_embs_multiinput(p, db_loader, device, n_limit=n_db)
        query_embs, query_labels = get_embs_multiinput(p, query_loader, device, n_limit=n_query)
    else:
        db_embs, db_labels = get_embs(model, db_loader, device, n_limit=n_db)
        query_embs, query_labels = get_embs(model, query_loader, device, n_limit=n_query)

    # ====================================================
    # knn
    # ====================================================
    _, knn_target = knn_faiss(db_embs.detach().numpy().copy(),
                              np.array(db_labels),
                              query_embs.detach().numpy().copy(),
                              #is_gpu=True,
                              is_gpu=False,
                             )
    knn_map5 = map_per_set(query_labels, knn_target.tolist(), k=5)
    print(knn_target.shape)
    print(knn_target[:5])
    LOGGER.info(f"db_embs, query_embs: {str(db_embs.shape)}, {str(query_embs.shape)}")
    LOGGER.info(f'MAP5: {knn_map5:.4f}')
    if CFG.is_wandb:
        wandb.log({f"[test]MAP5": knn_map5})
    query_df = pd.concat([query_df, pd.DataFrame(knn_target, columns=[f"pred{i}" for i in range(5)])], axis=1)
    return query_df


# In[33]:


LOGGER.info(f"========== knn test ==========")

for p in sorted(glob.glob(f"{OUTPUT_DIR}/*.pth")):
    LOGGER.info(f"load pth: {p}")
    query_df = knn_test_main(p, device)
    query_df.to_csv(f"{OUTPUT_DIR}/{Path(p).stem.split('_')[-1]}_pred_test.csv", index=False)

    #display(query_df)
    print("="*120)

#if len(glob.glob(f"{OUTPUT_DIR}/*.pth")) > 10:
#    from IPython.display import clear_output
#    clear_output()


# In[34]:


if CFG.is_wandb:
    wandb.finish()
