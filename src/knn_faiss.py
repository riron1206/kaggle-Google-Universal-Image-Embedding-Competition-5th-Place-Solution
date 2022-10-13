import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import faiss
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

def knn_faiss(train_embs: np.ndarray, train_ys: np.ndarray, test_embs: np.ndarray, k:int = 5, is_gpu: bool = True):
    print("running faiss...")

    D = len(train_embs[0])

    cpu_index = faiss.IndexFlatL2(D)

    if is_gpu:
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index.add(train_embs)
        cluster_distance, cluster_index = gpu_index.search(x=test_embs, k=k)
    else:
        cpu_index.add(train_embs)
        cluster_distance, cluster_index = cpu_index.search(x=test_embs, k=k)

    knn_target = train_ys[cluster_index]

    return cluster_distance, knn_target

def get_embs(model, loader, device, n_limit: int = -1, is_tf: bool = False):
    model.eval()
    embs = None
    targets = []
    for step, (images, labels) in tqdm(enumerate(loader)):
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
                emb = model(images)

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


# ========================================================================
# Metric
# ========================================================================
# https://www.kaggle.com/pestipeti/explanation-of-map5-scoring-metric
def map_per_image(label, predictions, k=5):
    try:
        return 1 / (predictions[:k].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions, k=5):
    return np.mean([map_per_image(l, p, k) for l,p in zip(labels, predictions)])