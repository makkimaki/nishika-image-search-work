import argparse 
import os 
import random 
import warnings 
from pathlib import Path 
from typing import Any, Dict, List, Optional

import pandas as pd 
import numpy as np 
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset
import torch.utils.data.distributed
import torchvision.transforms as transforms

import cv2
from tqdm import tqdm 


warnings.simplefilter('ignore', UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", "-tdir", type=str, metavar="PATH")
parser.add_argument("--epoch", "-ep", type=int, metavar="N")
parser.add_argument("--batch_size", "-bs", type=int, metavar="N")
parser.add_argument("--image_size", "-size", type=int, metaver="SIZE")


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class ISNet(nn.Module):
    def __init__(self, backbone, fc_dim=256, p=3.0, eval_p=4.0) -> None:
        super(ISNet, self).__init__()
        self.backbone = backbone
        print(f"Build with the model: {self.backbone.__class__.__name__}")
        self.fc = nn.Linear()
        self.bn = nn.BatchNorm2d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p
        
    def _init_params(self):
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        p = self.p if self.training else self.eval_p
        x = gem(x, p).view(batch_size, -1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x)
        return x

class ISTrainDataset(Dataset):
    def __init__(self,
                  csv,
                  transforms
                  ) -> None:
        self.csv = csv.reset_index()
        self.augmentations = transforms
    
    def __len__(self):
        return self.csv.shape[0]
    
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        image = cv2.imread(row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        return image, torch.tensor(row.cite_gid)

    

def train_loop(dataloader, model, loss_fn, optimizer, epoch, scheduler):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>4f} [{current:>5d}/{size:5d}]]")


loss_fn = 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
epochs = 

for idx in range(epochs):
    model.train()
    loss_fn = 


if __name__ == "__main__":
    args = parser.parse_args()
    train_model(args)





