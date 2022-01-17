#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import numpy as np
import random
import pandas as pd

import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChannelD,
    Compose,
    LoadImageD,
    Resized,
    ScaleIntensityD,
    EnsureTypeD,
)
from monai.networks import one_hot
from monai.metrics import *
from monai.utils import set_determinism

from model import Encoder, Decoder

set_determinism(seed=2021)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

root_dir = '../data/brainstruct'
data_dir = os.path.join(root_dir, "test")
all_filenames = glob(data_dir + '/*.nii.gz')
random.shuffle(all_filenames)
test_datadict = [{"im": fname, 'seg': fname} for fname in all_filenames]

batch_size = 32
num_workers = 12

transforms = Compose(
    [
        LoadImageD(keys=["im", 'seg']),
        AddChannelD(keys=["im", 'seg']),
        Resized(keys=['im', 'seg'], spatial_size=[96, 128, 96], mode='nearest'),
        ScaleIntensityD(keys=["im"]),
        EnsureTypeD(keys=["im", 'seg']),
    ]
)
test_ds = CacheDataset(test_datadict, transforms, num_workers=num_workers)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

dice_metric = DiceMetric(include_background=True, reduction='mean')
hd95_metric = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95)
mse_metric = MSEMetric()

cproot = './checkpoints/z354_05focal05hdloss'
encoder = Encoder(z=354).cuda()
decoder = Decoder(z=354).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint1 = torch.load(cproot + '/encoder.pth', map_location=device)
checkpoint2 = torch.load(cproot + '/decoder.pth', map_location=device)
encoder.load_state_dict(checkpoint1["encoder"])
decoder.load_state_dict(checkpoint2["decoder"])
encoder.eval()
decoder.eval()
print('load model')

dices = torch.zeros(1, 5).to(device)
hd95s = torch.zeros(1, 5)
mses = torch.zeros(1, 1).to(device)
zs = torch.zeros(1, 354).to(device)
imgnames = []

for data in test_loader:
    with torch.no_grad():
        img = data['im'].to(device)
        imgname = data['im_meta_dict']['filename_or_obj']
        target = data['seg'].to(device)
        target_oh = one_hot(target, 5)
        # ===================forward=====================
        encoded_data = encoder(img)
        output = decoder(encoded_data)

        output_arg = torch.argmax(output, dim=1, keepdim=True)
        output_oh = one_hot(output_arg, 5)

        # ===================metric========================
        dice = dice_metric(output_oh, target_oh)  # [batch, c]
        hd95 = hd95_metric(output_oh, target_oh)
        mse = mse_metric(output_arg, target)
        z = encoded_data.detach()

        dices = torch.cat((dices, dice), dim=0)
        hd95s = torch.cat((hd95s, hd95), dim=0)
        mses = torch.cat((mses, mse), dim=0)
        zs = torch.cat((zs, z), dim=0)
        imgnames.append(imgname)


# ===================save latent space========================
savez = zs[1:]
savez = savez.detach().cpu().numpy()

savename = []
for name in imgnames:
    for idx in name:
        n = idx.split('/')[-1]
        savename.append(n)

savez_df = pd.DataFrame(savez)
savez_df['name'] = savename
savez_df.to_csv('./feature_csv/z354_Feat.csv', index=False)

# ===================metric========================
# cal metric per class
dice_score0 = dices[1:, 0].cpu().numpy()
dice_score1 = dices[1:, 1].cpu().numpy()
dice_score2 = dices[1:, 2].cpu().numpy()
dice_score3 = dices[1:, 3].cpu().numpy()
dice_score4 = dices[1:, 4].cpu().numpy()

hd95s_score0 = hd95s[1:, 0]
hd95s_score1 = hd95s[1:, 1]
hd95s_score2 = hd95s[1:, 2]
hd95s_score3 = hd95s[1:, 3]
hd95s_score4 = hd95s[1:, 4]

mses_score = mses[1:, ...].cpu().numpy().squeeze()

data = {'name': savename, 'dice0': dice_score0, 'dice1': dice_score1,'dice2': dice_score2,'dice3': dice_score3,'dice4': dice_score4,
        'hd950': hd95s_score0, 'hd951': hd95s_score1, 'hd952': hd95s_score2, 'hd953': hd95s_score3, 'hd954': hd95s_score4,
        'mse': mses_score}

data_df = pd.DataFrame(data)
data_df.to_csv('./feature_csv/metric.csv', index=False)