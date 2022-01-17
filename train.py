#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
import random
from scipy.ndimage import distance_transform_edt as distance

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
from monai.losses import FocalLoss
from monai.metrics import *
from monai.utils import set_determinism

from model import Encoder, Decoder

set_determinism(seed=2021)


def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation or ground truth, shape = (batch_size, c, x, y, z)
    out_shape = (batch_size, c, x, y, z)
    output: the foreground Distance Map (SDM) shape = (batch_size,c, x, y, z)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]):
        for c in range(1, out_shape[1]):
            posmask = img_gt[b][c].astype(bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm


def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for multiclass segmentation
    input: seg_soft: softmax results, shape=(b,c,x,y,z)
           gt: ground truth, shape=(b,c,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,c,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,c,x,y,z)
    output: boundary_loss; sclar
    """
    delta_s = (seg_soft - gt.float()) ** 2
    s_dtm = seg_dtm ** 2
    g_dtm = gt_dtm ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bcxyz, bcxyz->bcxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss


os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# get image address
root_dir = '../data/brainstruct'
data_dir = os.path.join(root_dir, "aseg_crop")
all_filenames = glob(data_dir + '/*.nii.gz')
random.shuffle(all_filenames)
test_frac = 0.3
num_ims = len(all_filenames)
num_test = int(num_ims * test_frac)
num_train = num_ims - num_test
train_datadict = [{"im": fname, 'seg': fname} for fname in all_filenames[:num_train]]
val_datadict = [{"im": fname, 'seg': fname} for fname in all_filenames[-num_test:]]
print(f"total number of images: {num_ims}")
print(f"number of images for training: {len(train_datadict)}")
print(f"number of images for testing: {len(val_datadict)}")

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

train_ds = CacheDataset(train_datadict, transforms, num_workers=num_workers)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_ds = CacheDataset(val_datadict, transforms, num_workers=num_workers)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

encoder = Encoder(z=354).cuda()
decoder = Decoder(z=354).cuda()

num_epochs = 500
lr = 1e-4

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

criterion = FocalLoss(include_background=True, to_onehot_y=True)
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-5)

save_checkpoints_dir = './checkpoints/z354_05focal05hdloss'
if not os.path.exists(save_checkpoints_dir):
    os.makedirs(save_checkpoints_dir)

val_interval = 10
best_loss = 100.0
train_losses = []
test_losses = []

dice_metric = DiceMetric(include_background=True, reduction='mean')

for epoch in range(num_epochs):
    val_loss = []
    train_loss = []
    dices = torch.zeros(1, 5).to(device)
    encoder.train()
    decoder.train()
    alpha = 0.5

    for data in train_loader:
        img = data['im'].to(device)
        target = data['seg'].to(device)
        target_oh = one_hot(target, 5)
        # ===================forward=====================
        encoded_data = encoder(img)
        output = decoder(encoded_data)

        loss_focal = criterion(output, target)
        with torch.no_grad():
            gt_dtm_npy = compute_dtm(target_oh.cpu().numpy(), output.shape)
            gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(output.device.index)

            output_arg = torch.argmax(output, dim=1, keepdim=True)
            output_oh = one_hot(output_arg, 5)
            seg_dtm_npy = compute_dtm(output_oh.cpu().numpy(), output.shape)
            seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(output.device.index)

        loss_hd = hd_loss(output, target_oh, seg_dtm, gt_dtm)
        loss = alpha * (loss_focal) + (1 - alpha) * loss_hd
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================metric========================
        dice = dice_metric(output_oh, target_oh)
        dices = torch.cat((dices, dice), dim=0)

        # ===================log========================
        train_loss.append(loss.detach().cpu().numpy())

    dice_score = torch.mean(dices[1:, ...], dim=0).cpu().numpy().tolist()

    if (epoch + 1) % val_interval == 0:
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_img = val_data['im'].to(device)
                val_target = val_data['seg'].to(device)
                val_target_oh = one_hot(val_target, 5)

                # ===================forward=====================
                val_output = decoder(encoder(val_img))

                val_loss_focal = criterion(val_output, val_target)

                val_gt_dtm_npy = compute_dtm(val_target_oh.cpu().numpy(), val_output.shape)
                val_gt_dtm = torch.from_numpy(val_gt_dtm_npy).float().cuda(val_output.device.index)

                val_output_ = torch.argmax(val_output, dim=1, keepdim=True)
                val_output_ = one_hot(val_output_, 5)
                val_seg_dtm_npy = compute_dtm(val_output_.cpu().numpy(), val_output.shape)
                val_seg_dtm = torch.from_numpy(val_seg_dtm_npy).float().cuda(val_output.device.index)

                val_loss_hd = hd_loss(val_output, val_target, val_seg_dtm, val_gt_dtm)
                val_loss = alpha * (val_loss_focal) + (1 - alpha) * val_loss_hd
                # ===================log========================
                val_loss.append(val_loss.detach().cpu().numpy())

    train_loss_one = np.mean(train_loss)
    val_loss_one = np.mean(val_loss)

    train_losses.append(train_loss_one)
    test_losses.append(val_loss_one)

    if (epoch + 1) % 50 == 0 or (epoch + 1) == num_epochs:
        torch.save({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'loss': train_loss_one,
        }, save_checkpoints_dir + f'/encoder_{epoch + 1}.pth')

        torch.save({
            'epoch': epoch + 1,
            'decoder': decoder.state_dict(),
            'loss': train_loss_one,
        }, save_checkpoints_dir + f'/decoder_{epoch + 1}.pth')

    if val_loss_one < best_loss:
        best_loss = val_loss_one
        # save_model(model)
        torch.save({
            'epoch': epoch + 1,
            'encoder': encoder.state_dict(),
            'loss': val_loss_one,
        }, save_checkpoints_dir + f'/encoder.pth')

        torch.save({
            'epoch': epoch + 1,
            'decoder': decoder.state_dict(),
            'loss': val_loss_one,
        }, save_checkpoints_dir + f'/decoder.pth')

    print('epoch [{}/{}], train loss:{:.4f}, val loss:{:.4f}'.format(epoch + 1, num_epochs, train_loss_one,
                                                                     val_loss_one))
    print('train dice:{}'.format(dice_score))


