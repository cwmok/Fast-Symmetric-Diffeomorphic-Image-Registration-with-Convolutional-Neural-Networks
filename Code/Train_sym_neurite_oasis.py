import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from Models import SYMNet, SpatialTransform, SpatialTransformNearest, smoothloss, DiffeomorphicTransform, \
    CompositionTransform, magnitude_loss, neg_Jdet_loss, NCC
from Functions import generate_grid, Dataset_epoch_crop, Predict_dataset_crop
import torch.utils.data as Data

import matplotlib.pyplot as plt


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=150001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=100.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=3.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='../Data/OASIS/neurite-oasis.v1.0',
                    help="data path for training images")
opt = parser.parse_args()

lr = opt.lr
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath

# Create and initalize log file
if not os.path.isdir("../Log"):
    os.mkdir("../Log")

log_dir = "../Log/SYMNet_neurite_oasis_old.txt"

with open(log_dir, "w") as log:
    log.write("Validation Dice log for SYMNet_neurite_oasis:\n")


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1==i).sum()==0) or ((atlas==i).sum()==0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
    return dice/num_count


def train():
    model = SYMNet(2, 3, start_channel).cuda()
    loss_similarity = NCC(win=5)
    loss_smooth = smoothloss
    loss_magnitude = magnitude_loss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().cuda()
    transform_nearest = SpatialTransformNearest().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    com_transform = CompositionTransform().cuda()

    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[0:255]
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((6, iteration+1))

    training_generator = Data.DataLoader(Dataset_epoch_crop(names, norm=True), batch_size=1,
                                         shuffle=False, num_workers=2)
    step = 0

    while step <= iteration:
        for X, Y in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            F_xy, F_yx = model(X, Y)

            F_X_Y_half = diff_transform(F_xy, grid, range_flow)
            F_Y_X_half = diff_transform(F_yx, grid, range_flow)

            F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
            F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

            X_Y_half = transform(X, F_X_Y_half.permute(0, 2, 3, 4, 1) * range_flow, grid)
            Y_X_half = transform(Y, F_Y_X_half.permute(0, 2, 3, 4, 1) * range_flow, grid)

            F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
            F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

            X_Y = transform(X, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid)
            Y_X = transform(Y, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid)

            loss1 = loss_similarity(X_Y_half, Y_X_half)
            loss2 = loss_similarity(Y, X_Y) + loss_similarity(X, Y_X)
            loss3 = loss_magnitude(F_X_Y_half*range_flow, F_Y_X_half*range_flow)
            loss4 = loss_Jdet(F_X_Y.permute(0,2,3,4,1)*range_flow, grid) + loss_Jdet(F_Y_X.permute(0,2,3,4,1)*range_flow, grid)
            loss5 = loss_smooth(F_xy*range_flow) + loss_smooth(F_yx*range_flow)

            loss = loss1 + loss2 + magnitude * loss3 + local_ori * loss4 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step] = np.array([loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()])
            # sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_mid "{2:.4f}" - sim_full "{3:4f}" - mag "{4:.4f}" - Jdet "{5:.10f}" -smo "{6:.4f}" '.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))
            # sys.stdout.flush()
            print("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_mid "{2:.4f}" - sim_full "{3:4f}" - mag "{4:.4f}" - Jdet "{5:.10f}" -smo "{6:.4f}" '.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))


            if (step % n_checkpoint == 0):
                modelname = model_dir + '/SYMNet_neurite_oasis_smo30_update_' + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss_SYMNet_neurite_oasis_smo30_update_' + str(step) + '.npy', lossall)

                # Validation
                fixed_img = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[255]
                fixed_label = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))[255]
                imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))[256:261]
                labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/aligned_seg35.nii.gz'))[256:261]

                valid_generator = Data.DataLoader(Predict_dataset_crop(fixed_img, imgs, fixed_label, labels, norm=True),
                                                  batch_size=1,
                                                  shuffle=False, num_workers=2)

                use_cuda = True
                device = torch.device("cuda" if use_cuda else "cpu")
                dice_total = []
                print("\nValiding...")
                for batch_idx, data in enumerate(valid_generator):
                    X, Y, X_label, Y_label = data['move'].to(device), data['fixed'].to(device), data['move_label'].to(
                        device), data['fixed_label'].to(device)

                    with torch.no_grad():
                        F_xy, F_yx = model(X, Y)

                        F_X_Y_half = diff_transform(F_xy, grid, range_flow)
                        # F_Y_X_half = diff_transform(F_yx, grid, range_flow)
                        #
                        # F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
                        F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

                        F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)

                        X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]
                        Y_label = Y_label.data.cpu().numpy()[0, 0, :, :, :]

                        dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                        dice_total.append(dice_score)

                dice_total = np.array(dice_total)
                print("Dice mean: ", dice_total.mean())
                with open(log_dir, "a") as log:
                    log.write(str(step)+":"+str(dice_total.mean()) + "\n")
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss_SYMNet_neurite_oasis_update.npy', lossall)


imgshape = (160, 144, 192)

range_flow = 100
print('lr', lr, ' local_ori', local_ori, ' magnitude', magnitude, ' smooth', smooth, 'start_channel', start_channel, 'range_flow', range_flow)
train()
