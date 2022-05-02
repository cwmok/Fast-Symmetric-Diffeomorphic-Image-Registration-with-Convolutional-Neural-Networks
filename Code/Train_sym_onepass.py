import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
from Models import SYMNet, SpatialTransform, smoothloss, DiffeomorphicTransform, \
    CompositionTransform, magnitude_loss, neg_Jdet_loss, NCC
from Functions import generate_grid, Dataset_epoch
import torch.utils.data as Data


parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=140001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=100.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.1,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smooth", type=float,
                    dest="smooth", default=3.0,
                    help="Gradient smooth loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=10000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/PATH/TO/YOUR/DATA',
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


def train():
    model = SYMNet(2, 3, start_channel).cuda()
    loss_similarity = NCC()
    loss_smooth = smoothloss
    loss_magnitude = magnitude_loss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    com_transform = CompositionTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    names = sorted(glob.glob(datapath + '/*.nii'))
    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((6, iteration+1))

    training_generator = Data.DataLoader(Dataset_epoch(names, norm=False), batch_size=1,
                                         shuffle=True, num_workers=2)
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
            sys.stdout.write("\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_mid "{2:.4f}" - sim_full "{3:4f}" - mag "{4:.4f}" - Jdet "{5:.10f}" -smo "{6:.4f}" '.format(step, loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                modelname = model_dir + '/SYMNet_' + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss_SYMNet_' + str(step) + '.npy', lossall)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss_SYMNet.npy', lossall)


imgshape = (160, 192, 144)
range_flow = 100
train()
