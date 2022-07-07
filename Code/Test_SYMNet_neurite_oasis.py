import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import SYMNet,SpatialTransform, DiffeomorphicTransform, CompositionTransform
from Functions import generate_grid,save_img,save_flow, load_4D_with_header, imgnorm


parser = ArgumentParser()
parser.add_argument("--modelpath", type=str,
                    dest="modelpath", default='../Model/SYMNet_neurite_oasis_smo30_update_80000.pth',
                    help="frequency of saving models")
parser.add_argument("--savepath", type=str,
                    dest="savepath", default='../Result',
                    help="path for saving images")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=7,
                    help="number of start channels")
parser.add_argument("--fixed", type=str,
                    dest="fixed", default='../Data/image_A_full_size.nii.gz',
                    help="fixed image")
parser.add_argument("--moving", type=str,
                    dest="moving", default='../Data/image_B_full_size.nii.gz',
                    help="moving image")
opt = parser.parse_args()

savepath = opt.savepath
fixed_path = opt.fixed
moving_path = opt.moving

if not os.path.isdir(savepath):
    os.mkdir(savepath)


def test():
    model = SYMNet(2, 3, opt.start_channel).cuda()
    transform = SpatialTransform().cuda()

    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    com_transform = CompositionTransform().cuda()

    model.load_state_dict(torch.load(opt.modelpath))
    model.eval()
    transform.eval()
    diff_transform.eval()
    com_transform.eval()

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).cuda().float()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    fixed_img, fixed_header, fixed_affine = load_4D_with_header(fixed_path)
    moved_img, moved_header, moved_affine = load_4D_with_header(moving_path)

    norm = True
    if norm:
        fixed_img = imgnorm(fixed_img)
        moved_img = imgnorm(moved_img)

    fixed_img = torch.from_numpy(fixed_img).float().to(device).unsqueeze(dim=0)
    moved_img = torch.from_numpy(moved_img).float().to(device).unsqueeze(dim=0)

    with torch.no_grad():
        F_xy, F_yx = model(fixed_img, moved_img)

        F_X_Y_half = diff_transform(F_xy, grid, range_flow)
        F_Y_X_half = diff_transform(F_yx, grid, range_flow)

        F_X_Y_half_inv = diff_transform(-F_xy, grid, range_flow)
        F_Y_X_half_inv = diff_transform(-F_yx, grid, range_flow)

        F_X_Y = com_transform(F_X_Y_half, F_Y_X_half_inv, grid, range_flow)
        F_Y_X = com_transform(F_Y_X_half, F_X_Y_half_inv, grid, range_flow)

        F_BA = F_Y_X.permute(0, 2, 3, 4, 1).data.cpu().numpy()[0, :, :, :, :]
        F_BA = F_BA.astype(np.float32) * range_flow
        
        F_AB = F_X_Y.permute(0, 2, 3, 4, 1).data.cpu().numpy()[0, :, :, :, :]
        F_AB = F_AB.astype(np.float32) * range_flow
        
        warped_B = transform(moved_img, F_Y_X.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]
        warped_A = transform(fixed_img, F_X_Y.permute(0, 2, 3, 4, 1) * range_flow, grid).data.cpu().numpy()[0, 0, :, :, :]

        save_flow(F_BA, savepath + '/wrapped_flow_B_to_A_full_size.nii.gz')
        save_flow(F_AB, savepath + '/wrapped_flow_A_to_B_full_size.nii.gz')

        save_img(warped_B, savepath + '/wrapped_norm_B_to_A_full_size.nii.gz', header=moved_header, affine=moved_affine)
        save_img(warped_A, savepath + '/wrapped_norm_A_to_B_full_size.nii.gz', header=fixed_header, affine=fixed_affine)
        
        print("Finished.")
    


if __name__ == '__main__':
    imgshape = (160, 192, 224)
    range_flow = 100
    test()