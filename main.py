'''
FEATURE-BASED 3D PD-OCT IMAGE REGISTRATION
v1.0
Tiffany Tse
updated 11/25/2024

This code takes in 2D en-face projections to generate a feature map using SuperPoint.
Once features are identified and matched between fixed (reference) and moving (target) images using LightGlue,
the transformation is applied to the full 3D volume using thin plate spline interpolation.

Please see README for more information.

SuperPoint documentation:
https://github.com/rpautrat/SuperPoint
https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf

LightGlue documentation:
https://github.com/cvg/LightGlue
https://openaccess.thecvf.com/content/ICCV2023/papers/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.pdf

This version is built on Eyeliner:
https://github.com/QTIM-Lab/EyeLiner
'''

import cv2
import hdf5storage
import torch
import numpy as np
import h5py
import tifffile as tiff
import time

from os.path import join
from lightglue import viz2d
import matplotlib.pyplot as plt
from src.utils import load_image, load_volume
from src.eyeliner import EyeLinerP


def overlay_image_2d(reg, fixed, fname):
    img = cv2.addWeighted(fixed, 0.5, reg, 0.5, 0)[0]
    # img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    tiff.imwrite(join(output_dir,fname), img)

def overlay_image_3d(reg, fixed, fname):
    img = cv2.addWeighted(fixed, 0.5, reg, 0.5, 0)
    tiff.imwrite(join(output_dir,fname), img)

def visualize_key_points(kpts0, img0, kpts1, img1, fname):
    axes = viz2d.plot_images([img0, img1])
    viz2d.plot_matches(kpts0, kpts1, color="lime", lw=0.2)
    viz2d.save_plot(join(output_dir, f'{fname}_matches'))

    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([img0, img1])
    viz2d.plot_keypoints([kpts0, kpts1], ps=6) #, colors=[kpc0, kpc1], ps=6)
    viz2d.save_plot(join(output_dir, fname))

def main():
    start_time = time.time()
    print('loading volumes...')
    torch.set_grad_enabled(False)

    # load each image as a torch.Tensor and normalize to [0,1]
    fixed_image = load_image(join(input_dir, fixed_image_fname), size=(numAscans, numBscans))
    moving_image = load_image(join(input_dir, moving_image_fname), size=(numAscans, numBscans))
    fixed_vol = load_volume(join(input_dir, fixed_vol_fname), size=(numAscans, numBscans)).float()
    moving_vol = load_volume(join(input_dir, moving_vol_fname), size=(numAscans, numBscans)).float()

    # # TESTING WITH COMPLEX DATA
    # moving_vol = load_volume(join(input_dir, moving_vol_fname), size=(numAscans, numBscans)).to(torch.complex64)
    # real_part = torch.real(moving_vol)
    # imag_part = torch.imag(moving_vol)
    #
    # # Compute magnitude (Euclidean distance)
    # magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    #
    # # Compute phase (angle in radians)
    # phase = torch.atan2(imag_part, real_part)
    print('successfully loaded volumes')

    print('start registration...')
    # register images
    eyeliner = EyeLinerP(
        reg=reg, lambda_tps=1.0, image_size=(num_channels, numAscans, numBscans), device=device)
    fixed_kpts, moving_kpts, reg_image, reg_vol = eyeliner(fixed_image, moving_image, moving_vol)

    # visualize registered images, convert to numpy and remove batch dimension
    reg_image = reg_image.detach().cpu().numpy()[0]
    fixed_image = fixed_image.detach().cpu().numpy()[0]
    moving_image = moving_image.detach().cpu().numpy()[0]
    fixed_kpts = fixed_kpts.detach().cpu().numpy()[0]
    moving_kpts = moving_kpts.detach().cpu().numpy()[0]
    moving_vol = moving_vol.detach().cpu().numpy()[0]
    fixed_vol = fixed_vol.detach().cpu().numpy()[0]
    reg_vol = reg_vol.detach().cpu().numpy()[0]

    print('registration complete')

    overlay_image_3d(np.mean(fixed_vol[:, :, :], axis=0), np.mean(abs(reg_vol[:, :, :]), axis=0), f'vol_{moving_vol_fname[:-4]}_after_register.tif')
    overlay_image_3d(np.mean(fixed_vol[:, :, :], axis=0), np.mean(abs(moving_vol[:, :, :]), axis=0), f'vol_{moving_vol_fname[:-4]}_before_register.tif')

    # correct B-scan orientation
    fixed_vol = np.flip(np.transpose(fixed_vol, (0, 2, 1)), axis=0)
    moving_vol = np.flip(np.transpose(moving_vol, (0, 2, 1)), axis=0)
    reg_vol = np.flip(np.transpose(abs(reg_vol), (0, 2, 1)), axis=0)

    # # save volumes as tiff stack
    # for i in range(1000):
    #     tiff.imwrite('C:\\Users\\tiffa\\Documents\\1. Projects\\3D Reg\\EyeLiner-main\\outputs\\fixed_vol.tif', reg_vol[:, :, i], append=True)

    # overlay_image_2d(moving_image, fixed_image, f'{moving_vol_fname[:-4]}_before_register.tif')
    # overlay_image_2d(reg_image, fixed_image, f'{moving_vol_fname[:-4]}_after_register.tif')
    visualize_key_points(
        fixed_kpts, fixed_image[0], moving_kpts, moving_image[0], f'{moving_vol_fname[:-4]}_keypoints')

    print('saving .mat files...')
    # hdf5storage.savemat(join(output_dir,'fixed_vol.mat'), {'fixed':fixed_vol}, format='7.3')
    hdf5storage.savemat(join(output_dir, f'moving_vol_{moving_vol_fname[:-4]}.mat'), {'moving':moving_vol}, format='7.3')
    hdf5storage.savemat(join(output_dir, f'reg_vol_{moving_vol_fname[:-4]}.mat'), {'reg':reg_vol}, format='7.3')

    print('save complete')
    print('total time:', time.time() - start_time)

if __name__ == "__main__":
    num_channels=3
    # MIGHT NEED TO CHANGE THIS
    numPoints, numAscans, numBscans = 1000, 600, 600

    reg = ['tps','affine','perspective'][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    # CHANGE THIS
    input_dir = 'H:\\OCT_MJ_MFOV\\Reg'
    output_dir = 'H:\\OCT_MJ_MFOV\\Reg\\outputs'

    # using absolute data as inputs
    # load in 2D data to get transformation coordinates
    fixed_image_fname = 'fixed_292223_enface_oct.tif'
    moving_image_fname = '668818_enface_oct.tif'

    # load in volume data
    fixed_vol_fname = 'fixed_292223_octv.mat'
    moving_vol_fname = '668818_octv.mat'


    # # TESTING WITH COMPLEX DATA
    # moving_image_fname = 'fixed_372322_enface_oct.tif'
    # fixed_image_fname = 'moving_939084_enface_oct.tif'
    # fixed_vol_fname = 'moving_939084_octv.mat'
    # moving_vol_fname = 'RawData_372322/CompCplx.mat'

    main()
