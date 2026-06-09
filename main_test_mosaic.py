'''
FEATURE-BASED 3D PD-OCT IMAGE REGISTRATION
v1.2
Tiffany Tse
updated 01/21/2025

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
import os
import hdf5storage
import torch
import numpy as np
import tifffile as tiff
import time

from os.path import join
from lightglue import viz2d
import matplotlib.pyplot as plt
from src.global_mcorr import global_mcorr
from src.utils import load_image, load_volume, normalize_coordinates
from src.eyeliner import EyeLinerP


def overlay_image_2d(reg, fixed, fname):
    img = cv2.addWeighted(fixed, 0.5, reg, 0.5, 0)[0]
    # img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    tiff.imwrite(join(output_dir, fname), img)

def overlay_image_3d(reg, fixed, fname):
    img = cv2.addWeighted(fixed, 0.5, reg, 0.5, 0)
    tiff.imwrite(join(output_dir, fname), img)

def visualize_key_points(kpts0, img0, kpts1, img1, fname):
    axes = viz2d.plot_images([img0, img1])
    viz2d.plot_matches(kpts0, kpts1, color="cyan", lw=0.2)
    viz2d.save_plot(join(output_dir, f'{fname}_matches.png'))

    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([img0, img1])
    viz2d.plot_keypoints([kpts0, kpts1], ps=6)  #, colors=[kpc0, kpc1], ps=6)
    viz2d.save_plot(join(output_dir, fname))

def reorient_volume(vol,mode):
    v = np.copy(vol)
    if mode == 0:
        v = np.flip(np.transpose(vol, (0, 2, 1)), axis=0)
    elif mode == 1:
        v = np.flip(np.flip(np.transpose(vol, (0, 2, 1)), axis=0),axis=2)
    elif mode == 2:
        v = np.transpose(vol, (0, 2, 1))
    else:
        raise ValueError(f"Unknown mode {mode}")
    return v

def run_local(mode, fixed_vol_fname, fixed_image_fname, fixed_dopu_vol_fname, moving_vol_fname, moving_image_fname, moving_dopu_fname,
         num_channels, numPoints, numAscans, numBscans, reg, device, input_dir,
         output_dir, save_fixed_vol, mosaic=False):
    start_time = time.time()
    print('loading volumes...')
    torch.set_grad_enabled(False)

    # load each image as a torch.Tensor and normalize to [0,1]
    fixed_image = load_image(join(input_dir, fixed_image_fname), size=(numBscans, numAscans))
    moving_image = load_image(join(input_dir, moving_image_fname), size=(numBscans, numAscans))
    # fixed_vol = load_volume(join(input_dir, fixed_vol_fname), size=(numAscans, numBscans)).float()
    # fixed_dopu_vol = load_volume(join(input_dir, fixed_dopu_vol_fname),size=(numAscans,numBscans)).float()
    # moving_vol = load_volume(join(input_dir, moving_vol_fname), size=(numAscans, numBscans)).float()
    # moving_dopu = load_volume(join(input_dir, moving_dopu_fname), size=(numAscans, numBscans)).float()

    # fixed_image = torch.unsqueeze(torch.permute(torch.mean(fixed_vol[:,:,:,:],1), [0, 2, 1]), 0)
    # moving_image =  torch.unsqueeze(torch.permute(torch.mean(moving_vol[:,:,:,:],1), [0, 2, 1]), 0)

    print('successfully loaded volumes')

    print('start registration...')
    # register images
    eyeliner = EyeLinerP(
        reg=reg, lambda_tps=1.0, image_size=(num_channels, numBscans, numAscans), device=device)
    fixed_kpts, moving_kpts, reg_image, reg_vol, reg_dopu_vol = eyeliner(
        fixed_image, moving_image, moving_vol, moving_dopu, mosaic=mosaic)

    # visualize registered images, convert to numpy and remove batch dimension
    reg_image = reg_image.detach().cpu().numpy()[0]
    fixed_image = fixed_image.detach().cpu().numpy()[0]
    moving_image = moving_image.detach().cpu().numpy()[0]
    fixed_kpts = fixed_kpts.detach().cpu().numpy()[0]
    moving_kpts = moving_kpts.detach().cpu().numpy()[0]
    moving_vol = moving_vol.detach().cpu().numpy()[0]
    fixed_vol = fixed_vol.detach().cpu().numpy()[0]
    fixed_dopu_vol = fixed_dopu_vol.detach().cpu().numpy()[0]
    reg_vol = reg_vol.detach().cpu().numpy()[0]
    reg_dopu_vol = reg_dopu_vol.detach().cpu().numpy()[0]

    print('registration complete')

    if mosaic:
        # In mosaic mode the registered volume may be larger than fixed_vol,
        # so a pixel-aligned overlay with fixed_vol is not meaningful.
        # Save mean-projection of the registered volume on its own.
        tiff.imwrite(join(output_dir, f'vol_{moving_vol_fname[:-4]}_mosaic.tif'),
                     np.mean(abs(reg_vol[:, :, :]), axis=0))
    else:
        overlay_image_3d(np.mean(fixed_vol[:,:,:], axis=0), np.mean(abs(reg_vol[:, :, :]), axis=0),
                         f'vol_{moving_vol_fname[:-4]}_after_register.tif')
        overlay_image_3d(np.mean(fixed_vol[:,:,:], axis=0), np.mean(abs(moving_vol[:,:,:]), axis=0),
                         f'vol_{moving_vol_fname[:-4]}_before_register.tif')

    # overlay_image_2d(fixed_image, moving_image, f'{moving_image_fname[:-4]}_before_register.tif')
    # overlay_image_2d(reg_image, fixed_image, f'{moving_image_fname[:-4]}_after_register.tif')

    # correct B-scan orientation
    fixed_vol = reorient_volume(fixed_vol, mode)
    fixed_dopu_vol = reorient_volume(fixed_dopu_vol, mode)
    moving_vol = reorient_volume(moving_vol, mode)
    reg_vol = reorient_volume(reg_vol, mode)
    reg_dopu_vol = reorient_volume(reg_dopu_vol, mode)

    # overlay_image_2d(moving_image, fixed_image, f'{moving_vol_fname[:-4]}_before_register.tif')
    # overlay_image_2d(reg_image, fixed_image, f'{moving_vol_fname[:-4]}_after_register.tif')
    visualize_key_points(
        fixed_kpts, fixed_image[0], moving_kpts, moving_image[0], f'{moving_vol_fname[:-4]}_keypoints')

    print('saving .mat files...')
    if save_fixed_vol:
        hdf5storage.savemat(join(output_dir, f'fixed_vol_{fixed_vol_fname[:-4]}.mat'), {'fixed': fixed_vol},
                            format='7.3')
        hdf5storage.savemat(join(output_dir, f'fixed_dopu_vol_{fixed_dopu_vol_fname[:-4]}.mat'), {'fixed_dopu': fixed_dopu_vol},
                            format='7.3')
        save_fixed_vol = False  # Set the flag to False to prevent further saving

    # hdf5storage.savemat(join(output_dir, f'moving_vol_{moving_vol_fname[:-4]}.mat'), {'moving': moving_vol},
    #                     format='7.3')
    hdf5storage.savemat(join(output_dir, f'reg_vol_{moving_vol_fname[:-4]}.mat'), {'reg': reg_vol}, format='7.3')
    hdf5storage.savemat(join(output_dir, f'reg_dopu_vol_{moving_dopu_fname[:-4]}.mat'), {'reg_dopu': reg_dopu_vol}, format='7.3')

    print('save complete')
    print('total time:', time.time() - start_time)

    return save_fixed_vol


def run_global_montage(mode, fixed_vol_fname, fixed_image_fname, fixed_dopu_vol_fname,
                       moving_vol_fnames, moving_image_fnames, moving_dopu_fnames,
                       num_channels, numPoints, numAscans, numBscans, reg, device,
                       input_dir, output_dir):
    """Register all moving images to fixed and composite onto one shared canvas.

    Two-pass approach:
      Pass 1 - detect keypoints for every pair and compute per-pair canvas extents.
      Merge all extents into a single global canvas.
      Pass 2 - re-warp each moving image/volume using the global canvas extent, then
      composite the en-face mean-projections by averaging across all frames.

    The fixed image is also placed at its natural position (the [-1, 1] region) on
    the global canvas before averaging.

    Outputs
    -------
    global_montage.tif  - composite 2D en-face projection on the global canvas
    reg_vol_*.mat       - each registered volume on the global canvas
    reg_dopu_vol_*.mat  - each registered DOPU volume on the global canvas
    fixed_vol_*.mat     - fixed volume (saved once, on its original grid)
    """
    start_time = time.time()
    torch.set_grad_enabled(False)

    print('loading fixed volume...')
    fixed_image = load_image(join(input_dir, fixed_image_fname), size=(numBscans, numAscans))
    fixed_vol   = load_volume(join(input_dir, fixed_vol_fname),  size=(numAscans, numBscans)).float()
    fixed_dopu_vol = load_volume(join(input_dir, fixed_dopu_vol_fname), size=(numAscans, numBscans)).float()

    fixed_image = fixed_image.to(device)

    eyeliner = EyeLinerP(
        reg=reg, lambda_tps=1.0, image_size=(num_channels, numBscans, numAscans), device=device
    )

    # ------------------------------------------------------------------ Pass 1
    # Detect keypoints for every pair and accumulate the global canvas extent.
    # ------------------------------------------------------------------ Pass 1
    print('Pass 1: detecting keypoints and computing canvas extents...')
    all_kpts = []          # list of (fixed_kpts, moving_kpts) tensors
    gx_min, gx_max = -1.0, 1.0
    gy_min, gy_max = -1.0, 1.0

    for i, moving_image_fname in enumerate(moving_image_fnames):
        moving_image = load_image(join(input_dir, moving_image_fname),
                                  size=(numBscans, numAscans)).to(device)
        fixed_kpts, moving_kpts = eyeliner.get_corr_keypoints(fixed_image, moving_image)

        fixed_kpts_n  = normalize_coordinates(fixed_kpts,  eyeliner.image_size[1:])
        moving_kpts_n = normalize_coordinates(moving_kpts, eyeliner.image_size[1:])
        x_min, x_max, y_min, y_max = eyeliner.compute_canvas_extent(fixed_kpts_n, moving_kpts_n)

        gx_min = min(gx_min, x_min)
        gx_max = max(gx_max, x_max)
        gy_min = min(gy_min, y_min)
        gy_max = max(gy_max, y_max)

        all_kpts.append((fixed_kpts, moving_kpts))
        print(f'  [{i+1}/{len(moving_image_fnames)}] extent: '
              f'x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}]')

    global_extent = (gx_min, gx_max, gy_min, gy_max)
    H, W = numBscans, numAscans
    W_exp = int(round(W * (gx_max - gx_min) / 2.0))
    H_exp = int(round(H * (gy_max - gy_min) / 2.0))
    print(f'Global canvas: {H_exp} x {W_exp}  '
          f'(original {H} x {W})  '
          f'extent x=[{gx_min:.3f},{gx_max:.3f}] y=[{gy_min:.3f},{gy_max:.3f}]')

    # ------------------------------------------------------------------ Pass 2
    # Register each pair on the global canvas and accumulate the composite.
    # ------------------------------------------------------------------ Pass 2
    composite = np.zeros((H_exp, W_exp), dtype=np.float64)
    weight    = np.zeros((H_exp, W_exp), dtype=np.float64)

    # Place fixed en-face projection at its natural position on the global canvas.
    #fixed_vol_np  = fixed_vol.detach().cpu().numpy()[0]           # [D, H, W]
    #fixed_proj    = np.mean(np.abs(fixed_vol_np), axis=0)         # [H, W]
    row0 = int(round((-1.0 - gy_min) / (gy_max - gy_min) * H_exp))
    col0 = int(round((-1.0 - gx_min) / (gx_max - gx_min) * W_exp))
    r1 = min(row0 + H, H_exp)
    c1 = min(col0 + W, W_exp)
    # composite[row0:r1, col0:c1] += fixed_proj[:r1-row0, :c1-col0]
    weight[row0:r1, col0:c1]    += 1.0

    print('Pass 2: registering with global canvas extent...')
    save_fixed_vol = False
    for idx, (moving_vol_fname, moving_image_fname, moving_dopu_fname,
              (fixed_kpts, moving_kpts)) in enumerate(
            zip(moving_vol_fnames, moving_image_fnames, moving_dopu_fnames, all_kpts)):

        moving_vol  = load_volume(join(input_dir, moving_vol_fname),
                                  size=(numAscans, numBscans)).float().to(device)
        moving_dopu = load_volume(join(input_dir, moving_dopu_fname),
                                  size=(numAscans, numBscans)).float().to(device)

        theta   = eyeliner.get_registration(fixed_kpts, moving_kpts,
                                            mosaic=True, canvas_extent=global_extent)
        reg_vol  = eyeliner.apply_transform_to_volume(theta, moving_vol,  mode='bilinear')
        reg_dopu = eyeliner.apply_transform_to_volume(theta, moving_dopu, mode='bilinear')

        reg_vol_np  = reg_vol.detach().cpu().numpy()[0]   # [D, H_exp, W_exp]
        reg_dopu_np = reg_dopu.detach().cpu().numpy()[0]

        # Accumulate en-face projection into composite
        proj = np.mean(np.abs(reg_vol_np), axis=0)        # [H_exp, W_exp]
        mask = proj > 0
        composite += proj
        weight    += mask.astype(np.float64)

        # Save per-volume .mat files (reoriented)
        reg_vol_save  = reorient_volume(reg_vol_np,  mode)
        reg_dopu_save = reorient_volume(reg_dopu_np, mode)

        hdf5storage.savemat(
            join(output_dir, f'reg_vol_{moving_vol_fname[:-4]}.mat'),
            {'reg': reg_vol_save}, format='7.3')
        hdf5storage.savemat(
            join(output_dir, f'reg_dopu_vol_{moving_dopu_fname[:-4]}.mat'),
            {'reg_dopu': reg_dopu_save}, format='7.3')

        if save_fixed_vol:
            fixed_vol_save      = reorient_volume(fixed_vol_np, mode)
            fixed_dopu_vol_np   = fixed_dopu_vol.detach().cpu().numpy()[0]
            fixed_dopu_vol_save = reorient_volume(fixed_dopu_vol_np, mode)
            hdf5storage.savemat(
                join(output_dir, f'fixed_vol_{fixed_vol_fname[:-4]}.mat'),
                {'fixed': fixed_vol_save}, format='7.3')
            hdf5storage.savemat(
                join(output_dir, f'fixed_dopu_vol_{fixed_dopu_vol_fname[:-4]}.mat'),
                {'fixed_dopu': fixed_dopu_vol_save}, format='7.3')
            save_fixed_vol = False

        print(f'  [{idx+1}/{len(moving_vol_fnames)}] {moving_vol_fname} done')

    # Average overlapping regions and save composite montage
    weight[weight == 0] = 1.0
    montage = (composite / weight).astype(np.float32)
    montage_path = join(output_dir, 'global_montage.tif')
    tiff.imwrite(montage_path, montage)
    print(f'Saved global montage → {montage_path}  ({H_exp} x {W_exp} px)')
    print(f'Total time: {time.time() - start_time:.1f}s')


def run_global_montage_image_only(
    mode, input_dir, output_dir, fixed_image_fname, moving_image_fnames,
    num_channels, numPoints, numAscans, numBscans, reg, device,
):
    """Register all moving images to fixed and composite onto one shared canvas.

    Two-pass approach:
      Pass 1 - detect keypoints for every pair and compute per-pair canvas extents.
      Merge all extents into a single global canvas.
      Pass 2 - re-warp each moving image/volume using the global canvas extent, then
      composite the en-face mean-projections by averaging across all frames.

    The fixed image is also placed at its natural position (the [-1, 1] region) on
    the global canvas before averaging.

    Outputs
    -------
    global_montage.tif  - composite 2D en-face projection on the global canvas
    reg_vol_*.mat       - each registered volume on the global canvas
    reg_dopu_vol_*.mat  - each registered DOPU volume on the global canvas
    fixed_vol_*.mat     - fixed volume (saved once, on its original grid)
    """
    start_time = time.time()
    torch.set_grad_enabled(False)

    print('loading fixed volume...')
    fixed_image = load_image(join(input_dir, fixed_image_fname), size=(numBscans, numAscans))
    fixed_image = fixed_image.to(device)

    eyeliner = EyeLinerP(reg=reg, lambda_tps=1.0,
                         image_size=(num_channels, numBscans, numAscans), device=device)

    # ------------------------------------------------------------------ Pass 1
    # Detect keypoints for every pair and accumulate the global canvas extent.
    # ------------------------------------------------------------------ Pass 1
    print('Pass 1: detecting keypoints and computing canvas extents...')
    all_kpts = []          # list of (fixed_kpts, moving_kpts) tensors
    gx_min, gx_max = -1.0, 1.0
    gy_min, gy_max = -1.0, 1.0

    for i, moving_image_fname in enumerate(moving_image_fnames):
        moving_image = load_image(
            join(input_dir, moving_image_fname), size=(numBscans, numAscans)
        ).to(device)
        fixed_kpts, moving_kpts = eyeliner.get_corr_keypoints(fixed_image, moving_image)

        fixed_kpts_n  = normalize_coordinates(fixed_kpts,  eyeliner.image_size[1:])
        moving_kpts_n = normalize_coordinates(moving_kpts, eyeliner.image_size[1:])
        x_min, x_max, y_min, y_max = eyeliner.compute_canvas_extent(fixed_kpts_n, moving_kpts_n)

        gx_min = min(gx_min, x_min)
        gx_max = max(gx_max, x_max)
        gy_min = min(gy_min, y_min)
        gy_max = max(gy_max, y_max)

        all_kpts.append((fixed_kpts, moving_kpts))
        print(f'  [{i+1}/{len(moving_image_fnames)}] extent: '
              f'x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}]')

    global_extent = (gx_min, gx_max, gy_min, gy_max)
    H, W = numBscans, numAscans
    W_exp = int(round(W * (gx_max - gx_min) / 2.0))
    H_exp = int(round(H * (gy_max - gy_min) / 2.0))
    print(f'Global canvas: {H_exp} x {W_exp}  '
          f'(original {H} x {W})  '
          f'extent x=[{gx_min:.3f},{gx_max:.3f}] y=[{gy_min:.3f},{gy_max:.3f}]')

    # ------------------------------------------------------------------ Pass 2
    # Register each pair on the global canvas and accumulate the composite.
    # ------------------------------------------------------------------ Pass 2
    composite = np.zeros((H_exp, W_exp), dtype=np.float64)
    weight    = np.zeros((H_exp, W_exp), dtype=np.float64)

    # Place fixed en-face projection at its natural position on the global canvas.
    row0 = int(round((-1.0 - gy_min) / (gy_max - gy_min) * H_exp))
    col0 = int(round((-1.0 - gx_min) / (gx_max - gx_min) * W_exp))
    r1 = min(row0 + H, H_exp)
    c1 = min(col0 + W, W_exp)
    # composite[row0:r1, col0:c1] += fixed_proj[:r1-row0, :c1-col0]
    weight[row0:r1, col0:c1]    += 1.0

    print('Pass 2: registering with global canvas extent...')
    for idx, (moving_image_fname, (fixed_kpts, moving_kpts)) in enumerate(
        zip(moving_image_fnames, all_kpts)
    ):
        moving_img = load_image(join(input_dir, moving_image_fname),
                                size=(numBscans, numAscans)).to(device)
        theta = eyeliner.get_registration(
            fixed_kpts, moving_kpts, mosaic=True, canvas_extent=global_extent
        )
        reg_img = eyeliner.apply_transform_to_volume(theta, moving_img,  mode='bilinear')
        reg_img_np = reg_img.detach().cpu().numpy()[0]

        # Accumulate en-face projection into composite
        proj = np.mean(np.abs(reg_img_np), axis=0) # [H_exp, W_exp]
        mask = proj > 0
        composite += proj
        weight    += mask.astype(np.float64)

    # Average overlapping regions and save composite montage
    weight[weight == 0] = 1.0
    montage = (composite / weight).astype(np.float32)
    montage_path = join(output_dir, 'global_montage.tif')
    tiff.imwrite(montage_path, montage)
    np.save(join(output_dir, 'global_montage.npy'), montage)
    plt.imshow(montage, cmap='gray')
    plt.savefig(join(output_dir, 'global_montage.png'), dpi=300)
    plt.close()
    print(f'Saved global montage → {montage_path}  ({H_exp} x {W_exp} px)')
    print(f'Total time: {time.time() - start_time:.1f}s')


if __name__ == "__main__":

    ############################## input params #######################################
    num_channels = 3

    numPoints, numAscans, numBscans = 1000, 550, 550

    # MODE = 0 SINGLE TIMEPOINT REG ONLY
    # MODE = 1 SINGLE TIMEPOINT FOR LONGITUDINAL REG
    # MODE = 2 CO-REGISTER DIFFERENT TIMEPOINTS
    # note that modes 1 and 2 are to be used consecutively. otherwise, use mode 0.
    mode = [0,1,2][2]
    if mode == 2:
        FLAG_global_mcorr = 0
    else:
        FLAG_global_mcorr = 1

    input_dir = '/ubc/cs/research/kmyi/fred/code/pdoct-reg/data/inputs/'
    output_dir = '/ubc/cs/research/kmyi/fred/code/pdoct-reg/data/outputs/'

    ###################################################################################

    # MOSAIC = False  fixed frame registration: output has fixed frame's FOV (moving content
    #                 that falls outside the fixed FOV is zeroed).
    # MOSAIC = True   mosaicing: output canvas expands to cover the full warped-moving FOV,
    #                 so no moving content is cropped after registration.
    mosaic = True
    img_only = True

    reg = ['tps', 'affine', 'perspective'][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    if FLAG_global_mcorr == 1:
        global_mcorr(input_dir)  # Bidirectional global motion correction

    # fixed_image_fname = join(input_dir, "fixed_centre_registered_enface.tif")
    # fixed_vol_fname = next(
    #     (f for f in os.listdir(input_dir) if f.startswith('fixed_') and f.endswith('_octv_mcorr.mat')), None)
    # fixed_dopu_vol_fname = next(
    #     (f for f in os.listdir(input_dir) if f.startswith('fixed_') and f.endswith('_dopu_mcorr.mat')), None)
    fixed_image_fname = next(
        (f for f in os.listdir(input_dir) if f.startswith('fixed_') and f.endswith('.tif')), None)
    # if not fixed_vol_fname or not fixed_image_fname:
    #     raise FileNotFoundError("Fixed volume or fixed image not found in the input directory.")

    save_fixed_vol = True  # Flag to save only once

    # Collect moving volumes and images
    matched_images = [f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('fixed_')]
    # image_files = [f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('fixed_')]
    # dopu_vol_files = [f for f in os.listdir(input_dir) if f.endswith('_dopu_mcorr.mat') and not f.startswith('fixed_')]
    # moving_vol_files = [f for f in os.listdir(input_dir) if f.endswith('_octv_mcorr.mat') and not f.startswith('fixed_')]

    # Build matched lists of (vol, image, dopu) tuples
    # matched_vols, matched_images, matched_dopus = [], [], []
    # for moving_vol_fname in moving_vol_files:
    #     vol_prefix = moving_vol_fname.split('_')[0]
    #     moving_image_fname = next((img for img in image_files if img.startswith(vol_prefix)), None)
    #     moving_dopu_fname  = next((dopu for dopu in dopu_vol_files if dopu.startswith(vol_prefix)), None)
    #     if moving_image_fname and moving_dopu_fname:
    #         matched_vols.append(moving_vol_fname)
    #         matched_images.append(moving_image_fname)
    #         matched_dopus.append(moving_dopu_fname)
    #     else:
    #         print(f"No matching image/DOPU found for volume: {moving_vol_fname}, skipping.")


    if mosaic:
        # Global montage: all images onto one shared canvas.
        if img_only:
            run_global_montage_image_only(
                mode, input_dir, output_dir, fixed_image_fname, matched_images,
                num_channels, numPoints, numAscans, numBscans, reg, device,
            )
        else:
            run_global_montage(
                mode, fixed_vol_fname, fixed_image_fname,
                fixed_dopu_vol_fname, matched_vols, matched_images, matched_dopus,
                num_channels, numPoints, numAscans, numBscans, reg, device, input_dir, output_dir,
            )
    else:
        # Per-pair registration (original behaviour).
        save_fixed_vol = True
        for idx, (moving_vol_fname, moving_image_fname, moving_dopu_fname) in enumerate(
                zip(matched_vols, matched_images, matched_dopus), start=1):
            save_fixed_vol = run_local(
                mode=mode,
                fixed_vol_fname=fixed_vol_fname,
                fixed_image_fname=fixed_image_fname,
                fixed_dopu_vol_fname=fixed_dopu_vol_fname,
                moving_vol_fname=moving_vol_fname,
                moving_image_fname=moving_image_fname,
                moving_dopu_fname=moving_dopu_fname,
                num_channels=num_channels,
                numPoints=numPoints,
                numAscans=numAscans,
                numBscans=numBscans,
                reg=reg,
                device=device,
                input_dir=input_dir,
                output_dir=output_dir,
                save_fixed_vol=save_fixed_vol,
                mosaic=False,
            )
            print(f"Registration {idx}/{len(matched_vols)} complete")
