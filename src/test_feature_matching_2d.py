from src.eyeliner import EyeLinerP
from src.utils import load_image
from os.path import join
from lightglue import viz2d
import tifffile as tiff
from scipy.io import savemat
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_image_2d(reg, fixed, fname):
    img = cv2.addWeighted(fixed, 0.5, reg, 0.5, 0)[0]
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    tiff.imwrite(join(output_dir, fname), img_uint8)

def visualize_key_points(kpts0, img0, kpts1, img1, fname):
    axes = viz2d.plot_images([img0, img1])
    viz2d.plot_matches(kpts0, kpts1, color="cyan", lw=0.2)
    viz2d.save_plot(join(output_dir, f'{fname}_matches.png'))

    # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([img0, img1])
    viz2d.plot_keypoints([kpts0, kpts1], ps=6)  #, colors=[kpc0, kpc1], ps=6)
    viz2d.save_plot(join(output_dir, fname))


input_dir = 'I:\\Mo_test'
output_dir = 'I:\\Mo_test'

os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
fixed_image_fname = next(
        (f for f in os.listdir(input_dir) if f.startswith('fixed_') and f.endswith('.tif')), None)
moving_image_fname = next(
        (f for f in os.listdir(input_dir) if f.endswith('.tif') and not f.startswith('fixed_')), None)
# fixed_image_fname = "new_fixed.tif"
# moving_image_fname = "new_moving.tif"
numAscans, numBscans = 500,500

test_FM = EyeLinerP(image_size=(3,numBscans,numAscans))

fixed_image = load_image(join(input_dir, fixed_image_fname), size=(numBscans, numAscans))
moving_image = load_image(join(input_dir, moving_image_fname), size=(numBscans, numAscans))

fixed_image = fixed_image.to(test_FM.device)
moving_image = moving_image.to(test_FM.device)
fixed_kpts, moving_kpts = test_FM.get_corr_keypoints(fixed_image, moving_image)
theta = test_FM.get_registration(fixed_kpts, moving_kpts)
reg_image = test_FM.apply_transform_2d(theta, moving_image)

reg_image = reg_image.detach().cpu().numpy()[0]
fixed_image = fixed_image.detach().cpu().numpy()[0]
moving_image = moving_image.detach().cpu().numpy()[0]
fixed_kpts = fixed_kpts.detach().cpu().numpy()[0]
moving_kpts = moving_kpts.detach().cpu().numpy()[0]

plt.imshow(reg_image[0])
plt.show()

visualize_key_points(
        fixed_kpts, fixed_image[0], moving_kpts, moving_image[0], f'{moving_image_fname[:-4]}_keypoints')

overlay_image_2d(fixed_image,moving_image,f'{moving_image_fname[:-4]}_before_register.tif')
overlay_image_2d(reg_image,fixed_image,f'{moving_image_fname[:-4]}_after_register.tif')

# Save the data to a .mat file
# savemat(join(output_dir,'reg_image_3.mat'), {'reg_image': reg_image})
#
# print("Saved registration results to 'reg_data.mat'")