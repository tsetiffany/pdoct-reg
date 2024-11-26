
import cv2
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

from torch.nn import functional as F
from src.utils import normalize_coordinates, unnormalize_coordinates, TPS
from src.detectors import get_keypoints_splg, get_keypoints_loftr
from kornia.geometry.ransac import RANSAC
from src.oct_processing import combineCplx
from scipy.ndimage import gaussian_filter

class EyeLinerP():
    ''' API for pairwise retinal image registration '''
    def __init__(self, kp_method='splg', reg='affine',
                 lambda_tps=1., image_size=(3, 256, 256), device='cpu'
    ):
        self.kp_method = kp_method
        self.reg = reg
        self.lambda_tps = lambda_tps
        self.image_size = image_size
        self.device = device

    def get_corr_keypoints(self, fixed_image, moving_image):
        if self.kp_method == 'splg':
            keypoints_fixed, keypoints_moving = get_keypoints_splg(fixed_image, moving_image)
        else:
            keypoints_fixed, keypoints_moving = get_keypoints_loftr(fixed_image, moving_image)
        return keypoints_fixed, keypoints_moving

    def compute_lq_affine(self, points0, points1):
        # convert to homogenous coordinates
        P = torch.cat([
            points0, torch.ones(1, points0.shape[1], 1).to(self.device)], dim=2) # (b, n, 3)
        P = torch.permute(P, (0, 2, 1)) # (b, 3, n)
        Q = torch.cat([
            points1, torch.ones(1, points1.shape[1], 1).to(self.device)], dim=2) # (b, n, 3)
        Q = torch.permute(Q, (0, 2, 1)) # (b, 3, n)

        # compute lq sol
        Q_T = torch.permute(Q, (0, 2, 1)) # (b, n, 3)
        QQ_T = torch.einsum('bmj,bjn->bmn', Q, Q_T) # (b, 3, 3)

        try:
            A = P @ Q_T @ torch.linalg.inv(QQ_T)
        except:
            A = P @ Q_T @ torch.linalg.pinv(QQ_T)

        return A.cpu()

    def compute_tps(self, keypoints_moving, keypoints_fixed, grid_shape, lmbda):
        """
        @Params:
          grid_shape: [1,c,h,w]
        @Return
          theta: [1,num_matches,2]
          grid:  [1,h,w,2]
        """
        # print(grid_shape)
        theta, grid = TPS(dim=2).grid_from_points(
            keypoints_moving, keypoints_fixed, grid_shape=grid_shape,
            lmbda=torch.tensor(lmbda).to(self.device)
        )
        return theta, grid

    def get_registration(self, fixed_kpts, moving_kpts):

        if self.reg == 'tps':
            # scale between -1 and 1
            fixed_kpts_ = normalize_coordinates(fixed_kpts, self.image_size[1:])
            moving_kpts_ = normalize_coordinates(moving_kpts, self.image_size[1:])
            theta = self.compute_tps(
                moving_kpts_, fixed_kpts_, [1] + list(self.image_size), self.lambda_tps
            )
            # keypoints_fixed = unnormalize_coordinates(keypoints_fixed, self.image_size[1:])
            # keypoints_moving = unnormalize_coordinates(keypoints_moving, self.image_size[1:])

        elif self.reg == 'affine':
            # compute least squares solution using key points
            theta = self.compute_lq_affine(fixed_kpts, moving_kpts).float()

        elif self.reg == "perspective":
            theta, mask = cv2.findHomography(
                fixed_kpts.detach().cpu().numpy(),
                moving_kpts.detach().cpu().numpy(), cv2.RANSAC
            )
            # m_kpts0 = m_kpts0[mask[:,0]]
            # m_kpts1 = m_kpts1[mask[:,0]]

        else:
            raise NotImplementedError(
                'Only affine and thin-plate spline registration supported.')

        return theta

    def KPRefiner(self, keypoints_fixed, keypoints_moving):
        _, mask = RANSAC(model_type='homography')(
            keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0)
        )
        mask = mask.squeeze()
        keypoints_fixed_filtered = keypoints_fixed[:, mask]
        keypoints_moving_filtered = keypoints_moving[:, mask]
        return keypoints_fixed_filtered, keypoints_moving_filtered

    def apply_transform_2d(self, theta, moving_image):
        """
        @Params:
           theta:
             torch [3,3] if affine
             torch [1,h,w,2] if tps
           moving_image: torch [1,c,h,w]
        @Return
           warped: torch [1,c,h,w]
        """
        # print(moving_image.shape, theta.shape)
        _,c,h,w = moving_image.shape
        is_gray = c == 1

        if self.reg == "tps":
            theta, grid = theta
            assert grid.shape == (1,h,w,2)
            warped_image = F.grid_sample(
                moving_image, grid=grid, mode="bilinear",
                padding_mode="zeros", align_corners=False)

        elif self.reg == "affine":
            assert theta.shape == (1,3,3)
            warped_image = moving_image[0].permute(1,2,0).detach().cpu().numpy() # [h,w,c]
            affine_mat = theta[0].detach().cpu().numpy() # (3,3)
            warped_image = cv2.warpAffine(warped_image, affine_mat[:2,:], (h,w))
            if is_gray:
                warped_image = warped_image[...,None]
            warped_image = torch.tensor(warped_image)[None].permute(0,3,1,2) # [c,h,w]

        elif self.reg == "perspective":
            warped_image = moving_image[0].permute(1,2,0).detach().cpu().numpy() # [h,w,c]
            warped_image = cv2.warpPerspective(warped_image, theta, (h,w))
            if is_gray:
                warped_image = warped_image[...,None]
            warped_image = torch.tensor(warped_image)[None].permute(0,3,1,2) # [c,h,w]

        else:
            raise NotImplementedError('Only affine and deformation fields supported.')

        # print(warped_image.shape)
        return warped_image

    def apply_transform_to_volume(self, theta, moving_vol, mode='bilinear'):
        """
        Apply the 2D transformation (TPS) to each slice in the 3D volume.

        @Params:
            moving_vol: (torch.Tensor): Input volume of shape (1, d, h, w)

        @Return:
            transformed_vol: torch.Tensor [1,c,d,h,w] with the transformation applied to each depth slice.
        """
        if moving_vol.dim() == 4:  # [c, d, h, w] case
            moving_vol = moving_vol.unsqueeze(0)  # Add batch dimension to make it [1, c, d, h, w]

        _, c, d, h, w = moving_vol.shape
        is_gray = c == 1  # Check if the image is grayscale (single channel)

        # Initialize an empty list to store transformed slices
        warped_slices = []

        if self.reg == "tps":
            theta, grid = theta
            assert grid.shape == (1, h, w, 2)

            # Loop through each depth slice
            for i in range(d):
                # Extract the 2D slice from the 3D volume (shape: [1, c, h, w])
                moving_slice = moving_vol[:, :, i, :, :].float()

                # Apply the 2D transformation (TPS or affine) to the 2D slice
                warped_slice = F.grid_sample(
                    moving_slice, grid=grid, mode=mode,
                    padding_mode="zeros", align_corners=False
                )

                # Replace zero values (padding regions) with NaN
                # warped_slice[warped_slice == 0] = float('nan')

                # Add the transformed slice to the list
                warped_slices.append(warped_slice)

        # Stack all warped slices back into a 3D volume (along the depth dimension)
        warped_volume = torch.stack(warped_slices, dim=2)  # Shape: [1, c, d, h, w]

        return warped_volume.squeeze(0)

    @staticmethod
    def apply_transform_points(theta, moving_keypoints,
                               ctrl_keypoints=None, tgt_keypoints=None, lmbda=None):
        if theta.shape == (3, 3):
            moving_keypoints = torch.cat([
                moving_keypoints, torch.ones(moving_keypoints.shape[0], 1)], dim=1).T # (3,N)
            # (2,3) @ (3,N) = (2,N) --> (N,2)
            warped_kp = torch.mm(theta[:2, :], moving_keypoints).T
        else:
            # method 1: recompute transforms?
            moving_keypoints = normalize_coordinates(
                moving_keypoints.unsqueeze(0).float(), shape=(256, 256))
            tgt_keypoints = normalize_coordinates(
                tgt_keypoints.unsqueeze(0).float(), shape=(256, 256))
            ctrl_keypoints = normalize_coordinates(
                ctrl_keypoints.unsqueeze(0).float(), shape=(256, 256))

            warped_kp = TPS(dim=2).points_from_points(
                ctrl_keypoints,
                tgt_keypoints,
                moving_keypoints,
                lmbda=torch.tensor(lmbda),
            )
            warped_kp = unnormalize_coordinates(warped_kp, shape=(256, 256)).squeeze(0)

            # method 2: apply inverse transform to the fixed points
            # moving_keypoints = normalize_coordinates(moving_keypoints.unsqueeze(0).float(), shape=(256, 256))
            # ctrl_keypoints = normalize_coordinates(ctrl_keypoints.unsqueeze(0).float(), shape=(256, 256))
            # warped_kp = TPS(dim=2).deform_points(
            #     theta.unsqueeze(0),
            #     ctrl_keypoints,
            #     moving_keypoints
            # )
            # warped_kp = unnormalize_coordinates(warped_kp, shape=(256, 256)).squeeze(0)

        return warped_kp

    def __call__(self, fixed_image, moving_image, moving_vol):
        fixed_image = fixed_image.to(self.device)
        moving_image = moving_image.to(self.device)
        moving_vol = moving_vol.to(self.device)

        fixed_kpts, moving_kpts = self.get_corr_keypoints(fixed_image, moving_image)
        # kpts_fixed, kpts_moving = self.KPRefiner(fixed_kpts, moving_kpts)

        theta = self.get_registration(fixed_kpts, moving_kpts)
        reg_image = self.apply_transform_2d(theta, moving_image)

        reg_volume = self.apply_transform_to_volume(theta, moving_vol, mode="bilinear")

        # # TEST WITH COMPLEX DATA
        # magnitude = magnitude.to(self.device)
        # phase = phase.to(self.device)
        # reg_volume_magnitude = self.apply_transform_to_volume(theta, magnitude, mode="bilinear")
        # reg_volume_phase = self.apply_transform_to_volume(theta, phase, mode="bilinear")
        #
        # real_part = reg_volume_magnitude * torch.cos(reg_volume_phase)  # Real part: Magnitude * cos(phase)
        # imag_part = reg_volume_magnitude * torch.sin(reg_volume_phase)  # Imaginary part: Magnitude * sin(phase)
        #
        # reg_volume = torch.complex(real_part, imag_part)

        return fixed_kpts, moving_kpts, reg_image, reg_volume

