import os
import numpy as np
import mat73
import hdf5storage
from scipy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from skimage import exposure


def dftregistration(buf1ft, buf2ft):
    nr, nc = buf2ft.shape
    Nr = ifftshift(np.arange(-nr // 2, nr // 2 + (nr % 2)))
    Nc = ifftshift(np.arange(-nc // 2, nc // 2 + (nc % 2)))
    CC = ifft2(buf1ft * np.conj(buf2ft))
    CCabs = np.abs(CC)

    row_shift, col_shift = np.unravel_index(np.argmax(CCabs), CCabs.shape)
    row_shift = Nr[row_shift]
    col_shift = Nc[col_shift]

    CCmax = CCabs.max()
    error = 0
    diffphase = 0

    return [error, CCmax, row_shift, col_shift]


def global_mcorr(filepath):
    vol_files = [f for f in os.listdir(filepath) if f.endswith('_octv.mat')]

    for idx, oct_file in enumerate(vol_files,start=1):
        vol_prefix = oct_file.rsplit('_', 1)[0]
        dopu_file = vol_prefix + '_dopu.mat'

        # Full paths
        oct_path = os.path.join(filepath, oct_file)
        dopu_path = os.path.join(filepath, dopu_file)

        # Try loading OCT
        try:
            oct_data = mat73.loadmat(oct_path)
            OCT = oct_data['OCT']
        except Exception as e:
            print(f"Error loading {oct_path}: {e}")
            continue

        # Try loading DOPU
        try:
            dopu_data = mat73.loadmat(dopu_path)
            DOPU = dopu_data['DOPU']
        except Exception as e:
            print(f"Error loading {dopu_path}: {e}")
            DOPU = None  # Optional: allow OCT-only processing

        # Load the data

        OCT_mcorr = OCT.copy()
        DOPU_mcorr = DOPU.copy() if DOPU is not None else None

        numFrames = OCT_mcorr.shape[2]
        startFrame = round(numFrames / 2)
        axialShift = np.zeros(numFrames)

        # Backward motion correction
        for k in range(startFrame - 1, 0, -1):
            ref_frame = exposure.rescale_intensity(OCT_mcorr[:, :, k + 1])
            curr_frame = exposure.rescale_intensity(OCT_mcorr[:, :, k])

            output = dftregistration(fft2(ref_frame), fft2(curr_frame))
            axialShift[k] = output[2]
            OCT_mcorr[:, :, k] = np.roll(OCT_mcorr[:, :, k], int(round(output[2])), axis=0)
            DOPU_mcorr[:, :, k] = np.roll(DOPU_mcorr[:, :, k], int(round(output[2])), axis=0)

        # Forward motion correction
        for k in range(startFrame + 1, numFrames):
            ref_frame = exposure.rescale_intensity(OCT_mcorr[:, :, k - 1])
            curr_frame = exposure.rescale_intensity(OCT_mcorr[:, :, k])

            output = dftregistration(fft2(ref_frame), fft2(curr_frame))
            axialShift[k] = output[2]
            OCT_mcorr[:, :, k] = np.roll(OCT_mcorr[:, :, k], int(round(output[2])), axis=0)
            DOPU_mcorr[:, :, k] = np.roll(DOPU_mcorr[:, :, k], int(round(output[2])), axis=0)

        # Plot axial shifts
        name, _ = os.path.splitext(oct_file)
        dopu_name, _ = os.path.splitext(dopu_file)
        plt.figure()
        plt.plot(axialShift)
        plt.title('Axial Motion Estimation - File: ' + name)
        plt.xlim([1, 500])
        plt.ylim([-15, 15])
        plt.xlabel('Frame Index')
        plt.ylabel('Axial Shifts (Pixel)')

        # Save the graph to a file
        png_file = os.path.join(filepath, f"{name}_axshift")
        plt.savefig(png_file)
        # Clear the graph buffer
        plt.clf()

        # Save the processed data
        new_filename = os.path.join(filepath, f"{name}_mcorr.mat")
        new_dopu_filename = os.path.join(filepath, f"{dopu_name}_mcorr.mat")
        hdf5storage.savemat(new_filename, {'OCT_mcorr': OCT_mcorr}, format='7.3')
        hdf5storage.savemat(new_dopu_filename, {'DOPU_mcorr': DOPU_mcorr}, format='7.3')

        print(f'global motion correction {idx}/{len(vol_files)} complete')
