import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d

def mat2gray(img):
    """Normalize the image to the [0, 1] range (like MATLAB's mat2gray)."""
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min)

def imadjust(img, low_in=0.0, high_in=1.0, low_out=0.0, high_out=1.0):
    """
    Perform contrast stretching (like MATLAB's imadjust).
    Adjust intensities from the input range [low_in, high_in]
    to the output range [low_out, high_out].
    """
    img = np.clip((img - low_in) / (high_in - low_in), 0, 1)
    return img * (high_out - low_out) + low_out

def estPhaseOffset(cplxData_A, cplxData_B):
    """
    Combining cplxdata from P and S channels to be fed into the registration algorithm. Phase difference is calculated
    and compensated before summing.
    """
    CompCplx = np.zeros_like(cplxData_A, dtype=np.complex64)

    # Iterate over the third dimension
    for i in range(cplxData_A.shape[2]):
        OCT_P = cplxData_A[:, :, i]
        OCT_S = cplxData_B[:, :, i]

        # Phase correction
        phi = np.angle(np.sum(OCT_S * np.conj(OCT_P), axis=0))
        phi = np.tile(phi, (OCT_S.shape[0], 1))  # Replicate to match dimensions

        # Apply phase correction and combine the channels
        OCT_Cplx = OCT_P + OCT_S * np.exp(-1j * phi)

        # Store the result in the output array
        CompCplx[:, :, i] = OCT_Cplx

    return CompCplx

def separateCplx(CompCplx):
    real = np.real(CompCplx)
    im = np.imag(CompCplx)

    return real, im

def combineCplx(real, im):
    cplxData = real + 1j*im

    return cplxData
def dft_registration(buf1ft, buf2ft):
    nr, nc = buf2ft.shape
    Nr = np.fft.ifftshift(-np.fix(nr / 2).astype(int) + np.arange(nr))
    Nc = np.fft.ifftshift(-np.fix(nc / 2).astype(int) + np.arange(nc))

    # Compute the cross-correlation in the frequency domain
    CC = ifft2(buf1ft * np.conj(buf2ft))
    CCabs = np.abs(CC)

    # Find the location of the maximum in the cross-correlation
    row_shift, col_shift = np.unravel_index(np.argmax(CCabs), CCabs.shape)

    # Adjust shifts to represent relative shifts
    row_shift = Nr[row_shift]
    col_shift = Nc[col_shift]

    CCmax = np.max(CCabs)
    error = 0  # Placeholder for error, as it's not calculated in the provided code
    diffphase = 0  # Placeholder for diffphase, as it's not calculated in the provided code

    output = [error, CCmax, row_shift, col_shift]
    return output


def match_axial(refOCT, tarOCT, numBatch, scale):
    if scale == 1:
        refOCT_log = 20 * np.log10(refOCT)
        tarOCT_log = 20 * np.log10(tarOCT)
    else:
        refOCT_log = refOCT
        tarOCT_log = tarOCT

    numAscans = refOCT_log.shape[1]
    wdtBatch = numAscans // numBatch

    row_shift_all = np.zeros((numAscans, refOCT_log.shape[2]))
    col_shift_all = np.zeros((numAscans, refOCT_log.shape[2]))

    for k in range(refOCT_log.shape[2]):
        row_shift = np.zeros(numBatch)
        col_shift = np.zeros(numBatch)
        idxBatch = np.zeros(numBatch)

        for j in range(numBatch):
            # Extract and adjust the image
            ref_img = refOCT_log[:, wdtBatch * j:wdtBatch * (j + 1), k]
            tar_img = tarOCT_log[:, wdtBatch * j:wdtBatch * (j + 1), k]

            # Normalize the images
            ref_img = (ref_img - np.min(ref_img)) / (np.max(ref_img) - np.min(ref_img))
            tar_img = (tar_img - np.min(tar_img)) / (np.max(tar_img) - np.min(tar_img))

            idxBatch[j] = (wdtBatch * j) + (wdtBatch // 2)  # index of the center frame in each batch

            row_shift[j], col_shift[j] = dft_registration(ref_img, tar_img)

        # Smooth the shifts
        row_shift_smooth = np.round(gaussian_filter(row_shift, sigma=0.5))
        col_shift_smooth = np.round(gaussian_filter(col_shift, sigma=0.5))

        # Interpolate shifts
        interp_func_row = interp1d(idxBatch, row_shift_smooth, bounds_error=False, fill_value="extrapolate")
        interp_func_col = interp1d(idxBatch, col_shift_smooth, bounds_error=False, fill_value="extrapolate")

        row_shift_interp = np.round(interp_func_row(np.arange(numAscans))).astype(int)
        col_shift_interp = np.round(interp_func_col(np.arange(numAscans))).astype(int)

        row_shift_all[:, k] = row_shift_interp
        col_shift_all[:, k] = col_shift_interp

    return row_shift_all, col_shift_all