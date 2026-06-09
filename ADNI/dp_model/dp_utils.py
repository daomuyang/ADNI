"""Official label encoding utilities (UKBiobank_deep_pretrain)."""

import numpy as np
from scipy.stats import norm


def num2vect(x, bin_range, bin_step, sigma):
    """
    Convert scalar age(s) to soft histogram label(s).

    bin_range: (start, end), bin_step divides range evenly.
    sigma=0 hard index; sigma>0 Gaussian CDF bin probabilities.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if bin_length % bin_step != 0:
        raise ValueError("bin range must be divisible by bin_step")
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step).astype(int)
        return i, bin_centers
    if sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        v = np.zeros((len(x), bin_number))
        for j in range(len(x)):
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                v[j, i] = cdfs[1] - cdfs[0]
        return v, bin_centers
    raise ValueError(f"invalid sigma={sigma}")


def crop_center(data, out_sp):
    """Center crop when in_sp > out_sp (official implementation)."""
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        return data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    if nd == 4:
        return data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    raise ValueError(f"Wrong dimension: {nd}")
