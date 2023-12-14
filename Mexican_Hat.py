""" This script implements the method of sweeping over the receptive field with
a Mexican hat spot."""

import numpy as np


def Create_mexican_hat(section_size, sigma_sq, black_factor):
    """ Returns a 2D array containing a Mexican hat profile.

    Parameters
    ----------
    section_size : int
        Width/height of the Mexican hat array is 2x*section_size* - 1. Thereby,
        a *section_size*-wide and -high section can be taken that has an
        arbitrary position of the Mexican hat center inside this section.
    sigma_sq : float
        Parameter determining the size of the Mexican hat.
    black_factor : float
        Multiplicative factor applied to the surround of the Mexican hat.

    Returns
    -------
    mexican_hat : ndarray
        Array containing the Mexican hat. Array is normalized to have a peak
        value of 1. Width/height is 2x*section_size* - 1.
    """

    size = 2*section_size-1
    center = size/2 - 0.5
    xx, yy = np.mgrid[:size, :size]
    distance_sq = (xx - center) ** 2 + (yy - center) ** 2
    mexican_hat = (1/(np.pi*sigma_sq)
                   * (1 - 1/2 * (distance_sq/sigma_sq))
                   * np.exp(-distance_sq/(2*sigma_sq)))
    mexican_hat /= np.max(mexican_hat)
    mexican_hat[mexican_hat < 0] *= black_factor
    mexican_hat[mexican_hat < -1] = -1

    return mexican_hat


def Response_map(rgc, hat_sigma_sq, black_factor):
    """ Simulate the response map of the given RGC to a Mexican hat profile.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    hat_sigma_sq : float
        Parameter determining the size of the Mexican hat.
    black_factor : float
        Multiplicative factor applied to the surround of the Mexican hat.

    Returns
    -------
    response_map : ndarray
        2D array containing the responses of *rgc* to Mexican hats centered at
        all positions inside the receptive area. Has size
        *rgc.resolution* x *rgc.resolution*.
    """

    size = rgc.resolution
    mexican_hat = Create_mexican_hat(size, hat_sigma_sq, black_factor)
    response_map = np.empty((size, size))
    for x_pos in range(size):
        for y_pos in range(size):
            stimulus = mexican_hat[size-1-x_pos:2*size-1-x_pos,
                                   size-1-y_pos:2*size-1-y_pos]
            response_map[x_pos, y_pos] = rgc.response_to_flash(stimulus)

    return response_map
