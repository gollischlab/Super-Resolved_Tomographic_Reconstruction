""" This script provides a class for subunit models"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.optimize as opt


###############################################################################
# Dictionary storing the default parameters of the class
###############################################################################
DEFAULT_PARAMS = {"num_subunits" : None,                            # Planned number of subunits of the model. Note that this might be overruled by the subunit scenario. If None, this option has no effect. Can also be a list of two elements, if the scenario 'realistic gauss' is used.
                  "layout_seed" : None,                             # Non-negative seed of the random number generator in the subunit scenario 'realistic gauss'. If None, a random seed is used. Can also be a list of two elements, if the scenario 'realistic gauss' is used.
                  "overlap_factor" : 1.35,                          # Regulates the size of the subunits in the scenario 'realistic gauss' without changing their spacing, thereby adjusting their overlap. Note that the value is not related to any meaningful measure.
                  "irregularity" : 3,                               # Determines how strongly the subunit layout in the scenario 'realistic gauss' deviates from a hexagonal grid.
                  "swap_gauss_for_cosine" : False,                  # If True, subunit layouts which normally use Gaussian subunits, will instead use cosine-shaped subunits.
                  "opposing_polarity" : False,                      # If True and the scenario 'realistic gauss' is used to create two superimposed subunits, the second subunit layout will have Off polarity.
                  "weights_gauss_std" : 0.12,                       # Standard deviation of the Gaussian used to set the subunit weights if they are option 'gauss'. In units of *resolution*, i.e. simulation area size.
                  "poisson_seed" : None,                            # Non-negative seed of the random number generator of the spiking poisson process. If None, a random seed is used.
                  "spiking_coefficient" : 'realistic',              # Defines the coefficient between the RGC response and the expected number of spikes in response. 'realistic' means a coefficient is used that leads to an average of 30 spikes in response to a full-field flash of white.
                  "spiking_base_level" : 0.0}                       # Base spike level of the model. The expected number of spikes in response to a stimulus is increased by this value, i.e. this corresponds to a spontaneous activity or noise level.


###############################################################################
# Helper functions
###############################################################################
def Is_iterable(a):
    """ Check if a variable is iterable, e.g. an array.

    Paramters
    ---------
    a : arbitrary type
        The variable to check.

    Returns
    -------
    bool
        Whether *a* is iterable.
    """

    try:
        _ = iter(a)
        return True
    except TypeError:
        return False


def Gaussian_array(resolution, x0, y0, sigma_x, sigma_y, theta):
    """ Create a Gaussian distribution in an ndarray.

    Gaussian is normalised to a volume of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the array.
    x0 : float
        X-position of the center of the Gaussian.
    y0 : float
        Y-position of the center of the Gaussian.
    sigma_x : float
        Standard deviation in x-direction.
    sigma_y : float
        Standard deviation in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    gaussian : ndarray
        2D array containing the Gaussian.
    """

    xx, yy = np.mgrid[:resolution, :resolution]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    amplitude = 1/(2*np.pi*sigma_x*sigma_y)
    gaussian = amplitude*np.exp(-(a*((xx-x0)**2) + 2*b*(xx-x0)*(yy-y0)
                                  + c*((yy-y0)**2)))

    return gaussian


def Cosine_spot_array(resolution, x0, y0, size_x, size_y, theta):
    """ Create an elliptical spot with a cosine profile in an ndarray.

    Spot is normalised to a volume of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the array.
    x0 : float
        X-position of the center of the Spot.
    y0 : float
        Y-position of the center of the Spot.
    size_x : float
        For convenience, this is not the radius in x-direction of the cosine
        spot, but the standard deviation in x-direction a Gaussian would have
        if it was fitted to the spot. This makes this function more analagous
        in use to *Gaussian_array()*.
    size_y : float
        Same as *size_x*, but in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        2D array containing the spot.
    """

    xx, yy = np.mgrid[:resolution, :resolution]
    xx = xx - x0
    yy = yy - y0
    xx, yy = (np.cos(theta)*xx - np.sin(theta)*yy,
              np.sin(theta)*xx + np.cos(theta)*yy)
    xx /= 2.22*size_x
    yy /= 2.22*size_y
    rr = np.linalg.norm(np.stack([xx, yy]), axis=0)
    in_idxb = (rr <= 1)
    spot = np.zeros((resolution, resolution))
    spot[in_idxb] = np.cos(rr[in_idxb]*(np.pi/2))
    spot /= np.sum(spot)
    return spot


def TwoD_Gaussian(data_tuple, x0, y0, sigma_x, sigma_y, theta):
    """ Calculate the values of a 2D Gaussian with the specified parameters at
    the given positions.

    Gaussian is normalised to a volume of 1.

    Parameters
    ----------
    data_tuple : ndarray
        Locations at which the Gaussian should be evaluated. 2D with first
        column denoting x and second column denoting y.
    x0 : float
        X-position of the center of the Gaussian.
    y0 : float
        Y-position of the center of the Gaussian.
    sigma_x : float
        Standard deviation in x-direction.
    sigma_y : float
        Standard deviation in y-direction.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    ndarray
        1D array of the values of the gaussian at the specified positions.
    """

    x = data_tuple[:, 0]
    y = data_tuple[:, 1]
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    amplitude = 1/(2*np.pi*sigma_x*sigma_y)
    g = amplitude*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

    return g.ravel()


def Gaussian_layout(resolution, num_subunits, rng_seed, overlap_factor,
                    irregularity, swap_gauss_for_cosine,
                    center_and_rotate=False, return_complete_mosaic=False):
    """ Create a mosaic layout of gaussian subunits with variable standard
    deviations and variable rotation angles.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    num_subunits : int
        If not None, specifies the number of subunits.
    rng_seed : int
        Non-negative seed used for randomly generating the subunit layout. If
        None, a random seed is used.
    overlap_factor : float
        Regulates the size of the subunits without changing their spacing,
        thereby adjusting their overlap. Note that the value is not related to
        any meaningful measure.
    irregularity : float
        Determines how strongly the layout deviates from a hexagonal grid.
    swap_gauss_for_cosine : bool
        If True, creates cosine-shaped subunits instead of Gaussians.
    center_and_rotate : bool, optional
        If True, the layout is shifted such that the RF center is in the center
        of the simulation area (RF is assumed to be linear combination of
        subunits), and rotated by a random angle.
    return_complete_mosaic : bool, optional
        If True, an extensive mosaic consisting of 144 subunits covering more
        or less the entire simulation area is returned. *num_subunits* is thus
        not the actual number of subunits anymore, but only determines the size
        of the extensive mosaic.

    Returns
    -------
    subunits : ndarray
        3D array containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    """

    # Setting up random generator
    rng = np.random.default_rng(seed=rng_seed)

    # local constants
    size = 100
    sqrt_num_points = 8
    if num_subunits is None:
        num_subunits = rng.integers(4, 13)
    scale_layout_to_num_subunits = num_subunits # Describes which size equivalent the layout will be scaled to
    xlim = np.array([0, size])  # Describe the size of the area used to create the layout
    ylim = np.array([0, size])

    # Override the number of subunits if requested
    if return_complete_mosaic:
        num_subunits = 144

    # Generating a perturbed hexagonal grid
    xx, yy = np.mgrid[0:size:sqrt_num_points*1j, 0:size:sqrt_num_points*1j]
    points = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
    # If the number of subunits is high, add extra points to avoid problems in
    # the outer regions. Appending extra points here ensures that seeds for
    # smaller layouts still produce the same results as before, and also avoids
    # unnecessary computational cost for small layouts.
    if num_subunits > 16:
        dist = size/(sqrt_num_points-1)
        top = np.column_stack((np.repeat(np.linspace(0, size, sqrt_num_points), 2),
                               np.tile(size + np.array([1, 2]) * dist, sqrt_num_points)))
        bottom = top - [0, size + 3*dist]
        left = np.column_stack((np.repeat(-np.array([1, 2]) * dist, sqrt_num_points+4),
                                np.tile(np.linspace(-2*dist, size + 2*dist, sqrt_num_points+4), 2)))
        right = (left * [-1, 1]) + [size, 0]
        points = np.concatenate([points, top, bottom, left, right])
        xlim = np.array([-2*dist, size + 2*dist], dtype=int)
        ylim = np.array([-2*dist, size + 2*dist], dtype=int)
    points[::2, 0] += size/(sqrt_num_points-1)/2
    points[:, 1] *= np.sqrt(3)/2
    points = points + rng.normal(scale=irregularity, size=points.shape)

    # Calculating the voronoi sets
    xx, yy = np.mgrid[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    pixels = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
    closest = np.empty(pixels.shape[0])
    for counter, pixel in enumerate(pixels):
        closest[counter] = np.argmin(np.sum(np.square(points - pixel), axis=1))
    voronoi = np.reshape(closest, xx.shape)

    # Calculate the center of masses of the voronoi sets
    centers = np.empty_like(points)
    for i in range(points.shape[0]):
        if np.any(voronoi.ravel() == i):
            centers[i] = np.mean(pixels[voronoi.ravel() == i], axis=0)
        else:
            centers[i] = [np.nan, np.nan]

    # Choosing only the N sets that are closest to the screen center
    distances = np.sum(np.square(centers - size/2), axis=1)
    subunits_idxe = np.argsort(distances)[:num_subunits]
    subunits = np.empty((num_subunits, voronoi.shape[0], voronoi.shape[1]))
    for counter, idxe in enumerate(subunits_idxe):
        subunits[counter] = (voronoi == idxe).astype(int)
    centers = centers[subunits_idxe]

    # Fitting Gaussians to the selected Voronoi sets
    gauss_params = np.empty((num_subunits, 5))
    for i in range(num_subunits):
        initial_guess = (centers[i, 0], centers[i, 1], 1, 1, 0)
        if np.max(subunits[i]) > 0:
            try:
                gauss_params[i], _ = opt.curve_fit(TwoD_Gaussian, pixels,
                                                   (subunits[i].ravel()
                                                    / np.sum(subunits[i])),
                                                   p0=initial_guess)
            except RuntimeError:
                gauss_params[i] = np.full(5, np.NaN)
        else:
            gauss_params[i] = np.full(5, np.NaN)

    # Remove unsuccesfully fitted Gaussians
    failed_idxb = np.any(np.isnan(gauss_params), axis=1)
    gauss_params = gauss_params[np.logical_not(failed_idxb)]
    num_subunits -= np.sum(failed_idxb)

    # Adjusting the standard deviation of the Gaussians to be more realistic
    gauss_params[:, 2:4] *= overlap_factor

    # Rescaling everything to a constant RGC size
    gauss_params[:, :2] = size/2 + ((gauss_params[:, :2] - size/2)
                                    * 3 / np.sqrt(scale_layout_to_num_subunits))
    gauss_params[:, 2:4] *= 3 / np.sqrt(scale_layout_to_num_subunits)

    # Rescaling to the resolution of the model
    gauss_params[:, :4] *= resolution/size

    # Centering the RF and rotating the layout
    if center_and_rotate:
        # First determine the RF by estimating it to be a linear combination of
        # the subunits
        subunits = np.zeros((num_subunits, resolution, resolution))
        for i in range(gauss_params.shape[0]):
            if swap_gauss_for_cosine:
                subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
            else:
                subunits[i] = Gaussian_array(resolution, *gauss_params[i])
        rf = np.sum(subunits, axis=0)
        # Then determine the RF center by fitting a Gaussian
        xx, yy = np.mgrid[0:resolution, 0:resolution]
        pixels = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
        initial_guess = (resolution/2, resolution/2,
                         resolution/6, resolution/6, 0)
        rf_center = opt.curve_fit(TwoD_Gaussian, pixels, rf.ravel()/np.sum(rf),
                                  p0=initial_guess)[0][:2]
        # Shifting the RF to the origin for rotation
        gauss_params[:, :2] -= rf_center
        # Rotating the layout by a random angle around the RF center
        rot_angle = rng.uniform(-np.pi, np.pi)
        rot_mat = [[np.cos(rot_angle), -np.sin(rot_angle)],
                   [np.sin(rot_angle), np.cos(rot_angle)]]
        gauss_params[:, 4] -= rot_angle
        gauss_params[:, :2] = np.transpose(np.matmul(rot_mat,
                                                     np.transpose(gauss_params[:, :2])))
        # Shifting the RF to the center of the simulation area
        gauss_params[:, :2] += (resolution-1)/2

    # Creating the subunit array
    subunits = np.zeros((num_subunits, resolution, resolution))
    for i in range(gauss_params.shape[0]):
        if swap_gauss_for_cosine:
            subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
        else:
            subunits[i] = Gaussian_array(resolution, *gauss_params[i])

    return subunits, gauss_params


###############################################################################
# Functions for the subunit scenarios
###############################################################################
def Scenario_basic_gauss(resolution, swap_gauss_for_cosine=False):
    """ Create a 3D-array containing subunits according to scenario
    'basic gauss'.

    'basic gauss' is a 2x2 layout of gaussian subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    swap_gauss_for_cosine : bool, optional
        If True, creates cosine-shaped subunits instead of Gaussians. Default
        is False.

    Returns
    -------
    subunits : ndarray
        3D array containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    """

    pos_3_8 = 3*resolution/8 - 0.5
    pos_5_8 = 5*resolution/8 - 0.5
    sigma = resolution/10
    gauss_params = np.array([[pos_3_8, pos_3_8, sigma, sigma, 0],
                             [pos_3_8, pos_5_8, sigma, sigma, 0],
                             [pos_5_8, pos_3_8, sigma, sigma, 0],
                             [pos_5_8, pos_5_8, sigma, sigma, 0]])
    subunits = np.zeros((gauss_params.shape[0], resolution, resolution))
    for i in range(gauss_params.shape[0]):
        if swap_gauss_for_cosine:
            subunits[i] = Cosine_spot_array(resolution, *gauss_params[i])
        else:
            subunits[i] = Gaussian_array(resolution, *gauss_params[i])

    return subunits, gauss_params


def Scenario_realistic_gauss(resolution, num_subunits, rng_seed,
                             overlap_factor, irregularity,
                             swap_gauss_for_cosine, opposing_polarity=False):
    """ Create a 3D-array containing subunits according to scenario
    'realistic gauss'.

    'realistic gauss' is a mosaic layout of gaussian subunits with variable
    standard deviations and variable rotation angles. Optionally, two
    superimposed layouts can be created, where it is then ensured that both are
    centered and randomly rotated to avoid correlated positioning of subunits.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    num_subunits : int or list of int
        If not None, specifies the number of subunits. Can also be a list of
        two integers, in which case two superimposed layouts with the
        respective numbers of subunits are created.
    rng_seed : int or list of int
        Non-negative seed used for randomly generating the subunit layout. If
        None, a random seed is used. If *num_subunits* is a list, this can also
        be a list specifying the seed for each of the layouts.
    overlap_factor : float
        Regulates the size of the subunits without changing their spacing,
        thereby adjusting their overlap. Note that the value is not related to
        any meaningful measure.
    irregularity : float
        Determines how strongly the layout deviates from a hexagonal grid.
    swap_gauss_for_cosine : bool
        If True, creates cosine-shaped subunits instead of Gaussians.
    opposing_polarity : bool, optional
        If two superimposed layouts are created, i.e. *num_subunits* is a list,
        specifies whether the second layout will contain subunits with Off
        polarity.

    Returns
    -------
    subunits : ndarray
        3D array containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    num_subunits : int or list of int
        Contains the total number of subunits or the number of subunits of each
        of the two superimposed layouts, respectively.
    """

    if not Is_iterable(num_subunits):
        subunits, gauss_params = Gaussian_layout(resolution,
                                                 num_subunits,
                                                 rng_seed,
                                                 overlap_factor,
                                                 irregularity,
                                                 swap_gauss_for_cosine)
        num_subunits = subunits.shape[0]
    else:
        subunits_1, gauss_params_1 = Gaussian_layout(resolution,
                                                     num_subunits[0],
                                                     rng_seed[0] if Is_iterable(rng_seed) else rng_seed,
                                                     overlap_factor,
                                                     irregularity,
                                                     swap_gauss_for_cosine,
                                                     center_and_rotate=True)
        subunits_2, gauss_params_2 = Gaussian_layout(resolution,
                                                     num_subunits[1],
                                                     rng_seed[1] if Is_iterable(rng_seed) else rng_seed,
                                                     overlap_factor,
                                                     irregularity,
                                                     swap_gauss_for_cosine,
                                                     center_and_rotate=True)
        if opposing_polarity:
            subunits_2 *= -1
        subunits = np.concatenate([subunits_1, subunits_2])
        gauss_params = np.concatenate([gauss_params_1, gauss_params_2])
        num_subunits = [subunits_1.shape[0], subunits_2.shape[0]]

    return subunits, gauss_params, num_subunits


def Scenario_photoreceptors(resolution, num_subunits, rng_seed, overlap_factor,
                            irregularity, swap_gauss_for_cosine):
    """ Create 3D-arrays containing subunits and photoreceptors according to
    scenario 'photoreceptors'.

    'photoreceptors' is an LNLNLN model where the subunits receive input from
    another layer that represents the photoreceptors. Both these layers are
    mosaics following 'realistic gauss', with the photoreceptor Gaussians
    having on average half the diameter of the subunit Gaussians, and the
    connection weight being given by the weight of the subunit Gaussian at the
    location of a photoreceptor Gaussian. Cannot be combined with temporal
    dynamics.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    num_subunits : int
        If not None, specifies the number of subunits.
    rng_seed : int or list of int
        Non-negative seeds used for randomly generating the subunit and
        photoreceptor layouts. If None, random seeds are used.
    overlap_factor : float
        Regulates the size of the subunits and photoreceptors without changing
        their spacing, thereby adjusting their overlap. Note that the value is
        not related to any meaningful measure.
    irregularity : float
        Determines how strongly the layouts deviate from a hexagonal grid.
    swap_gauss_for_cosine : bool
        If True, creates cosine-shaped subunits and photoreceptors instead of
        Gaussians.

    Returns
    -------
    subunits : ndarray
        3D array containing the subunits.
    subunit_params : ndarray
        2D array containing parameters of the gaussian subunits. First index
        denotes subunit, second denotes: x-position, y-position, sigma_x,
        sigma_y, angle (radians).
    num_subunits : int
        Contains the total number of subunits.
    photoreceptors : ndarray
        3D array containing the photoreceptors.
    photoreceptor_params : ndarray
        2D array containing parameters of the gaussian photoreceptors. Like
        *subunit_params*.
    num_photoreceptors : int
        Contains the total number of photoreceptors.
    photoreceptor_weights : ndarray
        2D array containing the connection weights between subunits and
        photoreceptors. First dimension is subunit, second is photoreceptor.
        Weights to each subunit are normalised to a sum of one.
    photoreceptor_nl : function
        Function that computes the nonlinearity of the photoreceptor output.
    """

    # First create the photoreceptor layout
    photoreceptors, photoreceptor_params = Gaussian_layout(resolution,
                                                           4*num_subunits,
                                                           rng_seed[1] if Is_iterable(rng_seed) else rng_seed,
                                                           overlap_factor,
                                                           irregularity,
                                                           swap_gauss_for_cosine,
                                                           return_complete_mosaic=True)
    num_photoreceptors = photoreceptors.shape[0]

    # Next create the subunit layout
    subunits, subunit_params = Gaussian_layout(resolution,
                                               num_subunits,
                                               rng_seed[0] if Is_iterable(rng_seed) else rng_seed,
                                               overlap_factor,
                                               irregularity,
                                               swap_gauss_for_cosine)
    num_subunits = subunits.shape[0]

    # Compute the connection weights
    photoreceptor_weights = np.empty((num_subunits, num_photoreceptors))
    for i, params in enumerate(subunit_params):
        photoreceptor_weights[i] = TwoD_Gaussian(photoreceptor_params[:, :2],
                                                 subunit_params[i, 0],
                                                 subunit_params[i, 1],
                                                 subunit_params[i, 2],
                                                 subunit_params[i, 3],
                                                 subunit_params[i, 4])

    # Cut all connections to photoreceptors outside the 1.5-sigma ellipse of
    # a subunit, by finding out the weight threshold for each subunit
    amplitudes = 1 / (2*np.pi*subunit_params[:, 2] * subunit_params[:, 3])
    thresholds = np.exp(-(1.5**2)/2) * amplitudes
    within_idxb = (photoreceptor_weights > np.transpose([thresholds]))
    photoreceptor_weights[np.logical_not(within_idxb)] = 0

    # Remove all photoreceptors outside all 1.5-sigma ellipses
    within_idxb = np.any(within_idxb, axis=0)
    photoreceptors = photoreceptors[within_idxb]
    photoreceptor_params = photoreceptor_params[within_idxb]
    num_photoreceptors = np.sum(within_idxb)
    photoreceptor_weights = photoreceptor_weights[:, within_idxb]

    # Normalize the weights to each subunit to a sum of one
    photoreceptor_weights /= np.transpose([np.sum(photoreceptor_weights,
                                                  axis=1)])

    return (subunits, subunit_params, num_subunits,
            photoreceptors, photoreceptor_params, num_photoreceptors,
            photoreceptor_weights, Photoreceptor_nl_linear_linear)


###############################################################################
# Functions for the photoreceptor nonlinearity
###############################################################################
def Photoreceptor_nl_linear_linear(signal):
    """ Linear-linear nonlinearity for photoreceptors.

    Parameters
    ----------
    signal : ndarray
        Contains all photoreceptor signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = signal
    response[signal < 0] *= 0.5

    return response


###############################################################################
# Functions for the subunit nonlinearity
###############################################################################
def Subunit_nl_threshold_quadratic(signal):
    """ Threshold quadratic nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = np.square(signal[signal > 0])

    return response


def Subunit_nl_threshold_linear(signal):
    """ Threshold linear (relu) nonlinearity for subunits.

    Parameters
    ----------
    signal : ndarray
        Contains all subunit signals.

    Returns
    -------
    response : ndarray
        Contains the result of the nonlinearity applied elementwise to
        *signal*.
    """

    response = np.zeros_like(signal)
    response[signal > 0] = signal[signal > 0]

    return response


###############################################################################
# Functions for the connection weights of subunits to the RGC
###############################################################################
def Weights_equal(num_subunits):
    """ Create equal connections weights between subunits and RGC.

    Weights are normalised to have a sum of 1.

    Parameters
    ----------
    num_subunits : int
        Number of subunits.

    Returns
    -------
    weights : ndarray
        Contains the connection weights.
    """

    return np.ones(num_subunits) / num_subunits


def Weights_gauss(resolution, subunit_locs, gauss_std):
    """ Create connection weights between subunits and RGC according to a
    Gaussian located at the center of the receptive area.

    Weights are normalised to have a sum of 1.

    Parameters
    ----------
    resolution : int
        Width and height of the square receptive area of the RGC in pixels.
    subunit_locs : int
        Locations of the subunits.
    gauss_std : float
        Standard deviation of the Gaussian used to set the subunit weights in
        units of *resolution*, i.e. simulation area size.

    Returns
    -------
    weights : ndarray
        Contains the connection weights.
    """

    weights = TwoD_Gaussian(subunit_locs, (resolution-1)/2, (resolution-1)/2,
                            gauss_std*resolution, gauss_std*resolution, 0)

    return weights / np.sum(weights)


###############################################################################
# Functions for the output nonlinearity
###############################################################################
def Output_nl_none(signal):
    """ No output nonlinearity.

    Parameters
    ----------
    signal : float
        Contains the RGC signal.

    Returns
    -------
    response : float
        Contains the result of the nonlinearity applied to *signal*.
    """

    return signal


###############################################################################
# Functions for the spiking process
###############################################################################
def Spiking_none(coefficient, shift, base):
    """ No spiking process. Converts the input into the expected number of
    spikes, giving only the statistically expected spike count, not an actual
    spike count.

    Parameters
    ----------
    coefficient : float
        Coefficient between *response* and resulting expected spike count.
    shift : float
        Shift to be added to *response* before multiplication with
        *coefficient*. Should be used to prevent negative expected spike
        counts. Thus corresponds to the response to full-field black.
    base : float
        Base level spike count that is added to *response* after multiplication
        with *coefficient*. Could technically also be included in *shift*, but
        this parameter makes it more convenient to simulate spontaneous
        activity. Should be larger than or equal to zero.

    Returns
    -------
    Spikes : function
        Function that returns the statistically expected number of spikes for
        a given response.

        Parameters

        response : float
            Contains the RGC response.
        Returns

        float
            The expected spike count resulting from *response*.
    """

    def Expected_spikes(response):
        return coefficient * (response + shift) + base

    return Expected_spikes


def Spiking_poisson(coefficient, shift, rng_seed, base):
    """ Poisson spiking process.

    Parameters
    ----------
    coefficient : float
        Coefficient between *response* and expected value of spikes.
    shift : float
        Shift to be added to *response* before multiplication with
        *coefficient*. Should be used to prevent negative Poisson rates.
        Thus corresponds to the response to full-field black.
    rng_seed : int
        Non-negative seed used for randomly generating the spike number in the
        poisson process. If None, a random seed is used.
    base : float
        Base level firing rate that is added to *response* after multiplication
        with *coefficient*. Could technically also be included in *shift*, but
        this parameter makes it more convenient to simulate spontaneous
        activity. Should be larger than or equal to zero.

    Returns
    -------
    Spikes : function
        Function that returns the number of spikes for a given response.

        Parameters

        response : float
            Contains the RGC response.
        Returns

        float
            The result of the spiking process applied to *response*.
    """

    rng = np.random.default_rng(seed=rng_seed)

    def Spikes(response):
        return rng.poisson(coefficient * (response + shift) + base)

    return Spikes


###############################################################################
# Subunit Class
###############################################################################
class Subunit_Model:
    """ This class implements a model of a retinal ganglion cell (RGC) based
    on an LNLN structure with linear subunits, a nonlinearity, a linear
    ganglion cell and an output nonlinearity. It can be used to simulate
    responses to arbitrary spatial or spatiotemporal stimuli. Most stages of
    the model are not required and have multiple options, such that the desired
    model can be set up in a sandbox-principle.

    Attributes
    ----------
    resolution : int
        Edge length of the simulation area in pixels.
    scenario : string
        Name of the subunit layout scenario.
    subunit_nonlinearity : string
        Name of the subunit nonlinearity.
    subunit_weights : string
        Name of the choice for the weights of the subunits.
    rgc_nonlinearity : string
        Name of the RGC/output nonlinearity.
    rgc_spiking : string
        Name of the spike generation process.
    num_subunits : int or list
        Current number of subunits of the model. In contrast to the keyword
        argument *num_subunits* (stored in *params*), the attribute always
        contains the correct number and thus also never None. If the scenario
        is 'realistic gauss', this can also be a list of two elements.
    params : dict
        Contains the parameters of the model object. Refer to the global
        variable *DEFAULT_PARAMS* for more info.

    Methods
    -------
    set_subunits(scenario, **kwargs)
        Set the subunit scenario.
    set_temporal_filter(length=21, mean_1=3, sigma_1=1.5,
                        amplitude_2=-0.1, mean_2=7, sigma_2=3,
                        flip_on_off=False)
        Set the temporal filter of the subunits.
    set_subunit_nl(subunit_nonlinearity)
        Set the subunit nonlinearity.
    set_weights(subunit_weights, **kwargs)
        Set the subunit to RGC connection weights.
    set_output_nl(rgc_nonlinearity)
        Set the RGC output nonlinearity.
    set_spiking(rgc_spiking, **kwargs)
        Set the spiking process.
    clear_history()
        Clears all stimulus-dependent history of the model.
    get_receptive_field()
        Compute the receptive field of the model.
    plot_subunit_ellipses(savepath=None)
        Plot the subunit ellipses in one plot.
    plot_receptive_field(savepath=None)
        Plot the receptive field of the model.
    plot_temporal_filter(savepath=None)
        Plot the temporal filter of the subunits.
    response_to_flash(image)
        Calculate the response of the model to a flash of the stimulus.
    response_to_frame(frame)
        Calculate the response of the model during the next frame of the
        stimulus.

    Parameters for initialization
    -----------------------------
    resolution : int, optional
        Width and height in pixels of the square area in which the receptive
        field of the RGC lies. Default is 40. From *resolution* a physical
        pixel size can be deduced using the 1.5-sigma ellipse diameter for
        marmoset Off parasol cells of about 120 microns and assuming that the
        other parameters that influence the size of the modelled receptive
        field are left at their defaults. *resolution* 120 would correspond to
        pixels of 2.5 microns, *resolution* 40 would correspond to 7.5 micron
        pixels, and *resolution* 20 to 15 microns.
    scenario : string, optional
        Defines the subunit layout. Note that some options overrule the
        keyword argument *num_subunits*. Options:
            'basic gauss': 2x2 layout of Gaussian subunits. Gaussians are not
            truncated, so they do slightly overlap.

            'realistic gauss' (default): is a mosaic layout of gaussian
            subunits with variable standard deviations and variable rotation
            angles. Optionally, two superimposed layouts can be created, where
            it is then ensured that both are centered and randomly rotated to
            avoid correlated positioning of subunits.

            'photoreceptors': Creates an LNLNLN model where the subunits
            receive input from another layer that represents the
            photoreceptors. Both these layers are mosaics following 'realistic
            gauss', with the photoreceptor Gaussians having on average half the
            diameter of the subunit Gaussians, and the connection weight being
            given by the weight of the subunit Gaussian at the location of a
            photoreceptor Gaussian. Cannot be combined with temporal dynamics.
    subunit_nonlinearity : string, optional
        Nonlinearity of the subunits. Options:
            'threshold-linear': relu (default).
            
            'threshold-quadratic'.
    subunit_weights : string, optional
        Weights connecting the subunits with the RGC. Options:
            'equal' (default): All subunits have equal weights.

            'gauss': Weights correspond to a 2D Gaussian located at the
            receptive area's center.
    rgc_nonlinearity : string, optional
        Output nonlinearity of the RGC. Options:
            None (default).
    rgc_spiking : string, optional
        Spiking process of the RGC. Options:
            None (default): No random spiking process, model outputs firing
            rate.

            'poisson': Spiking via Poisson distribution. Model outputs
            number of spikes.
    **kwargs
        Additional keyword arguments. Check the global variable
        *DEFAULT_PARAMS* for more information.
    """


    def __init__(self, resolution=40,
                 scenario='realistic gauss',
                 subunit_nonlinearity='threshold-linear',
                 subunit_weights='equal',
                 rgc_nonlinearity=None,
                 rgc_spiking=None,
                 **kwargs):
        """ Create an RGC model containing subunits.

        Parameters
        ----------
        resolution : int, optional
            Width and height in pixels of the square area in which the receptive
            field of the RGC lies. Default is 40. From *resolution* a physical
            pixel size can be deduced using the 1.5-sigma ellipse diameter for
            marmoset Off parasol cells of about 120 microns and assuming that the
            other parameters that influence the size of the modelled receptive
            field are left at their defaults. *resolution* 120 would correspond to
            pixels of 2.5 microns, *resolution* 40 would correspond to 7.5 micron
            pixels, and *resolution* 20 to 15 microns.
        scenario : string, optional
            Defines the subunit layout. Note that some options overrule the
            keyword argument *num_subunits*. Options:
                'basic gauss': 2x2 layout of Gaussian subunits. Gaussians are not
                truncated, so they do slightly overlap.

                'realistic gauss' (default): is a mosaic layout of gaussian
                subunits with variable standard deviations and variable rotation
                angles. Optionally, two superimposed layouts can be created,
                where it is then ensured that both are centered and randomly
                rotated to avoid correlated positioning of subunits.

                'photoreceptors': Creates an LNLNLN model where the subunits
                receive input from another layer that represents the
                photoreceptors. Both these layers are mosaics following
                'realistic gauss', with the photoreceptor Gaussians having on
                average half the diameter of the subunit Gaussians, and the
                connection weight being given by the weight of the subunit
                Gaussian at the location of a photoreceptor Gaussian. Cannot be
                combined with temporal dynamics.
        subunit_nonlinearity : string, optional
            Nonlinearity of the subunits. Options:
            'threshold-linear': relu (default).
            
            'threshold-quadratic'.
        subunit_weights : string, optional
            Weights connecting the subunits with the RGC. Options:
                'equal' (default): All subunits have equal weights.
    
                'gauss': Weights correspond to a 2D Gaussian located at the
                receptive area's center.
        rgc_nonlinearity : string, optional
            Output nonlinearity of the RGC. Options:
                None (default).
        rgc_spiking : string, optional
            Spiking process of the RGC. Options:
                None (default): No random spiking process, model outputs firing
                rate.
    
                'poisson': Spiking via Poisson distribution. Model outputs
                number of spikes.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        # Store status of initialization
        self.initialized = False

        # Preprocessing parameters
        self.resolution = resolution
        self.scenario = scenario
        self.params = DEFAULT_PARAMS.copy()
        self.params.update(kwargs)

        # Setting up the subunit scenario
        self.set_subunits(scenario)

        # Setting up the temporal filter of the subunits
        self.set_temporal_filter()

        # Defining the subunit nonlinearity
        self.set_subunit_nl(subunit_nonlinearity)

        # Defining the subunit to RGC connection weights
        self.set_weights(subunit_weights)

        # Defining the RGC output nonlinearity
        self.set_output_nl(rgc_nonlinearity)

        # Defining the spiking process
        self.set_spiking(rgc_spiking)

        # Initialize variables that store stimulus-dependent history
        self.clear_history()

        # Initialization completed
        self.initialized = True


    def set_subunits(self, scenario, **kwargs):
        """ Set the subunit scenario.

        Parameters
        ----------
        scenario : string
            Defines the subunit layout. Note that some options overrule the
            keyword argument *num_subunits*. Options:
                'basic gauss': 2x2 layout of Gaussian subunits. Gaussians are
                not truncated, so they do slightly overlap.

                'realistic gauss' (default): is a mosaic layout of gaussian
                subunits with variable standard deviations and variable rotation
                angles. Optionally, two superimposed layouts can be created,
                where it is then ensured that both are centered and randomly
                rotated to avoid correlated positioning of subunits.

                'photoreceptors': Creates an LNLNLN model where the subunits
                receive input from another layer that represents the
                photoreceptors. Both these layers are mosaics following
                'realistic gauss', with the photoreceptor Gaussians having on
                average half the diameter of the subunit Gaussians, and the
                connection weight being given by the weight of the subunit
                Gaussian at the location of a photoreceptor Gaussian. Cannot be
                combined with temporal dynamics.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        # Setting up the subunit scenario
        if scenario == 'basic gauss':
            self.subunits, self.subunit_params = Scenario_basic_gauss(self.resolution,
                                                                      self.params['swap_gauss_for_cosine'])
        elif scenario == 'realistic gauss':
            temp = Scenario_realistic_gauss(self.resolution,
                                            self.params['num_subunits'],
                                            self.params['layout_seed'],
                                            self.params['overlap_factor'],
                                            self.params['irregularity'],
                                            self.params['swap_gauss_for_cosine'],
                                            self.params['opposing_polarity'])
            self.subunits, self.subunit_params, self.num_subunits = temp
        elif scenario == 'photoreceptors':
            temp = Scenario_photoreceptors(self.resolution,
                                           self.params['num_subunits'],
                                           self.params['layout_seed'],
                                           self.params['overlap_factor'],
                                           self.params['irregularity'],
                                           self.params['swap_gauss_for_cosine'])
            (self.subunits, self.subunit_params, self.num_subunits,
             self.photoreceptors, self.photoreceptor_params,
             self.num_photoreceptors, self.photoreceptor_weights,
             self.photoreceptor_nl) = temp

        if scenario != 'realistic gauss':
            self.num_subunits = self.subunits.shape[0]

        # If this function has been used to change (i.e. not initialize) the
        # subunits, then the history of subunit views has to be cleared
        if self.initialized:
            self.clear_history()


    def set_temporal_filter(self, length=21, mean_1=3, sigma_1=1.5,
                            amplitude_2=-0.1, mean_2=7, sigma_2=3,
                            flip_on_off=False):
        """ Set the temporal filter of the subunits.

        First value in filter is weight of current frame, each value
        corresponds to one frame. Filter is generated from difference (sum to
        be precise) of Gaussians. Filter is normalized such that method
        *response_to_flash()* generates the same response as a nine frame long
        presentation via method *response_to_frame()*.

        Parameters
        ----------
        length : int, optional
            Length of the temporal filter in frames.
        mean_1 : float, optional
            Mean of the first Gaussian.
        sigma_1 : float, optional
            Standard deviation of the first Gaussian.
        amplitude_2 : float, optional
            Amplitude of the second Gaussian relative to the first Gaussian's
            amplitude.
        mean_2 : float, optional
            Mean of the second Gaussian.
        sigma_2 : float, optional
            Standard deviation of the second Gaussian.
        flip_on_off : bool, optional
            If True, the polarity of the temporal filter is flipped, thereby
            changing an On/Off filter to an Off/On filter.
        """

        # Define the filter
        x = np.arange(length)
        self.temporal_filter = (np.exp(-1/2*np.square((x - mean_1)/sigma_1))
                                + amplitude_2
                                * np.exp(-1/2*np.square((x - mean_2)/sigma_2)))

        # Normalize the filter with a 9 frame-long white screen
        conv = np.convolve(self.temporal_filter, np.ones(9))
        self.temporal_filter /= np.sum(conv[conv > 0])

        # Potentially adjust the polarity
        if flip_on_off:
            self.temporal_filter *= -1

        # If this function has been used to change (i.e. not initialize) the
        # filter, then the history of subunit views has to be cleared
        if self.initialized:
            self.clear_history()


    def set_subunit_nl(self, subunit_nonlinearity):
        """ Set the subunit nonlinearity.

        Parameters
        ----------
        subunit_nonlinearity : string
            Nonlinearity of the subunits. Options:
                'threshold-quadratic'.

                'threshold-linear': relu.
        """

        if subunit_nonlinearity == 'threshold-quadratic':
            self.subunit_nl = Subunit_nl_threshold_quadratic
        elif subunit_nonlinearity == 'threshold-linear':
            self.subunit_nl = Subunit_nl_threshold_linear


    def set_weights(self, subunit_weights, **kwargs):
        """ Set the subunit to RGC connection weights.

        Parameters
        ----------
        subunit_weights : string
            Weights connecting the subunits with the RGC. Options:
                'equal': All subunits have equal weights.

                'gauss': Weights correspond to a 2D Gaussian located at the
                receptive area's center.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        if subunit_weights == 'equal':
            if not Is_iterable(self.num_subunits):
                self.weights = Weights_equal(self.num_subunits)
            else:
                self.weights = 0.5 * np.concatenate([Weights_equal(self.num_subunits[0]),
                                                     Weights_equal(self.num_subunits[1])])
        elif subunit_weights == 'gauss':
            if not Is_iterable(self.num_subunits):
                self.weights = Weights_gauss(self.resolution,
                                             self.subunit_params[:, :2],
                                             self.params['weights_gauss_std'])
            else:
                self.weights = 0.5 * np.concatenate([Weights_gauss(self.resolution,
                                                                   self.subunit_params[:self.num_subunits[0], :2],
                                                                   self.params['weights_gauss_std']),
                                                     Weights_gauss(self.resolution,
                                                                   self.subunit_params[self.num_subunits[0]:, :2],
                                                                   self.params['weights_gauss_std'])])


    def set_output_nl(self, rgc_nonlinearity):
        """ Set the RGC output nonlinearity.

        Parameters
        ----------
        rgc_nonlinearity : string
            Output nonlinearity of the RGC. Options:
                None.
        """

        if rgc_nonlinearity is None:
            self.output_nl = Output_nl_none


    def set_spiking(self, rgc_spiking, **kwargs):
        """ Set the spiking process.

        Parameters
        ----------
        rgc_spiking : string
            Spiking process of the RGC. Options:
                None: No random spiking process, model outputs expected value
                for spike count.

                'poisson': Spiking via Poisson distribution. Model outputs
                actual number of spikes.
        **kwargs
            Additional keyword arguments. Check the global variable
            *DEFAULT_PARAMS* for more information.
        """

        self.params.update(kwargs)

        # First set up a dummie spike process and find out how negative the
        # response to full-field black is so that negative Poisson rates can be
        # prevented
        self.spiking = Spiking_none(1, 0, 0)
        stim = -np.ones((self.resolution, self.resolution))
        shift = np.abs(min(self.response_to_flash(stim), 0))
        # Next find out the response to a white flash in order to calibrate
        # the number of spikes
        if self.params['spiking_coefficient'] == 'realistic':
            stim = np.ones((self.resolution, self.resolution))
            resp = self.response_to_flash(stim)
            coefficient = 30 / (resp + shift)
        else:
            coefficient = self.params['spiking_coefficient']

        # Now set the spiking process with the parameters that have been
        # determined previously
        if rgc_spiking == 'poisson':
            self.spiking = Spiking_poisson(coefficient, shift,
                                           self.params['poisson_seed'],
                                           self.params['spiking_base_level'])
        else:
            self.spiking = Spiking_none(coefficient, shift,
                                        self.params['spiking_base_level'])


    def clear_history(self):
        """ Clears all stimulus-dependent history of the model."""

        # This variable stores the multiplication of the subunits with the
        # stimulus history to avoid computing this multiple times.
        self.sub_views_history = np.zeros((self.subunits.shape[0],
                                           self.temporal_filter.size))


    def get_receptive_field(self):
        """ Compute the receptive field of the model.

        Receptive field is computed by flashing a white pixel at each location
        and taking the spike-free response as the receptive field strength at
        that location.

        Returns
        -------
        ndarray
            2D array containing the receptive field. Shape is *self.resolution*
            x *self.resolution*.
        ndarray
            1D array containing the parameters of a Gaussian fitted to the RF
            in the order x-position, y-position, sigma_x, sigma_y, angle
            (radians).
        """

        # Spiking is deactivated to get a noise-free RF
        spiking = self.spiking
        self.set_spiking(None)
        # Test RF
        rf = np.empty((self.resolution, self.resolution))
        for x in range(self.resolution):
            for y in range(self.resolution):
                stimulus = np.zeros((self.resolution, self.resolution))
                stimulus[x, y] = 1
                rf[x, y] = self.response_to_flash(stimulus)
        # Baseline activity should not influence the RF
        rf -= self.params['spiking_base_level']
        # Fit ellipse to RF
        xx, yy = np.mgrid[0:self.resolution, 0:self.resolution]
        pixels = np.transpose(np.vstack([xx.ravel(), yy.ravel()]))
        initial_guess = (self.resolution/2, self.resolution/2,
                         self.resolution/6, self.resolution/6, 0)
        rf_params, _ = opt.curve_fit(TwoD_Gaussian, pixels,
                                     rf.ravel()/np.sum(rf), p0=initial_guess)
        # Reset spiking
        self.spiking = spiking

        return rf, rf_params


    def plot_subunit_ellipses(self, savepath=None):
        """ Plot the 1.5-sigma subunit ellipses in one plot.

        Only available if subunits are Gaussians.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.suptitle("Subunit layout")
        ax.set_xlim((0, self.resolution-1))
        ax.set_ylim((0, self.resolution-1))
        if Is_iterable(self.num_subunits):
            colors = np.repeat(['black', 'lawngreen'], self.num_subunits)
        else:
            colors = np.repeat(['black'], self.num_subunits)
        for i in range(self.subunits.shape[0]):
            rf_ellipse = Ellipse(self.subunit_params[i, 0:2],
                                 self.subunit_params[i, 2]*2*1.5,
                                 self.subunit_params[i, 3]*2*1.5,
                                 -self.subunit_params[i, 4]*180/np.pi,
                                 fill=False, color=colors[i])
            ax.add_patch(rf_ellipse)
        _, rf_params = self.get_receptive_field()
        rf_ellipse = Ellipse(rf_params[0:2],
                             rf_params[2]*2*1.5,
                             rf_params[3]*2*1.5,
                             -rf_params[4]*180/np.pi,
                             fill=False, color='red')
        ax.add_patch(rf_ellipse)

        if self.scenario == 'photoreceptors':
            for p in range(self.photoreceptors.shape[0]):
                rf_ellipse = Ellipse(self.photoreceptor_params[p, 0:2],
                                     self.photoreceptor_params[p, 2]*2*1.5,
                                     self.photoreceptor_params[p, 3]*2*1.5,
                                     -self.photoreceptor_params[p, 4]*180/np.pi,
                                     fill=False, color='blue')
                ax.add_patch(rf_ellipse)
                for s in range(self.subunits.shape[0]):
                    ax.plot([self.subunit_params[s, 0],
                             self.photoreceptor_params[p, 0]],
                            [self.subunit_params[s, 1],
                             self.photoreceptor_params[p, 1]],
                            color='green',
                            lw=10*self.photoreceptor_weights[s, p])

        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_receptive_field(self, savepath=None):
        """ Plot the receptive field of the model.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        rf, rf_params = self.get_receptive_field()
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.suptitle("Receptive field of RGC")
        ax.imshow(np.transpose(rf), origin='lower', cmap='gray_r')
        rf_ellipse = Ellipse(rf_params[0:2],
                             rf_params[2]*2*1.5,
                             rf_params[3]*2*1.5,
                             -rf_params[4]*180/np.pi,
                             fill=False, color='red')
        ax.add_patch(rf_ellipse)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def plot_temporal_filter(self, savepath=None):
        """ Plot the temporal filter of the subunits.

        Parameters
        ----------
        savepath : string, optional
            If provided, the plot is not shown in terminal but saved at the
            given location.
        """

        t_filter = self.temporal_filter
        fig, ax = plt.subplots()
        fig.suptitle("Temporal filter of subunits")
        ax.set_ylabel("Stimulus")
        ax.set_xlabel("Frames in the past")
        ax.set_ylim(-1.1*np.max(np.abs(t_filter)),
                    1.1*np.max(np.abs(t_filter)))
        ax.set_xlim(t_filter.size - 1, 0)
        ax.grid()
        ax.plot(t_filter)
        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, dpi=300)
        plt.clf()
        plt.close('all')


    def response_to_flash(self, image):
        """ Calculate the response of the model to a flash of the stimulus.

        Parameters
        ----------
        image : ndarray
            Stimulus to be flashed. Must have the same size as given in
            initialization of the model. Entry values -1 mean full black, 0
            means grey, +1 means full white.

        Returns
        -------
        response
            Response strength to the stimulus. Proportional to firing rate.
        """

        if self.scenario == 'photoreceptors':
            photo_sums = np.sum(np.multiply(self.photoreceptors, image),
                                axis=(1, 2))
            photo_responses = self.photoreceptor_nl(photo_sums)
            sub_sums = np.sum(np.multiply(self.photoreceptor_weights,
                                          photo_responses),
                              axis=1)
        else:
            sub_sums = np.sum(np.multiply(self.subunits, image), axis=(1, 2))
        sub_responses = self.subunit_nl(sub_sums)
        rgc_sum = np.sum(sub_responses * self.weights)
        rgc_response = self.output_nl(rgc_sum)
        spikes = self.spiking(rgc_response)

        return spikes


    def response_to_frame(self, frame):
        """ Calculate the response of the model during the next frame of the
        stimulus.

        Grey screen is assumed at the beginning of the simulation. Does not
        work properly when scenario is 'photoreceptors'.

        Parameters
        ----------
        frame : ndarray
            Stimulus in the next frame. Must have the same size as given in
            initialization of the model. Entry values -1 mean full black, 0
            means grey, +1 means full white.

        Returns
        -------
        response
            Response strength during this frame. Proportional to firing rate.
        """

        # Compute subunit views on new frame
        sub_views = np.sum(np.multiply(self.subunits, frame), axis=(1, 2))
        # Update the sub_views history
        self.sub_views_history[:, 1:] = self.sub_views_history[:, :-1]
        self.sub_views_history[:, 0] = sub_views
        # Apply temporal filter
        sub_sums = np.sum(np.multiply(self.sub_views_history,
                                      self.temporal_filter),
                          axis=1)
        # Apply subunit nonlinearity
        sub_responses = self.subunit_nl(sub_sums)
        # Sum subunit responses
        rgc_sum = np.sum(sub_responses * self.weights)
        # Apply RGC nonlinearity
        rgc_response = self.output_nl(rgc_sum)
        # Apply spiking process
        spikes = self.spiking(rgc_response)

        return spikes


###############################################################################
# Testing
###############################################################################
if __name__ == '__main__':
    # Instantiation of a Subunit_Model object. For reference, all available
    # parameters are listed in this command. Check the documentation for infos
    # about them. Additionally, there are keyword arguments. Check the global
    # variable *DEFAULT_PARAMS* for more info.
    rgc = Subunit_Model(resolution=40, scenario='realistic gauss',
                        subunit_nonlinearity='threshold-linear',
                        subunit_weights='equal', rgc_nonlinearity=None,
                        rgc_spiking=None)
    # Plotting the receptive field of the model.
    rgc.plot_receptive_field("Receptive field")
    # Plotting the 1.5-sigma subunit ellipses of the model.
    rgc.plot_subunit_ellipses("Subunits")
