""" This script implements the super-resolved tomographic reconstruction
technique based on presenting Ricker stripes to detect subunits in a retinal
ganglion cell model."""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from skimage.transform import iradon
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
import pathlib
import Subunit_Model


###############################################################################
# Global parameters
###############################################################################
SCENARIO = 'realistic gauss'            # Model subunit layout type.
NUM_SUBUNITS = 10                       # Model number of subunits.
LAYOUT_SEED = None                      # Model random seed used for generating the subunit layout (and potentially number).
SUBUNIT_NL = 'threshold-linear'         # Model nonlinearity of the subunits.
SYNAPTIC_WEIGHTS = 'equal'              # Model weights of the individual subunits.
RGC_NL = None                           # Model output nonlinearity.
POISSON_SEED = None                     # Model random seed used for generating spikes in the poisson process.
RESOLUTION = 40                         # Width and height of the simulated area in pixels.
NUM_POSITIONS = 60                      # Number of stripe positions.
NUM_ANGLES = 36                         # Number of stripe angles.
HALF_W = 2.5                            # Half width of the Ricker stripe profile.
SURROUND_FACTOR = 2.5                   # Factor that strengthens the suppressive surround of the stimulus.
SMOOTHING = (0.025,                     # Gaussian sigma for smoothing the sinograms. First value is in position-direction and relative to simulation area size,
             5.0)                       # second is in angle-direction and in units of degrees.


###############################################################################
# Plotting functions
###############################################################################
def Plot_stimulus(resolution, half_w, surround_factor, num_positions,
                  savepath=None):
    """ Plot an examplary stimulus with the given characteristics.

    Parameters
    ----------
    resolution : int
        Resolution/size of the stimulus as measured along an edge.
    half_w : float
        Half width of the Ricker stripe profile.
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Ricker stripe.
    num_positions : int
        Number of total stripe positions. Needed to define the rotation axis.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    """

    stimulus = Create_stimulus(resolution, half_w, 0, (resolution-1)/2,
                               surround_factor, num_positions)
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Stimulus")
    ax.imshow(np.transpose(stimulus), origin='lower',
              cmap='gray', vmin=-1, vmax=1)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


def Plot_sinogram(sinogram, savepath=None):
    """ Plot a sinogram.

    Parameters
    ----------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        stripe, second denotes angle of the stripe.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    """

    num_angles = sinogram.shape[1]
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Sinogram")
    ax.set_xlabel("Position")
    ax.set_ylabel("Angle")
    ax.set_yticks([0, num_angles/4, num_angles/2, num_angles*3/4])
    ax.set_yticklabels(["0°", "45°", "90°", "135°"],
                       rotation='vertical', va='center')
    im = ax.imshow(np.transpose(sinogram), origin='lower', aspect='auto',
                   cmap='Greys', vmin=0, vmax=np.max(sinogram))
    cax = inset_axes(ax,
                     width="2%",
                     height="100%",
                     loc='lower left',
                     bbox_to_anchor=(1.02, 0., 1, 1),
                     bbox_transform=ax.transAxes,
                     borderpad=0)
    cbar = fig.colorbar(im, cax=cax,
                        ticks=[np.nanmin(sinogram), np.nanmax(sinogram)])
    cbar.ax.tick_params(labelsize=5)
    cbar.ax.set_yticklabels([f"{np.nanmin(sinogram):.3g}",
                             f"{np.nanmax(sinogram):.3g}"])
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


def Plot_reconstruction(reconstruction, savepath=None, coordinates=None,
                        f_score=None):
    """ Plot a reconstruction of the subunit layout.

    Parameters
    ----------
    reconstruction : ndarray
        2D array containing the reconstruction.
    savepath : string, optional
        If provided, the plot is not shown in terminal but saved at the given
        location.
    coordinates : ndarray, optional
        If provided, the coordinates contained in this 2D array are marked in
        the plot.
    f_score : float or list of float, optional
        If provided, the F-score will be written in the plot.
    """

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle("Reconstruction")
    ax.imshow(np.transpose(reconstruction), origin='lower', cmap='RdBu_r',
              vmin=-np.max(np.abs(reconstruction)),
              vmax=np.max(np.abs(reconstruction)))
    if coordinates is not None and coordinates.size > 0:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c='yellow',
                   marker='x')
    if f_score is None:
        pass
    elif type(f_score) == list:
        if len(f_score) == 3:
            fig.text(0.05, 0.01,
                     f"F-scores of hotspots: {f_score[0]:.2f} (both), "
                     + f"{f_score[1]:.2f} (1st layout), "
                     + f"{f_score[2]:.2f} (2nd layout)", size=6)
        elif len(f_score) == 2:
            fig.text(0.05, 0.01,
                     f"F-scores of hotspots: {f_score[0]:.2f} (subunits), "
                     + f"{f_score[1]:.2f} (photoreceptors), ", size=6)
    else:
        fig.text(0.05, 0.01, f"F-score of hotspots: {f_score:.2f}", size=6)
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, dpi=300)
    plt.clf()
    plt.close('all')


###############################################################################
# Calculating functions
###############################################################################
def Ricker_wavelet(pos, half_w, surround_factor):
    """ Compute the value of a Ricker wavelet at the given location.

    Wavelet is centered at 0 and normalized to a maximum value of 1.

    Parameters
    ----------
    pos : float
        Location at which the Ricker wavelet is evaluated.
    half_w : float
        Half width of the wavelet.
    surround_factor : float, optional
        Multiplicative factor applied to the surround part of the wavelet. The
        resulting values of the wavelet will be clipped to the range [-1, 1].

    Returns
    -------
    float
        Value of the Ricker wavelet at *pos*.
    """

    value = (1-np.square(pos/half_w)) * np.exp(-np.square(pos/half_w)/2)
    value[value <= 0] *= surround_factor
    value[value < -1] = -1
    value[value > 1] = 1

    return value


def Create_stimulus(resolution, half_w, angle, position, surround_factor,
                    num_positions):
    """ Return a Ricker stripe stimulus with the given specifications.

    Parameters
    ----------
    resolution : int
        Size of the image as measured along an edge.
    half_w : float
        Half width of the Ricker stripe profile.
    angle : float
        Rotational angle of the stripe in radians. 0 yields a vertical stripe,
        higher values turn the stripe clockwise.
    position : int
        Defines the position of the stripe. Should be from 0 to *resolution*-1.
        Low values mean stripe is placed at the left (for a vertical stripe).
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Ricker stripe.
    num_positions : int
        Number of total stripe positions. Needed to define the rotation axis.

    Returns
    -------
    ndarray
        2D containing the stimulus.
    """

    # 1st index of any array is x-coordinate, 2nd is y-coordinate
    yy, xx = np.array(np.meshgrid(np.arange(0, resolution),
                                  np.arange(0, resolution)), dtype=float)
    # Image needs to be rotated around the central (or right neighboring)
    # stripe position so that the sinogram satisfies the demands of scikit
    # learn's iradon function
    shift = (num_positions//2) * (resolution-1)/(num_positions-1)
    yy -= shift
    xx -= shift
    # Applying the rotation
    xx_rot = np.cos(angle) * xx - np.sin(angle) * yy
    # Shifting to the desired stripe position
    xx_rot += shift - position
    # Calculating the rotated and shifted image
    image = Ricker_wavelet(xx_rot, half_w, surround_factor)

    return image


def Measure_sinogram(rgc, num_positions, num_angles, half_w, surround_factor):
    """ Compute the sinogram of the given cell using stimuli with the given
    properties.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    num_positions : int
        Number of stripe positions to be measured.
    num_angles : int
        Number of stripe angles to be measured.
    half_w : float
        Half width of the Ricker stripe profile.
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Ricker stripe.

    Returns
    -------
    ndarray
        2D array containing the sinogram. First index denotes position of the
        stripe, second denotes angle of the stripe.
    """

    resolution = rgc.resolution
    responses = np.empty((num_positions, num_angles))
    for counter_a, angle in enumerate(np.linspace(0, np.pi, num=num_angles,
                                                  endpoint=False)):
        for counter_p, position in enumerate(np.linspace(0, resolution-1,
                                                         num=num_positions,
                                                         endpoint=True)):
            stimulus = Create_stimulus(resolution, half_w, angle, position,
                                       surround_factor, num_positions)
            responses[counter_p, counter_a] = rgc.response_to_flash(stimulus)

    return responses


def Reconstruction(sinogram, sigma, return_smoothed=False):
    """ Reconstructs the subunit layout from the given sinogram using filtered
    back-projection (FBP).

    Parameters
    ----------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        stripe, second denotes angle of the stripe.
    sigma : tuple
        Contains the standard deviations of the Gaussian smoothing of the
        sinogram in position-direction relative to the simulation area size and
        in angle-direction in units degrees.
    return_smoothed : bool, optional
        If True, returns the smoothed sinogram.

    Returns
    -------
    ndarray
        2D array containing the FBP.
    ndarray, optional
        2D array containing the smoothed sinogram. Only provided if
        return_smoothed is True.
    """

    # Convert the smoothing parameter units into units of sinogram elements
    sigma_pos = sigma[0] * (sinogram.shape[0]-1)
    sigma_ang = sigma[1] / 180 * sinogram.shape[1]
    # Apply smoothing
    smoothed = gaussian_filter(sinogram, sigma=(sigma_pos, sigma_ang))
    # Transposing the FBP due to different angle conventions
    fbp = np.transpose(iradon(smoothed))

    if return_smoothed:
        return fbp, smoothed
    else:
        return fbp


def Find_hotspots(fbp):
    """ Localizes hotspots in the FBP by finding local maxima higher than 30%
    of the global maximum in the inner 90% circle.

    Parameters
    ----------
    fbp : ndarray
        2D array containing the FBP.

    Returns
    -------
    ndarray
        2D array containing the x- and y-coordinates of all identified
        hotspots.
    """

    global_max = np.max(fbp)
    coordinates = peak_local_max(fbp, min_distance=1)
    coordinates = [coord for coord in coordinates
                   if fbp[coord[0], coord[1]] >= 0.3*global_max]
    coordinates = np.array(coordinates)
    if coordinates.size > 0:
        distances = np.sqrt(np.sum(np.square(coordinates - (fbp.shape[0]-1)/2),
                                   axis=1))
        coordinates = coordinates[distances <= 0.9 * (fbp.shape[0]-1)/2]

    return coordinates


def F_score(rgc, locations, num_positions, extended_statistics=False,
            subunits_of_interest='all', use_photoreceptors=False):
    """ Calculates the F-score for the detected subunit locations.

    Detected subunits are determined by checking if a detected location falls
    within the 0.75 sigma ellipse of the real subunit.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    locations : ndarray
        2D array containing the x- and y-coordinates of all locations that are
        tested against the subunit locations.
    num_positions : int
        Number of stripe positions that were measured. This is equal to the
        edge size of the reconstruction.
    extended_statistics : bool, optional
        If True, details about the kind of mistakes will also be computed and
        returned. Default is False.
    subunits_of_interest : str or ndarray, optional
        References the subunits that *locations* aims to detect and that should
        be considered in the F-score calculation. If not 'all' (default), must
        be a 1D boolean array with the same length as the number of subunits of
        *rgc*.
    use_photoreceptors : bool, optional
        If True, F-score is calculated for the photoreceptors instead of the
        subunits. This assumes, that *rgc* is modelling photoreceptors.

    Returns
    -------
    f_score : float
        F-score of *locations* as detections of subunits in *rgc*.
    misaligned : int, optional
        Number of subunits that were detected by a location in the 1.5 sigma,
        but not 0.75 sigma ellipse. Only returned if extended_statistics is
        True.
    solo_subunits : int, optional
        Number of subunits that had no free location in the 1.5 sigma ellipse.
        Only returned if extended_statistics is True.
    solo_locations : int, optional
        Number of locations that were in no free subunit's 1.5 sigma ellipse.
        Only returned if extended_statistics is True.
    """

    # Preprocesssing
    if use_photoreceptors:
        subunit_params = rgc.photoreceptor_params
        num_subunits = rgc.num_photoreceptors
    else:
        subunit_params = rgc.subunit_params
        num_subunits = rgc.subunits.shape[0]
    if type(subunits_of_interest) == str and subunits_of_interest == 'all':
        subunits_of_interest = np.ones(num_subunits, dtype=bool)

    # First create subunit arrays with the parameters of the rgc's subunits,
    # but with the array size of the reconstruction, i.e. scaled versions of
    # rgc.subunits.
    scaling = (num_positions-1) / (rgc.resolution-1)
    subunits = [Subunit_Model.Gaussian_array(num_positions,
                                             params[0]*scaling,
                                             params[1]*scaling,
                                             params[2]*scaling,
                                             params[3]*scaling,
                                             params[4])
                for params in subunit_params[subunits_of_interest]]
    subunits = np.array(subunits)
    # Calculate the amplitude of the subunit gaussians.
    amplitudes = 1/(2*np.pi*subunit_params[subunits_of_interest, 2]
                    *subunit_params[subunits_of_interest, 3]*scaling**2)

    # If the 0.75 sigma ellipses don't overlap (which is usually the case) and
    # extended statistics are not required, a quicker approach can be used.
    if not extended_statistics:
        # Calculate what the values of Gaussians with the given amplitudes at
        # 0.75 sigma.
        thresholds = np.exp((-0.75**2)/2) * amplitudes
        # Convert the subunit arrays into arrays that are true at all locations
        # within 0.75 sigma.
        within = np.transpose(np.transpose(subunits) >= thresholds)
    # Make sure that the 0.75 sigma ellipses really don't overlap
    if not extended_statistics and not np.max(np.sum(within, axis=0) > 1):
        # Use the locations as indices to find out which subunits each location
        # has hit
        detections_idxb = [within[:, loc[0], loc[1]] for loc in locations]
        # Count which subunits have been detected while avoiding redundant
        # detections
        true_positives = np.sum(np.any(np.array(detections_idxb), axis=0))
    # In case more information about the kind of mistakes is required or the
    # 0.75 sigma ellipses did overlap, a more detailed calculation needs to be
    # done
    else:
        # Scale the subunits to an amplitude of one
        subunits = subunits / np.transpose([[amplitudes]])
        # Identify the weight of each subunit at every identified location
        sub_loc = subunits[:,
                           locations[:, 0] if locations.size > 0 else [],
                           locations[:, 1] if locations.size > 0 else []]
        # Later computations are simplified by adding a row and column of zeros
        # here
        sub_loc = np.vstack([np.column_stack([sub_loc,
                                              np.zeros(sub_loc.shape[0])]),
                             np.zeros(sub_loc.shape[1]+1)])
        # Iteratively remove the subunit-location combination that is closest
        # together until no combination closer than 0.75 sigma is left
        threshold = np.exp((-0.75**2)/2)
        true_positives = 0
        while np.any(sub_loc > threshold):
            max_id = np.unravel_index(np.argmax(sub_loc), sub_loc.shape)
            sub_loc = np.delete(np.delete(sub_loc, max_id[0], axis=0),
                                max_id[1], axis=1)
            true_positives += 1
        if extended_statistics:
            # Then remove combinations closer than 1.5 sigma and count them as
            # misaligned
            threshold = np.exp((-1.5**2)/2)
            misaligned = 0
            while np.any(sub_loc > threshold):
                max_id = np.unravel_index(np.argmax(sub_loc), sub_loc.shape)
                sub_loc = np.delete(np.delete(sub_loc, max_id[0], axis=0),
                                    max_id[1], axis=1)
                misaligned += 1
            # Evaluate how many subunits and locations are left
            solo_subunits, solo_locations = np.array(sub_loc.shape) - 1

    # Calculate the F-score
    f_score = 2*true_positives/(locations.shape[0] + np.sum(subunits_of_interest))

    # Return results
    if extended_statistics:
        return f_score, misaligned, solo_subunits, solo_locations
    else:
        return f_score


def STR_analysis(rgc, num_positions, num_angles, half_w, surround_factor,
                 smoothing, known_sinogram=None):
    """ Using STR to analyze a model cell.

    Parameters
    ----------
    rgc : Subunit_Model
        Object of the class Subunit_Model from Subunit_Model.py that is
        investigated.
    num_positions : int
        Number of stripe positions to be measured.
    num_angles : int
        Number of stripe angles to be measured.
    half_w : float
        Half width of the Ricker stripe profile.
    surround_factor : float
        Multiplicative factor applied to the sidebands of the Ricker stripe.
    smoothing : tuple
        Contains the standard deviations of the Gaussian smoothing of the
        sinogram in position-direction relative to the simulation area size and
        in angle-direction in units degrees.
    known_sinogram : ndarray, optional
        If the sinogram of the cell is already known and measuring it again
        should be avoided (e.g. for performance reasons), the sinogram can be
        passed here and only the rest of the analysis will be performed.
        *known_sinogram* should the be like *sinogram* and will also be
        returned as such.

    Returns
    -------
    sinogram : ndarray
        2D array containing the sinogram. First index denotes position of the
        stripe, second denotes angle of the stripe.
    smoothed : ndarray
        2D array containing the smoothed sinogram.
    fbp : ndarray
        2D array containing the FBP.
    hotspots : ndarray
        Coordinates of the hotspots in the FBP.
    f_score : float or list of float
        F-score of how well the detected hotspots correspond to the subunits of
        the model. If the model consists of two superimposed layouts, this is a
        list containing the F-score for all subunits, the first layout, and the
        second layout. If the model uses photoreceptors, the list contains the
        F-score for the subunits and the F-score for the photoreceptors.
    """

    if known_sinogram is None:
        sinogram = Measure_sinogram(rgc, num_positions, num_angles, half_w,
                                    surround_factor)
    else:
        sinogram = known_sinogram
    fbp, smoothed = Reconstruction(sinogram, smoothing, return_smoothed=True)
    hotspots = Find_hotspots(fbp)
    if rgc.scenario == 'photoreceptors':
        f_score = [F_score(rgc, hotspots, num_positions),
                   F_score(rgc, hotspots, num_positions,
                           use_photoreceptors=True)]
    elif type(rgc.num_subunits) == int:
        f_score = F_score(rgc, hotspots, num_positions)
    else:
        f_score = [F_score(rgc, hotspots, num_positions),
                   F_score(rgc, hotspots, num_positions,
                           subunits_of_interest=np.repeat([True, False],
                                                          rgc.num_subunits)),
                   F_score(rgc, hotspots, num_positions,
                           subunits_of_interest=np.repeat([False, True],
                                                          rgc.num_subunits))]

    return sinogram, smoothed, fbp, hotspots, f_score


###############################################################################
# Main program
###############################################################################
if __name__ == '__main__':

    # Set the model up
    rgc = Subunit_Model.Subunit_Model(resolution=RESOLUTION, scenario=SCENARIO,
                                      subunit_nonlinearity=SUBUNIT_NL,
                                      subunit_weights=SYNAPTIC_WEIGHTS,
                                      rgc_nonlinearity=RGC_NL,
                                      rgc_spiking=None,
                                      num_subunits=NUM_SUBUNITS,
                                      layout_seed=LAYOUT_SEED)

    # Applying STR
    temp = STR_analysis(rgc, NUM_POSITIONS, NUM_ANGLES, HALF_W,
                        SURROUND_FACTOR, (0, 0))
    sinogram, _, fbp, hotspots, f_score = temp

    # Doing the same for a spiking neuron
    rgc.set_spiking('poisson', spiking_coefficient='realistic',
                    poisson_seed=POISSON_SEED)
    temp = STR_analysis(rgc, NUM_POSITIONS, NUM_ANGLES, HALF_W, 
                        SURROUND_FACTOR, SMOOTHING)
    sinogram_sp, sinogram_sp_f, fbp_sp_f, hotspots_sp_f, f_score_sp_f = temp

    # Creating a folder for the plots
    pathlib.Path("STR").mkdir(parents=True, exist_ok=True)

    # Plots
    rgc.plot_subunit_ellipses("STR/1 Subunit Layout")
    rgc.plot_receptive_field("STR/2 Receptive Field")
    Plot_stimulus(RESOLUTION, HALF_W, SURROUND_FACTOR, NUM_POSITIONS,
                  "STR/3 Stimulus")
    Plot_sinogram(sinogram, "STR/4 Sinogram")
    Plot_sinogram(sinogram_sp, "STR/5 Sinogram Spiking")
    Plot_sinogram(sinogram_sp_f, "STR/6 Sinogram Spiking Filtered")
    Plot_reconstruction(fbp, "STR/7 FBP", coordinates=hotspots, 
                        f_score=f_score)
    Plot_reconstruction(fbp_sp_f, "STR/8 FBP Spiking Filtered",
                        coordinates=hotspots_sp_f, f_score=f_score_sp_f)
