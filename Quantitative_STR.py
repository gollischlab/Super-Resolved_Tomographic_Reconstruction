""" This script performs quantitative analyses of the super-resolved
tomographic reconstruction method by doing large-scale simulations."""


import numpy as np
from itertools import product
from multiprocess import Pool, cpu_count
from tqdm import tqdm
import pathlib
import Subunit_Model, STR


###############################################################################
# Global parameters
###############################################################################
# Number of model initialisations per parameter set
NUM_INITIALISATIONS = 1000
# Parameters that are not intended to be changed
RGC_NL = None
RGC_SPIKING = 'poisson'
SPIKING_COEFFICIENT = 'realistic'
RESOLUTION = 40
# Parameters that are intended to be changed and affect the model
# Changeable parameters must always be contained in lists!
SCENARIO = ['realistic gauss']
NUM_SUBUNITS = [10]
SUBUNIT_NL = ['threshold-linear']
SYNAPTIC_WEIGHTS = ['equal']
OVERLAP_FACTOR = [1.35]
SWAP_GAUSS_FOR_COSINE = [False]
SPIKING_BASE_LEVEL = [0.0]
OPPOSING_POLARITY = [False]
# Parameters that are intended to be changed and don't affect the model
# Changeable parameters must always be contained in lists!
NUM_POSITIONS = [60]
NUM_ANGLES = [36]
HALF_W = [2.5]
SURROUND_FACTOR = [2.5]
SMOOTHING_POSITION = [0.025]
SMOOTHING_ANGLE = [5.0]
# Parameters calculated from the others
LAYOUT_SEEDS = np.reshape(np.arange(0, 2*NUM_INITIALISATIONS),
                          (NUM_INITIALISATIONS, 2), order='F')
POISSON_SEEDS = np.arange(0, NUM_INITIALISATIONS) + 10000
MODEL_PARAM_LIST = list(product(SCENARIO, NUM_SUBUNITS, SUBUNIT_NL,
                                SYNAPTIC_WEIGHTS, OVERLAP_FACTOR,
                                SWAP_GAUSS_FOR_COSINE, SPIKING_BASE_LEVEL,
                                OPPOSING_POLARITY))
NUM_MODEL_INITS = len(MODEL_PARAM_LIST)
MEASUREMENT_PARAM_LIST = list(product(NUM_POSITIONS, NUM_ANGLES, HALF_W,
                                      SURROUND_FACTOR, SMOOTHING_POSITION,
                                      SMOOTHING_ANGLE))
NUM_MEASUREMENTS = len(MEASUREMENT_PARAM_LIST)


###############################################################################
# Function that completely analyses one model initialization seed
###############################################################################
def Measure_seed(seed_index):
    """ Measure all F-scores for one combination of layout and poisson seed.

    Parameters
    ----------
    seed_index : int
        Index of the seed to be used in *LAYOUT_SEEDS* and *POISSON_SEEDS*.

    Returns
    -------
    ndarray
        1D array containing all the F-scores calculated for that seed.
    """

    # Create a container for all F-score results
    f_scores = np.empty(NUM_MODEL_INITS * NUM_MEASUREMENTS, dtype=object)
    # Outer loop goes over all parameters that require new model initialisations
    for counter_model, (scenario, num_subunits, subunit_nl, synaptic_weights, overlap_factor, swap_gauss_for_cosine, spiking_base_level, opposing_polarity) in enumerate(MODEL_PARAM_LIST):
        # Initialising the model. Remember to reset the spiking!
        rgc = Subunit_Model.Subunit_Model(resolution=RESOLUTION,
                                          scenario=scenario,
                                          subunit_nonlinearity=subunit_nl,
                                          subunit_weights=synaptic_weights,
                                          rgc_nonlinearity=RGC_NL,
                                          rgc_spiking=RGC_SPIKING,
                                          num_subunits=num_subunits,
                                          overlap_factor=overlap_factor,
                                          swap_gauss_for_cosine=swap_gauss_for_cosine,
                                          layout_seed=LAYOUT_SEEDS[seed_index, 0] if (type(num_subunits) == int and scenario != 'photoreceptors') else LAYOUT_SEEDS[seed_index],
                                          poisson_seed=POISSON_SEEDS[seed_index],
                                          spiking_coefficient=SPIKING_COEFFICIENT,
                                          spiking_base_level=spiking_base_level,
                                          opposing_polarity=opposing_polarity)

        # Inner loop goes over all parameters that only require new measurements
        for counter_measurement, (num_positions, num_angles, half_w, surround_factor, smoothing_position, smoothing_angle) in enumerate(MEASUREMENT_PARAM_LIST):
            # Resetting the stored sinogram if a new num_positions, num_angles,
            # half_w or surround_factor is used, which requires a new sinogram
            # measurement
            if counter_measurement%(len(SMOOTHING_POSITION)*len(SMOOTHING_ANGLE)) == 0:
                sinogram = None
            # Measuring and calculating F-scores
            rgc.set_spiking(RGC_SPIKING,
                            spiking_coefficient=SPIKING_COEFFICIENT,
                            poisson_seed=POISSON_SEEDS[seed_index])
            temp = STR.STR_analysis(rgc,
                                    num_positions,
                                    num_angles,
                                    half_w,
                                    surround_factor,
                                    (smoothing_position, smoothing_angle),
                                    known_sinogram=sinogram)
            sinogram = temp[0]
            f_scores[counter_model * NUM_MEASUREMENTS + counter_measurement] = temp[4]

    return f_scores



###############################################################################
# Main program
###############################################################################
if __name__ == '__main__':
    # Creating a folder for the results
    pathlib.Path("Quantitative STR").mkdir(parents=True, exist_ok=True)
    # Calculate all F-scores of the different intialization seeds
    print("Running simulations...")
    with Pool(min(cpu_count(), 40)) as pool:
        f_scores = list(tqdm(pool.imap_unordered(Measure_seed,
                                                 list(range(NUM_INITIALISATIONS))),
                             total=NUM_INITIALISATIONS))
    f_scores = np.array(f_scores)

    # To properly save all results in the correct files, the for-loops of the
    # function Measure_seed need to be recreated
    print("Saving results...")
    progressbar = tqdm(total=NUM_MODEL_INITS*NUM_MEASUREMENTS)
    for counter_model, (scenario, num_subunits, subunit_nl, synaptic_weights, overlap_factor, swap_gauss_for_cosine, spiking_base_level, opposing_polarity) in enumerate(MODEL_PARAM_LIST):
        for counter_measurement, (num_positions, num_angles, half_w, surround_factor, smoothing_position, smoothing_angle) in enumerate(MEASUREMENT_PARAM_LIST):
            # Saving the results
            f_scores_out = np.array(list(f_scores[:, counter_model * NUM_MEASUREMENTS + counter_measurement]))
            np.savez("Quantitative STR/"
                     + ("photoreceptors_" if scenario == 'photoreceptors' else "")
                     + f"{num_subunits}_"
                     + ("opposing-polarity_" if type(num_subunits) != int and opposing_polarity else "")
                     + f"{subunit_nl}_{synaptic_weights}_"
                     + f"{overlap_factor:g}_{swap_gauss_for_cosine}_"
                     + f"{spiking_base_level:g}_"
                     + f"{num_positions}_{num_angles}_"
                     + f"{half_w:g}_{surround_factor:g}_"
                     + f"{smoothing_position:g}_{smoothing_angle:g}_"
                     + f"{NUM_INITIALISATIONS}.npz",
                     f_scores=f_scores_out,
                     layout_seeds=LAYOUT_SEEDS[:, 0] if (type(num_subunits) == int and scenario != 'photoreceptors') else LAYOUT_SEEDS,
                     poisson_seeds=POISSON_SEEDS,
                     scenario=scenario,
                     rgc_nl=RGC_NL,
                     rgc_spiking=RGC_SPIKING,
                     spiking_coefficient=SPIKING_COEFFICIENT,
                     resolution=RESOLUTION,
                     num_positions=num_positions,
                     num_angles=num_angles,
                     num_subunits=num_subunits,
                     subunit_nl=subunit_nl,
                     synaptic_weights=synaptic_weights,
                     overlap_factor=overlap_factor,
                     swap_gauss_for_cosine=swap_gauss_for_cosine,
                     spiking_base_level=spiking_base_level,
                     opposing_polarity=opposing_polarity,
                     half_w=half_w,
                     surround_factor=surround_factor,
                     smoothing_position=smoothing_position,
                     smoothing_angle=smoothing_angle)
            progressbar.update(1)
    progressbar.close()
