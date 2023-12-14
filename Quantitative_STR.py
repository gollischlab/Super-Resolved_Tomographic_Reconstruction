""" This script performs quantitative analyses of the super-resolved
tomographic reconstruction method by doing large-scale simulations."""


import numpy as np
from itertools import product
import pathlib
import Subunit_Model, STR


###############################################################################
# Global parameters
###############################################################################
# Number of model initialisations per parameter set
NUM_INITIALISATIONS = 1000
# Parameters that are not intended to be changed
SCENARIO = 'realistic gauss'
RGC_NL = None
RGC_SPIKING = 'poisson'
SPIKING_COEFFICIENT = 'realistic'
RESOLUTION = 40
NUM_POSITIONS = 60
NUM_ANGLES = 36
# Parameters that are intended to be changed and affect the model
# Changeable parameters must always be contained in lists!
NUM_SUBUNITS = [10]
SUBUNIT_NL = ['threshold-linear']
SYNAPTIC_WEIGHTS = ['equal']
OVERLAP_FACTOR = [1.35]
SWAP_GAUSS_FOR_COSINE = [False]
SPIKING_BASE_LEVEL = [0.0]
# Parameters that are intended to be changed and don't affect the model
# Changeable parameters must always be contained in lists!
HALF_W = [2.5]
SURROUND_FACTOR = [2.5]
SMOOTHING_POSITION = [1.5]
SMOOTHING_ANGLE = [1.0]
# Parameters calculated from the others
LAYOUT_SEEDS = np.arange(0, NUM_INITIALISATIONS)
POISSON_SEEDS = np.arange(0, NUM_INITIALISATIONS) + 10000


###############################################################################
# Main program
###############################################################################
# Creating a folder for the results
pathlib.Path("Quantitative STR").mkdir(parents=True, exist_ok=True)

# Outer loop goes over all parameters that require new model initialisations
model_param_list = list(product(NUM_SUBUNITS, SUBUNIT_NL, SYNAPTIC_WEIGHTS, OVERLAP_FACTOR, SWAP_GAUSS_FOR_COSINE, SPIKING_BASE_LEVEL))
for counter_model, (num_subunits, subunit_nl, synaptic_weights, overlap_factor, swap_gauss_for_cosine, spiking_base_level) in enumerate(model_param_list):
    # Initialising all models. Remember to reset the spiking!
    print(f"Model Initialisation {counter_model+1}/{len(model_param_list)}")
    rgcs = np.empty(NUM_INITIALISATIONS, dtype=object)
    for counter_init in range(NUM_INITIALISATIONS):
        print(f"\r{counter_init/NUM_INITIALISATIONS*100:.1f}%     ", end="")
        rgc = Subunit_Model.Subunit_Model(resolution=RESOLUTION,
                                          scenario=SCENARIO,
                                          subunit_nonlinearity=subunit_nl,
                                          subunit_weights=synaptic_weights,
                                          rgc_nonlinearity=RGC_NL,
                                          rgc_spiking=RGC_SPIKING,
                                          num_subunits=num_subunits,
                                          overlap_factor=overlap_factor,
                                          swap_gauss_for_cosine=swap_gauss_for_cosine,
                                          layout_seed=LAYOUT_SEEDS[counter_init],
                                          poisson_seed=POISSON_SEEDS[counter_init],
                                          spiking_coefficient=SPIKING_COEFFICIENT,
                                          spiking_base_level=spiking_base_level)
        rgcs[counter_init] = rgc
    print("\r100%     ")

    # Inner loop goes over all parameters that only require new measurements
    measurement_param_list = list(product(HALF_W, SURROUND_FACTOR, SMOOTHING_POSITION, SMOOTHING_ANGLE))
    for counter_measurement, (half_w, surround_factor, smoothing_position, smoothing_angle) in enumerate(measurement_param_list):
        # Resetting the stored sinograms if a new half_w or surround_factor is
        # used, which requires a new sinogram measurement
        if counter_measurement%(len(SMOOTHING_POSITION)*len(SMOOTHING_ANGLE)) == 0:
            sinograms = [None for i in range(NUM_INITIALISATIONS)]
        # Measuring and calculating F-scores
        print(f"Measurement {counter_measurement+1}/{len(measurement_param_list)} from model initialisation {counter_model+1}/{len(model_param_list)}")
        f_scores = np.empty(NUM_INITIALISATIONS)
        for counter_init in range(NUM_INITIALISATIONS):
            print(f"\r{counter_init/NUM_INITIALISATIONS*100:.1f}%     ", end="")
            rgcs[counter_init].set_spiking(RGC_SPIKING,
                                           spiking_coefficient=SPIKING_COEFFICIENT,
                                           poisson_seed=POISSON_SEEDS[counter_init])
            temp = STR.STR_analysis(rgcs[counter_init],
                                    NUM_POSITIONS,
                                    NUM_ANGLES,
                                    half_w,
                                    surround_factor,
                                    (smoothing_position,
                                     smoothing_angle),
                                    known_sinogram=sinograms[counter_init])
            sinograms[counter_init] = temp[0]
            f_scores[counter_init] = temp[4]
        print("\r100%     ")

        # Saving the results
        np.savez("Quantitative STR/"
                 + f"{num_subunits}_{subunit_nl}_{synaptic_weights}_"
                 + f"{overlap_factor}_{swap_gauss_for_cosine}_"
                 + f"{spiking_base_level}_{half_w}_{surround_factor}_"
                 + f"{smoothing_position}_{smoothing_angle}_"
                 + f"{NUM_INITIALISATIONS}.npz",
                 f_scores=f_scores,
                 layout_seeds=LAYOUT_SEEDS,
                 poisson_seeds=POISSON_SEEDS,
                 scneario=SCENARIO,
                 rgc_nl=RGC_NL,
                 rgc_spiking=RGC_SPIKING,
                 spiking_coefficient=SPIKING_COEFFICIENT,
                 resolution=RESOLUTION,
                 num_positions=NUM_POSITIONS,
                 num_angles=NUM_ANGLES,
                 num_subunits=num_subunits,
                 subunit_nl=subunit_nl,
                 synaptic_weights=synaptic_weights,
                 overlap_factor=overlap_factor,
                 swap_gauss_for_cosine=swap_gauss_for_cosine,
                 spiking_base_level=spiking_base_level,
                 half_w=half_w,
                 surround_factor=surround_factor,
                 smoothing_position=smoothing_position,
                 smoothing_angle=smoothing_angle)
