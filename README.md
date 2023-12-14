# Super-Resolved_Tomographic_Reconstruction
Simulations of subunit models of retinal ganglion cells and inference of their subunits with super-resolved tomographic reconstruction (STR).

This code accompanies the manuscript by Krüppel et al.: "Applying Super-Resolution and Tomography Concepts to Identify Receptive Field Subunits in the Retina". It implements the retinal ganglion cell simulations and the STR method described there and can be used to reproduce all presented simulation data. For a description of the model and method basics, please refer to the manuscript. The following description will outline the contents and use of the provided scripts and how to reproduce the presented data.

### Subunit_Model.py
This script provides a class for a subunit model of a retinal ganglion cell. The class *Subunit_Model* can be used to generate an LNLNP model in a sandbox principle, allowing, for example, to try different subunit nonlinearities. It can easily be extended beyond the setups presented in the manuscript. More detailed tuning of the model can be done via the global parameter *DEFAULT_PARAMS* that controls, e.g., the noise level. The method *response_to_flash* calculates the response of the model cell to a flash of a given stimulus, e.g., a Ricker stripe stimulus.

### Mexican_Hat.py
This script provides two functions that are requried to probe the receptive field of a simulated ganglion cell with a Mexican hat-shaped stimulus, illustrated in manuscript figure 1. The function *Response_map* performs such a probing and returns the responses of the model.

### STR.py
The main STR analysis is implemented here. The main program instantiates a *Subunit_Model* and analyzes it with STR. It generates several plots depicting this analysis and saves them in a subfolder. One can change, for example, the global parameters to investigate how they affect the analysis.

### Quantitative_STR.py
This script can be used to calculate average F-scores evaluating the reconstruction quality using certain model and analysis parameters. The results of each parameter combination are saved in a separate file in a subfolder, containing the parameter values in the file name.

### Examples shown in figures
The default parameters are described in the manuscript and correspond to the default parameters in the scripts. Parameter changes, e.g. a different subunit nonlinearity, are mentioned in the text and can be implemented by changing the corresponding parameters in the script. In combination with the following seeds of the random number generators, this information can be used to replicate the results shown in the manuscript's figures. Each instantiation of a *Subunit_Model* has two seeds - one for generating the subunit layout and one for the poisson spiking process.

- Fig 2 (top): 20 (layout seed), 2000 (poisson seed)
- Fig 2 (center): 30, 3000
- Fig 2 (bottom): 46, 4000
- Fig 3: 4, 1025
- Fig 4 (1st row): 1501, 151
- Fig 4 (2nd row): 1517, 150
- Fig 4 (3rd row): 1525, 150
- Fig 4 (4th row): 1531, 152
- Fig 4 (5th row): 1553, 150

Sample seeds were chosen to generate representative layouts and F-scores. The 1000 seeds for the calculation of average F-scores covered the range [0, 1000) for the layout and [10000, 11000) for the poisson process.
