# MMAP2MAR

# Contact

Thomas Tiotto (`t.f.tiotto@rug.nl`)

Alessandro Antonucci (`alessandro@idsia.ch`)

# Description

We present a heuristic strategy for marginal MAP (MMAP) queries in graphical models. 
The algorithm is based on a reduction of the task to a polynomial number of marginal inference computations. 

Given an input evidence, the marginals mass functions of the variables to be explained are computed. 
Marginal information gain is used to decide the variables to be explained ﬁrst, and their most probable marginal states are consequently moved to the evidence. 
The sequential iteration of this procedure leads to a MMAP explanation and the minimum information gain obtained during the process can be regarded as a conﬁdence measure for the explanation. 
Preliminary experiments show that the proposed conﬁdence measure is properly detecting instances for which the algorithm is accurate and, for sufﬁciently high conﬁdence levels, the algorithm gives the exact solution or an approximation whose Hamming distance from the exact one is small.

# Dependencies

`MMAP2MAR` relies on `Merlin` (https://github.com/radum2275/merlin) to calculate the partial marginal and marginal MAP inferences.
The testing relied on `Merlin 1.7.0`.

`Numpy` is also necessary for execution.
 
# Source Code

The repository is organized along the following directory structure:

* `src` - contains the source files
* `uai` - contains example graphical models
* `merlin` - contains the `merlin` executable that should be called `merlin`, intermediate working files are also generated in here

# Runnig the Solver

MMAP2MAR accepts the following command line arguments:

* `--input_file <filename>` - is the input graphical model file in `.uai` format
* `--evidence_file <filename>` - is the (optional) initial evidence file, in case it is not supplied a random initial evidence is generated
* `--evidence_size <size>` - is the size of initial random evidence, should be supplied if `--evidence_file` is not passed
* `--files_folder <foldername>` - is the folder where the `--input_file` will be sourced, defaults to `../uai/`
* `--merlin_folder <foldername>` - is the folder where the `merlin` executable will be sourced, defaults to `../merlin/`
* `--iterations <number>` - is the number of solutions generated and whose characteristics are averaged, it makes sense to execute more than once when no initial evidence is supplied as in that case a different random evidence will be used for every iteration, defaults to `1`
* `--entropy_threshold <threshold>` - is the threshold that the minimum entropy of the marginal mass functions must be below to be accepted, a lower threshold corresponds to a more "cautious" algorithm, corresponds to epsilon in the paper

Example of command line:
`python mmap2mar.py --iterations 10 --input_file driverlog01ac.wcsp.uai --evidence_size 3 --entropy_threshold 0.2`

This example will run the MMAP2MAR algorithm with a threshold of 0.2 over 10 iterations, each one with a random initial evidence of size 3, on the network defined by `../uai/driverlog01ac.wcsp.uai`.
The output of the algorithm can be found in the `../uai/driverlog01ac.wcsp.uai_work.evid` file that is created alongside the  input file.

All files (input, evidence) must be specified in the UAI file format.


# File Formats

## File Format

Refer to the Merlin documentation available on GitHub for a good description of the UAI input file format used to define the network the algorithm is run on, the initial evidence file and the final output of the algorithm.
