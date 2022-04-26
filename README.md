Self Supervised Denoising - Transform 2022
=========

This repository contains the codes created for the Self-supervised denosiing tutorial presented at Transform 2022.

Authors: 
 - Claire Birnie (claire.birnie@kaust.edu.sa), and 
 - Sixiu Liu (sixiu.liu@kaust.edu.sa)
 
The tutorial was originally presented as a live-stream event on YouTube on April 27 2022 at 11 UTC. 

YouTube Link: https://www.youtube.com/watch?v=d9yv90-JCZ0

Tutorial overview
---------------------------

Self-supervised learning offers a solution to the common limitation of the lack of noisy-clean pairs of data for training deep learning seismic 
denoising procedures.

In this tutorial, we will explain the theory behind blind-spot networks and how these can be used in a self-supervised manner, removing any 
requirement of clean-noisy training data pairs. We will deep dive into how the original methodologies for random noise can be adapted to handle 
realistic noise in seismic data, both pseudo-random noise and structured noise. Furthermore, each sub-topic presented will be followed by a live, 
code-along session such that all participants will be able to recreate the work shown and can afterwards apply it to their own use cases. 

If you found the tutorial useful please consider citing our work in your studies:

> Birnie, C., M. Ravasi, S. Liu, and T. Alkhalifah, 2021, The potential of self-supervised networks for random noise 
> suppression in seismic data: Artificial Intelligence in Geosciences.

> Liu, S., C. Birnie, and T. Alkhalifah, 2022, Coherent noise suppression via a self-supervised deep learning scheme: 
> 83rd EAGE Conference and Exhibition 2022, European Association of Geoscientists & Engineers, 1–5

Repository overview
---------------------------

The top level of the repository contains the skeleton tutorial notebooks that will be completed during the live YouTube tutorial.
This level also contains the necessary files for setting up the conda environment (and submitting jobs for those working on the 
KAUST IBEX cluster). As well as the standard git files - README, .gitignore, etc. 

The **Solutions** folder contains the completed notebooks. Note, there is no one *correct* way in which to write the necessary functions 
therefore the proposed solutions are only there to serve as guidance. 

Disclaimer: the code has all been wrote and tested on Linux operating systems, where GPU access is available. Neither of the authors are professional 
software developers therefore, whilst we have spent significant time testing the code, we cannot gaurantee it is free of bugs.

Installation instructions
---------------------------

As these procedures are based on deep-learning we encourage you to use a GPU if possible.

**Data Download**

The data utilised in this tutorial series can be downloaded from: https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/vjjry6BZ3n3Ewei

using the password: kaust

This folder contains synthetically generated shot gathers modelled using the Hess VTI model and a post-stack seismic section
of the Hess VTI model. The folder also contains a field data example originally downloaded from Madagascar of a post-stack
seismic section that is often used benchmarking new random noise suppression algorithms.


**Environment creation**

We have made a conda environment file which contains all the necessary packages to run the tutorials. For ease of use,
an installation script has been written to create the conda environment and  check the necessary packages were 
correctly installed. The environment can be created with the following command (executed when in this folder):

    ./install_tt2022ssd.sh
    
The enviornment can then be activated by running the command:

    conda activate tt2022ssd
    

**KAUST-IBEX Specific instructions**

For KAUST students/employees we have created a slurm script for running the environment and jupyter notebook. Prior to
submitting the slurm job, log in to a GPU login node and run the environment creation file. This will make the 
environment available across all GPU work nodes. 

Line 17 of the slurm file will need updated to point to the home directory where you have cloned the tutorial material, 
i.e. wherever this README is currently sat in your IBEX directory. In our case, we have it in our scratch space. Submit
the job following the normal procedure:

    slurm tt2022ssd-jupyter.slurm

Once the job is accepted and running, view the log file (`cat slurm[XXXX].out`). This gives you the tunneling instructions 
and the path to the jupyter server instance.

