Self Supervised Denoising - Transform 2022
=========

Authors: 
 - Claire Birnie (claire.birnie@kaust.edu.sa), and 
 - Sixiu Liu (sixiu.liu@kaust.edu.sa)

Tutorial overview
---------------------------


Repository overview
---------------------------



Disclaimer: the code has all been wrote and tested on Linux operating systems, where GPU access is available. 

Installation instructions
---------------------------

**Environment creation**
We have made a conda environment file which contains all the necessary packages to run the tutorials. For ease of use,
an installation script has been written to create the conda environment and  check the necessary packages were 
correctly installed. The environment can be created with the following command (executed when in this folder):

    ./install_tt2022ssd.sh

**KAUST-IBEX Specific instructions**
For KAUST students/employees we have created a slurm script for running the environment and jupyter notebook. Prior to
submitting the slurm job, log in to a GPU login node and run the environment creation file. This will make the 
environment available across all GPU work nodes. 

Line 17 of the slurm file will need updated to point to the home directory where you have cloned the tutorial material, 
i.e. wherever this README is currently sat in your IBEX directory. In our case, we have it in our scratch space. Submit
the job following the normal procedure:

    slurm tt2022ssd-jupyter.slurm

Once the job is accepted and running, view the log file (`cat slurm[XXXX].`). This gives you the tunneling instructions 
and the path to the jupyter server instance.

