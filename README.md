# AtomDNN
 
AtomDNN is a package for training descriptor-based Behler-Parinello type of neural network potentials. It provides:

**Generation of descriptors**: users just need to provide the atomic structures saved as extxyz files. We use LAMMPS as a Calculator (through an added Compute comand) to compute descritpors and their derivatives w.r.t. atom positions, while users are free to use any other tools to generate descriptors. Note that the parallelized LAMMPS can save computation time since the calculation of the derivatives may be time consuming.

**Potential training**: we use the tf.module in Tensorflow2 to build and train the neural network. Potential energy, atomic forces and stress can be used for training. Any activation functions and loss functions supported by Tensorflow can be used. The trained potential is finally saved as a tensorflow model.

**Integration to LAMMPS**: a new pair style pair_tfdnn is added to LAMMPS in order to do the predictions with the trained tensorflow model in MD and MS simulations. In this pair style, the prediction is performed with tensorflow C APIs. 

# Documentation

An online documentation can be found [here](https://gao-group.github.io/atomdnn/).

# Authors

This package is developed by [Gao research group](https://www.gao-group.org/) at University of Texas at San Antonio. Contributors include: Wei Gao, Daniel Millan, Daniela Posso and Colton Kubena.

# License

This software is licensed under the GNU General Public License version 3 or any later version (GPL-3.0-or-later).
