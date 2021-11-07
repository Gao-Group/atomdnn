.. AtomDNN documentation master file, created by
   sphinx-quickstart on Sun Oct 17 10:39:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AtomDNN's documentation!
===================================
AtomDNN is a package for training descriptor-based Behler-Parinello type of neural
network potentials. It provides:

- **Generation of descriptors**: users just need to provide the atomic structures
  saved as extxyz files. we use LAMMPS as a Calculator
  (through an added Compute comand) to compute descritpors and their
  derivatives w.r.t. atom positions (`ACSF <https://aip.scitation.org/doi/abs/10.1063/1.3553717>`_ and
  `SOAP <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.87.184115>`_ are currently supported), while users are free to use any other
  tools to generate descriptors. Note that the parallelized LAMMPS can save
  computation time since the calculation of the derivatives may be
  time consuming.

- **Potential training**: we use the `tf.module in Tensorflow2
  <https://www.tensorflow.org/api_docs/python/tf/Module>`_ to build
  and train the neurall network. Potential energy, atomic forces and
  stress can be used for training. Any activation functions and loss
  functions supported by Tensorflow can be used. The trained potential is finally saved as a
  tensorflow model. The descritpors parameters are saved in the same
  folder for the predictions with LAMMPS.

- **Integration to LAMMPS/ASE**: a new pair style pair_tfdnn is added to
  LAMMPS in order to do the predictions with the trained tensorflow
  model in MD and MS simulations. In this pair style, the prediction is
  performed with tensorflow C APIs. To use the potential in ASE, one
  just needs to use LAMMPS as an extenral Calculator, like any other potentials.


.. toctree::
   :glob:
   :caption: Getting Started
   :maxdepth: 1

   getstarted/install.rst
   getstarted/example.rst


.. toctree::
   :glob:
   :caption: Tutorials
   :maxdepth: 1

   tutorials/data_pipeline.rst
   tutorials/energy_calculation.rst
   tutorials/force_calculation.rst
   tutorials/lammps.rst

.. toctree::
   :glob:
   :caption: Modules
   :maxdepth: 1

   module/descriptor.rst
   module/data.rst
   module/network.rst



.. toctree::
   :glob:
   :caption: About
   :maxdepth: 1

   about/authors.rst
   about/contact.rst
   about/license.rst
   about/funding.rst
