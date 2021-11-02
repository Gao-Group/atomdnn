=================
LAMMPS interface
=================

A new pair style  **tfdnn** (Tensorflow2 Deep Neural Network) is added to LAMMPS in order to do the predictions with the trained tensorflow model in MD and MS simulations. In this pair style, the prediction is performed with tensorflow C APIs. In order to run the tensorflow APIs in LAMMPS, firstly, one needs to build the tensorflow source code to get tensorflow API libraries (this could be a lengthy process, a guild can be found `here <https://gist.github.com/kmhofmann/e368a2ebba05f807fa1a90b3bf9a1e03>`_). After that, to build LAMMPS, one needs to copy pair_tfdnn.cpp, pair_tfdnn.h and Makefile (all inside atomdnn/lammps) to src in lammps and run the compile.

Once LAMMPS is built, the trained potential can be used just like any other LAMMPS potentials. A sample lammps input script::

  clear
  dimension 3
  boundary p p p
  units metal
  atom_style atomic
  neighbor 	0.2 bin

  read_data lmpdata
  mass 1 12

  pair_style   tfdnn
  pair_coeff   * * example.tfdnn

  fix NVE all nve
  run 100

where the tfdnn is the pair style name and example.tfdnn is the trained potential.
