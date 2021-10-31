=============
Data pipeline
=============

From the initial data of atomic structures to the tensorflow
dataset that can be used for training, a data pipeline goes through
the following three steps, where we use the example inside
atomdnn/example to walk through the process.




- **Data pipeline**: users just need to provide the atomic structures
  saved as extxyz files (with lattice, atom coordinates, potential energy, atomic forces and stress),
  which are converted to lammps datafiles in order to compute
  descritpors and derivatives (saved as lammps dump files). We then
  create a Data object to read the descritpors and derivatives (from
  lammps dump files) as the inputs, and the potential energy, atomic force and stress (from
  extxyz files) as the outputs. The data object is finally saved as Tensorflow
  dataset, which is used for traning and also can be conveniently
  shared among collaborators. This data pipeline is designed for
  data flexibility and transparency.



Step 1: Read atomic structures
==============================
In the package, we put an example in the atomdnn/example
folder. Inside, the extxyz folder contains 50 extxyz files.
the following code is used to read the extxyz files and convert them to lammps data files:

.. code-block:: python

   from atomdnn.data import *
   xyzfile_name = './extxyz/example_extxyz.*'
   lmpdata_path = './lmpdata'
   convert_extxyz_to_lmpdata(xyzfile_name=xyzfile_name, lmpdata_path=lmpdata_path)


Step 2: Generate descriptors
=============================
We use LAMMPS to compute descriptors through two customized Compute
commands (respectively for descriptors and their derivatives). The
Compute subroutines can be found inside atomdnn/lammps. Copy the
compute subroutines along with the dump_local.cpp (a patch is added) to LAMMPS/src and compile LAMMPS to get the serial or mpi executables,
which can be used to compute descriptors.    


.. note::

   The ACSF parameters defined in this lammps input file should be consistant with
   the parameters saved along with the trained neural network which are
   used for predictions.


Step 3: Create data object
===========================
# Creat Data object and reead inputdata from LAMMPS dump files

### read inputdata function:

### read_inputdata (fp_filename=None, der_filename=None, image_num=None,            read_der=TFAtomDNN.compute_force)

- **fp_filename**: fingerprints file path, use wildcard * in the name for a serials of files.

- **fp_filename**: derivatives file name, use wildcard * in the name for a serials of files.

- **image_num**: if not set, read all images

- **read_der**: set to true if read derivative data


It will generate input data in the form of tensorflow tensors, which can be accessed using keys:

- **input_dict [ 'fingerprints' ] [ i ] [ j ] [ k ]** gives the k-th fingerprint of j-th atom in i-th image.
    
- **input_dict [ 'atom_type' ] [ i ] [ j ]** gives the atom type of j-th atom in i-th image.
    
- **input_dict [ 'dGdr' ] [ i ] [ j ] [ k ] [ m ]** gives the derivative of k-th fingerprint in j-th derivative data blcok of i-th image w.r.t m-th cooridinate.
    
- **input_dict [ 'center_atom_id' ] [ i ] [ j ]** gives the center atom id in j-th derivative data block of i-th image.
    
- **input_dict [ 'neighbor_atom_id' ] [ i ] [ j ]** gives the neighbor atom id in j-th derivative data block of i-th image. Note that the neighbors could be ghost atoms.
    
- **input_dict [ 'neighbor_atom_coord' ] [ i ] [ j ] [ m ] [ 0 ]** gives the m-th coordiate of neighbor atom in j-th derivative block of i-th image. Note that the last dimension of the array is 1 which is added for matrix multiplication in neural network force_stress_layer. 





