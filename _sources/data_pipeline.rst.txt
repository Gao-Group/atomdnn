=============
Data pipeline
=============

From the initial data of atomic structures to the tensorflow
dataset that can be used for training, a data pipeline goes through
the following three steps, which is demonstated in the example
script. This data pipeline is designed for convenient data
manipulation and data transparency.


Step 1: Create descriptor
-------------------------

  Users just need to provide the atomic structures saved as extxyz files (with lattice, atom coordinates, potential energy, atomic forces and stress),
  which are converted to lammps datafiles (with ASE, lmpdata files are not saved) in order to compute descriptors and derivatives (in the format of lammps dump files). LAMMPS has to be pre-compiled with the subroutines(compute and
  dump_local) inside atomdnn/lammps folder. The function to create descriptors is:

  .. code-block:: python

     create_descriptors(xyzfile_path = xyzfile_path, \
		        xyzfile_name = xyzfile_name, \
			lmpexe = lmpexe, \
			descriptors_path = descriptors_path, \
			descriptor = descriptor, \
			descriptor_filename = descriptor_filename, \
			der_filename = der_filename)


Step 2: Read inputs&outputs
---------------------------

  We use a python class object 'Data' to handle reading inputs and
  outputs. The inputs are descriptors and their derivatives and
  outputs include potential energies, atomic forces and stress.

  .. code-block:: python

    mdata = Data()
    mdata.read_inputdata(fp_filename=fp_filename,der_filename=der_filename)
    mdata.read_outputdata(xyzfile_path=xyzfile_path, xyzfile_name=xyzfile_name)


  This data object can be conveniently manipulated. For example, the inputs can be accessed as (assuming the Data object is called mdata):

    - **mdata.input_dict['fingerprints'][i][j][k]** gives the k-th fingerprint of j-th atom in i-th image.
    - **mdata.input_dict['atom_type'][i][j]** gives the atom type of j-th atom in i-th image.
    - **mdata.input_dict['volume'][i]** gives the volume of i-th image
    - **mdata.input_dict['dGdr'][i][j][k][m]** gives the derivative of k-th fingerprint in j-th derivative pair of i-th image w.r.t m-th cooridinate.
    - **mdata.input_dict['center_atom_id'][i][j]** gives the center atom id in j-th derivative pair of i-th image.
    - **mdata.input_dict ['neighbor_atom_id'][i][j]** gives the neighbor atom id in j-th derivative pair of i-th image. Note that the neighbors could be ghost atoms.
    - **mdata.input_dict ['neighbor_atom_coord'][i][j][m][0]** gives the m-th coordiate of neighbor atom in j-th derivative pair of i-th image. Note that the last dimension of the array is 1 which is added for matrix multiplication in neural network force_stress_layer.

  The outputs can be accessed as:

    - **mdata.output_dict['pe'][i]** gives the potential energy of i-th image.
    - **mdata.output_dict['force'][i][j]** gives the force vector of j-th atom in i-th image.
    - **mdata.output_dict['stress'][i]** gives the stress of i-th image


Step 3: Create tensorflow dataset
---------------------------------

  For the convenience of reuse and share the data, data are saved in the format of tensorflow dataset.

  .. code-block:: python

     tf_dataset = tf.data.Dataset.from_tensor_slices((mdata.input_dict,mdata.output_dict))
     tf.data.experimental.save(tf_dataset, dataset_path)
