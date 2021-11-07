=====
Data
=====



The ``Data`` class creates a data object that handles reading inputs and outputs.

For example, ``mydata = Data()`` creates a Data object ``mydata``. After read inputs and outputs data using :func:`~atomdnn.data.Data.read_inputdata` and :func:`~atomdnn.data.Data.read_outputdata`, the data are stored as numpy array in two dictionaries, which can be accessed by ``mydata.input_dict['keys']`` and ``mydata.output_dict['key']``, where the ``'key'`` refers to input and output variables.

**Input data dictionary** ``'key'``:

    - ``'fingerprints'``: 3D array *[num_images, num_atoms, num_fingerprints]* of the fingerprints.

    - ``'atom_type'``: 2D array *[num_images, num_atoms]* of the atom element type, starting from 1 and sequentially increases.

    - ``'volume'``: 2D array *[num_images,1]* of cell volume, the last dimension is used for matrix multiplication.

    - ``'dGdr'``: 4D array *[num_images, num_der_paris, num_fingerprints, 3]* of the derivative of fingerprints w.r.t atom coordiante, the last dimension is three coordinates.

    - ``'neighbor_atom_coord'``: 4D array *[num_images, num_der_pairs, 3, 1]* to store the neighbor atom coordinates, the last dimension is added for the convenience of matrix multiplication in tensorflow.

    - ``'center_atom_id'``: 2D array *[num_images, num_der_pairs]* to store the center atom ID.

    - ``'neighbor_atom_id'``: 2D array *[num_images, num_der_pairs]* to store the neighbor atom ID, note that the neighbors could be ghost atoms.

**Output data dictionary** ``'keys'``:

    - ``'pe'``: 1D array *[num_images]* of potential energy.

    - ``'force'``: 3D array *[num_images, num_atoms, 3]* of atomic force.

    - ``'stress'``: 2D array *[num_images, 9]* of stress tensor with 9 components (6 independent).


Class
-----

.. autoclass:: atomdnn.data.Data
  :members:




Functions
---------
These functions are used to manipulate Tensorflow dataset

.. autofunction:: atomdnn.data.split_dataset


.. autofunction:: atomdnn.data.get_input_dict


.. autofunction:: atomdnn.data.get_output_dict


.. autofunction:: atomdnn.data.slice_dataset
