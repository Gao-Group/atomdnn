import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import os
import atomdnn
from atomdnn.data import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MaxNLocator
from atomdnn.descriptor import create_descriptors, get_num_fingerprints
import shutil
import math
import pickle

if atomdnn.compute_force:
    input_signature_dict = [{"fingerprints": tf.TensorSpec(shape=[None,None,None], dtype=atomdnn.data_type, name="fingerprints"),
                    "atom_type": tf.TensorSpec(shape=[None,None], dtype=tf.int32, name="atom_type"),
                    "dGdr": tf.TensorSpec(shape=[None,None,None,None], dtype=atomdnn.data_type, name="dgdr"),
                    "center_atom_id": tf.TensorSpec(shape=[None,None], dtype=tf.int32, name="center_atom_id"),
                    "neighbor_atom_id": tf.TensorSpec(shape=[None,None], dtype=tf.int32, name="neighbor_atom_id"),
                    "neighbor_atom_coord": tf.TensorSpec(shape=[None,None,None,None], dtype=atomdnn.data_type, name="neighbor_atom_coord")}]
else:
    input_signature_dict = [{"fingerprints": tf.TensorSpec(shape=[None,None,None], dtype=atomdnn.data_type, name="fingerprints"),
                         "atom_type": tf.TensorSpec(shape=[None,None], dtype=tf.int32, name="atom_type")}]

    
            
  
class Network(tf.Module):
    """
    The nueral network is built based on tf.Module

    Args:
            elements(python list): list of element in the system, e.g. [C,O,H]
            descriptor(dictionary): descriptor parameters
            arch: network architecture, e.g. [10,10,10], 3 dense layers each with 10 neurons
            activation_function: any avialiable Tensorflow activation function, e.g. 'relu'
            weighs_initializer: the way to initialize weights, default one is tf.random.normal
            bias_initializer: the way to initialize bias, default one is tf.zeros
            import_dir: the directory of a saved model to be loaded
    """

    def __init__(self, elements=None, descriptor=None, arch=None, \
                 activation_function=None, weights_initializer=tf.random.normal, bias_initializer=tf.zeros, import_dir=None):
        """
        Initialize Network object.
        """
        
        super(Network,self).__init__()
        self.epsilon = atomdnn.epsilon

        if import_dir:
            self.inflate_from_file(import_dir)
        else:
            if arch:
                self.arch = arch
            else:
                self.arch = [10]
                print ('network arch is set to [10] by default.',flush=True)
                
            if activation_function:
                self.activation_function = activation_function
            else:
                self.activation_function = 'tanh'
                print ('activation function is set to tanh by default.',flush=True)
            self.tf_activation_function = tf.keras.activations.get(self.activation_function)

            if descriptor is not None:
                self.descriptor = descriptor
            else:
                raise ValueError ('Network has no descriptor input.')
                            
            if elements is not None:
                self.elements = elements 
            else:
                raise ValueError ('Network has no elements input.')

            self.num_fingerprints = get_num_fingerprints(descriptor,elements)
            self.data_type = atomdnn.data_type
            self.weights_initializer = weights_initializer
            self.bias_initializer = bias_initializer
            self.scaling = None
            self.validation = False
                    
            self.built = False
            self.params = [] # layer and node parameters
        
            # training parameter
            self.loss_fun=None # string
            self.optimizer=None # string
            self.tf_optimizer=None
            self.lr=None

            
        
        
    def inflate_from_file(self, import_dir):
        '''
        Inflate network object from a SavedModel.

        Args:
            import_dir: directory of a saved tensorflow neural network model
        '''
        imported = tf.saved_model.load(import_dir)
        
        print('Loading network information ...',flush=True)
        self.built = True
        self.arch = imported.saved_arch.numpy()
        print('  network arch: ',self.arch,flush=True)
        self.num_fingerprints = imported.saved_num_fingerprints.numpy()
        print('  number of fingerprints: ', self.num_fingerprints,flush=True)
        self.data_type = imported.saved_data_type.numpy().decode()
        print('  data type: ',self.data_type,flush=True)
        self.elements = [imported.saved_elements.numpy()[i].decode() for i in range(imported.saved_elements.shape[0])]
        print('  elements: ',list(self.elements),flush=True)
        
        self.descriptor = {}
        for key, value in imported.saved_descriptor.items():
            value = value.numpy()
            if isinstance(value, bytes):
                value = value.decode()
            self.descriptor[key] = value
        print('  symmetry function parameters: saved as <imported_model>.descriptor',flush=True)

        self.activation_function = imported.saved_activation_function.numpy().decode()
        self.tf_activation_function = tf.keras.activations.get(self.activation_function)
        print('  activation function: ',self.activation_function,flush=True)

        if hasattr(imported,'scaling'):
            self.scaling = imported.scaling
            self.scaling_factor = imported.scaling_factor
            print('  scaling: ',self.scaling.numpy().decode(),flush=True)
        else:
            self.scaling = None

        if hasattr(imported, 'dropout'):
            self.dropout = imported.dropout
            print('  dropout rate has been set to ', self.dropout.numpy())
        else:
            self.dropout = tf.Variable(0., dtype=self.data_type)

        self.loss_fun = imported.saved_loss_fun.numpy().decode()
        print('  loss function: ',self.loss_fun, flush=True )

        self.optimizer = imported.saved_optimizer.numpy().decode()
        self.tf_optimizer = tf.keras.optimizers.get(self.optimizer)
        print('  optimizer: ',self.optimizer,flush=True)

        self.loss_weights = {}
        for key in imported.saved_loss_weights:
             self.loss_weights[key] = imported.saved_loss_weights[key].numpy()
        print('  loss weights:', self.loss_weights, flush=True)

        if hasattr(imported, 'saved_loss_weights_history'):
            self.loss_weights_history = {}
            for key in imported.saved_loss_weights_history:
                self.loss_weights_history[key] = imported.saved_loss_weights_history[key].numpy().tolist()
            print('  loss weights history', flush=True)     


        if hasattr(imported, 'saved_softadapt_params'):
            self.softadapt_params = {}
            for key,value in imported.saved_softadapt_params.items():
                print('saved_softadapt_params.VALUE():', value, ' | type:', type(value))
                self.softadapt_params[key] = value.numpy()
               

        self.lr = imported.saved_lr.numpy()
        K.set_value(self.tf_optimizer.learning_rate, self.lr)
        print("  learning rate: %5.3E"%self.lr,flush=True)

        if imported.save_training_history == True:
            with open(import_dir+'/training_history_data', 'rb') as in_: 
                history = pickle.load(in_)
            if hasattr(imported,'decay'):
                if hasattr(imported, 'val_loss'):
                    self.lr_history, self.train_loss, self.val_loss = history
                    msg = "  lr_history, train_loss and val_loss are loaded"
                else:
                    attributes = dir(imported)
                    print('attrobutes:', attributes)
                    self.lr_history, self.train_loss = history
                    msg = "  lr_history and train_loss are loaded"
            else:
                if hasattr(imported, 'val_loss'):
                    self.train_loss, self.val_loss = history
                    msg = "  train_loss and val_loss are loaded"
                else:
                    self.train_loss = history
                    msg = "  train_loss are loaded"
            print(msg, flush=True)
                                      
        try:
            self.params = []
            for param in imported.params:
                self.params.append(param)
        except AttributeError:
            raise AttributeError('imported_object does not have params attribute.')
 
        print('Network has been inflated! self.built:',self.built,flush=True)
        
        
        
    @tf.function(input_signature=input_signature_dict)    
    def __call__(self, x):
        """
        This is called when evaluate the network using C API for LAMMPS.
        """
        if len(x)==0:
            print('__call__ method - input is empty.')
            return {'pe':None,'force': None, 'stress':None}
        else:
            return self._predict(x)
            
        
    def _build(self):
        """
        Initialize the weigths and biases.
        """

        # initialize dropout layer
        self.dropout = tf.Variable(0., dtype=self.data_type)

        # Declare layer-wise weights and biases
        for element in self.elements:
            self.W1 = tf.Variable(
                self.weights_initializer(shape=(self.num_fingerprints,self. arch[0]), dtype=self.data_type))
            self.b1 = tf.Variable(self.bias_initializer(shape=(1,self.arch[0]), dtype=self.data_type))
            self.params.append(self.W1)
            self.params.append(self.b1)
            nlayer = 1
        
            for nneuron_layer in self.arch[1:]:
                self.params.append(tf.Variable(self.weights_initializer(shape=(self.arch[nlayer], self.arch[nlayer-1]), dtype=self.data_type)))
                self.params.append(tf.Variable(self.bias_initializer([1, self.arch[nlayer]], dtype=self.data_type)))
                nlayer+=1
            
            self.params.append(tf.Variable(self.weights_initializer(shape=(self.arch[-1], 1), dtype=self.data_type)))
            self.params.append(tf.Variable(self.bias_initializer([1,], dtype=self.data_type)))

        # if hasattr(self, 'regularization') and self.regularization is not None:
        #     self.params.append(tf.Variable(np.zeros(1), dtype=self.data_type, name='regularization'))
            # self.params.append(tf.Variable(0.), name='regularizer', dtype=self.data_type)

        # for lw in self.loss_weights:
        #     self.params.append(self.loss_weights[lw])
        # print('network.py -> _build() -> self.loss_weights:', self.loss_weights)

    
    def compute_pe (self, fingerprints, atom_type):
        '''
        Forward pass of the network to compute potential energy. Parallel networks are used for multiple atom types.
        
        Args:
            fingerprints: 3D array *[batch_size,atom_num,fingerprints_num]*
            atom_type: 2D array *[batch_size, atom_num]*

        Returns:
            total potential energy and per-atom potential energy
        '''
        nimages = tf.shape(atom_type)[0]
        max_natoms = tf.shape(atom_type)[1]
        
        atom_pe = tf.zeros([nimages,max_natoms,1],dtype=self.data_type)

        nlayer = len(self.arch)+1 # include one linear layer at the end

        nelements = len(self.elements)
        
        for i in range(nelements):
            type_id = i + 1
            type_array = tf.ones([nimages,max_natoms],dtype='int32')*type_id
            type_mask = tf.cast(tf.math.equal(atom_type,type_array),dtype=self.data_type)
            type_onehot = tf.linalg.diag(type_mask)
            fingerprints_i = tf.matmul(type_onehot,fingerprints)
            
            # if dropout, it happens before first dense layer
            dropped = tf.nn.dropout(fingerprints_i, rate=self.dropout)

            # first dense layer
            Z = tf.matmul(dropped, self.params[nlayer*2*i]) + self.params[nlayer*2*i+1]
            Z = self.tf_activation_function(Z)

            # sequential dense layer
            for j in range(1,nlayer-1):
                Z = tf.matmul(Z,self.params[nlayer*2*i+j*2]) + self.params[nlayer*2*i+j*2+1]
                Z = self.tf_activation_function(Z)

            # linear layer
            Z = tf.matmul(Z, self.params[nlayer*2*(i+1)-2]) + self.params[nlayer*2*(i+1)-1]

            # apply the mask
            mask = tf.reshape(type_mask,[nimages,max_natoms,1])
            atom_pe += Z * mask
        
        regularizer = 0.
        if hasattr(self, 'regularization') and self.regularization is not None:
            for i,layer_i in enumerate(self.params[0:-2]):
                # we don't include bias terms when regularizing
                if i%2==0:
                    regularizer += tf.nn.l2_loss(layer_i)

            # regularizer = tf.Variable(np.array([self.regularization_tf.get_config()[self.regularization] * regularizer]), dtype=self.data_type, name='regularizer')
            # self.params[-1] = regularizer.numpy()
            regularizer = self.regularization_tf.get_config()[self.regularization] * regularizer
            # print('Debug!-> regularizer:', regularizer)
        # else:
        #     print('Debug!-> no regularizer')

        total_pe = tf.reshape(tf.math.reduce_sum(atom_pe, axis=1),[nimages]) + regularizer

        return total_pe, atom_pe
        
    
    def compute_force_stress (self, dEdG, input_dict):
        """
        Compute force and stress.
        
        Args:
            dEdG: derivatives of per-atom potential energy w.r.t. descriptors
            input_dict: input dictionary data
        
        Returns: 
            force per atom and stress tensor
        """
                
        if self.scaling=='std':
            dEdG = dEdG/self.scaling_factor[1]
        elif self.scaling=='norm':
            dEdG = dEdG/(self.scaling_factor[1] - self.scaling_factor[0])
            
        dGdr = input_dict['dGdr']
        if not tf.is_tensor(dGdr):
            tf.convert_to_tensor(dGdr,dtype=self.data_type)
            
        center_atom_id = input_dict['center_atom_id']
        if not tf.is_tensor(center_atom_id):
            tf.convert_to_tensor(center_atom_id,dtype='int32')
       
        neighbor_atom_id = input_dict['neighbor_atom_id']
        if not tf.is_tensor(neighbor_atom_id):
            tf.convert_to_tensor(neighbor_atom_id,dtype='int32')
        
        neighbor_atom_coord = input_dict['neighbor_atom_coord']
        if not tf.is_tensor(neighbor_atom_coord):
            tf.convert_to_tensor(neighbor_atom_coord,dtype=self.data_type)
    
        num_image = tf.shape(center_atom_id)[0]
        max_block = tf.shape(center_atom_id)[1]
        num_atoms = len(input_dict['fingerprints'][0])
        
        # form dEdG matrix corresponding to the blocks in dGdr data file
        center_atom_id_reshape = tf.reshape(center_atom_id,shape=[num_image,-1,1])
        dEdG_block = tf.gather_nd(dEdG,center_atom_id_reshape,batch_dims=1)     
        dEdG_block = tf.reshape(dEdG_block, [num_image,max_block,1,self.num_fingerprints])
        
        # compute force
        force_block = tf.matmul(dEdG_block, dGdr)
        force_block_reshape = tf.reshape(force_block,[num_image,max_block,3])
        neighbor_onehot = tf.one_hot(neighbor_atom_id, depth=num_atoms,axis=1, dtype=self.data_type)     
        force = -tf.matmul(neighbor_onehot,force_block_reshape)        
        
        # compute stress    
        stress_block = tf.reshape(tf.matmul(neighbor_atom_coord,force_block),[num_image,max_block,9])     
        stress = tf.reduce_sum(stress_block,axis=1)
        stress = tf.math.divide(stress,input_dict['volume']) * atomdnn.stress_unit_convert 
        return force, stress


  
    def _compute_force_stress (self, dEdG, input_dict):
        """
        Only used for __call__ method and C API.
        The same as compute_force_stress but return force and stress for derivative pairs.
        """

        if self.scaling is not None:
            if tf.math.equal(self.scaling, 'std'):
                dEdG = dEdG/self.scaling_factor[1]
            elif tf.math.equal(self.scaling, 'norm'):
                dEdG = dEdG/(self.scaling_factor[1] - self.scaling_factor[0])
        
        fingerprints = input_dict['fingerprints']
        dGdr = input_dict['dGdr']
        center_atom_id = input_dict['center_atom_id']
        neighbor_atom_coord = input_dict['neighbor_atom_coord']  
    
        num_image = tf.shape(center_atom_id)[0]
        max_block = tf.shape(center_atom_id)[1]
        num_atoms = tf.shape(fingerprints)[1]
        
        # form dEdG matrix corresponding to the blocks in dGdr data file  
        center_atom_id_reshape = tf.reshape(center_atom_id,shape=[num_image,-1,1])
        dEdG_block = tf.gather_nd(dEdG,center_atom_id_reshape,batch_dims=1)     
        dEdG_block = tf.reshape(dEdG_block, [num_image,max_block,1,self.num_fingerprints])
        
        # compute force block      
        force_block = -tf.matmul(dEdG_block, dGdr)
        
        # compute stress block    
        stress_block = tf.reshape(tf.matmul(neighbor_atom_coord,force_block),[num_image,max_block,9])     
        
        return force_block, stress_block
    

        
    
    def predict(self, input_dict,compute_force=atomdnn.compute_force,output_atom_pe=False, training=False,evaluation=False,save_to_file=None):
        """
        Predict energy, force and stress.

        Args:
            input_dict: input dictionary data
            compute_force(bool): True to compute force and stress
            output_atom_pe(bool): True to output potential energies for atoms
            training(bool): True when used during training

        Returns:
            dictionary: potential energy, force and stress
        """
        if not self.built:
            self._build()
            self.built = True

        fingerprints = input_dict['fingerprints']
        
        if self.num_fingerprints!=len(fingerprints[0][0]):
            raise ValueError('Network and inputdata have different number of fingerprints, check descriptor parameters.\n self.num_fingerprints=',\
                self.num_fingerprints,'\n len(fingerprints[0][0]):',len(fingerprints[0][0]))
        
        if not tf.is_tensor(fingerprints):
            fingerprints = tf.convert_to_tensor(fingerprints)

        if not training:          
            if self.scaling == 'std':  # standardize with the mean and deviation
                fingerprints = (fingerprints - self.scaling_factor[0])/self.scaling_factor[1]
            elif self.scaling == 'norm': # normalize with the minimum and maximum
                fingerprints = (fingerprints - self.scaling_factor[0])/(self.scaling_factor[1] - self.scaling_factor[0])
        
        if not compute_force:
            total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'])
            if training:
                nimages = tf.shape(input_dict['atom_type'])[0]
                natoms_tensor = tf.math.count_nonzero(input_dict['atom_type'],1, keepdims=True)
                natoms_tensor = tf.cast(tf.reshape(natoms_tensor,[nimages]),dtype=self.data_type)
                pe_per_atom = tf.divide(total_pe,natoms_tensor)
                return {'pe_per_atom':pe_per_atom} 
            else:
                return {'pe':total_pe.numpy()}

        else: 
            with tf.GradientTape() as dEdG_tape:
                dEdG_tape.watch(fingerprints)
                total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'])
            dEdG = dEdG_tape.gradient(atom_pe, fingerprints)
            force, stress = self.compute_force_stress(dEdG, input_dict)
            if training or evaluation:
                nimages = tf.shape(input_dict['atom_type'])[0]
                natoms_tensor = tf.math.count_nonzero(input_dict['atom_type'],1, keepdims=True)
                natoms_tensor = tf.cast(tf.reshape(natoms_tensor,[nimages]),dtype=self.data_type)
                pe_per_atom = tf.divide(total_pe,natoms_tensor)
                return {'pe_per_atom': pe_per_atom, 'force':force, 'stress':stress}
            else:
                if not output_atom_pe:
                    return  {'pe':total_pe.numpy(), 'force':force.numpy(), 'stress':stress.numpy()}
                else:
                    return  {'pe':total_pe.numpy(), 'atom_pe':atom_pe.numpy(), 'force':force.numpy(), 'stress':stress.numpy()}

                
                
    def inference(self, filename, format, **kwargs):
        """
        Predict potential energy, force and stress directly from one atomic structure input file. This function first computes descriptors and then call predict function.

        Arg:
            filename: name of the atomic structure input file
            format: 'lammp-data','extxyz','vasp' etc. See complete list on https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read 
            kwargs: used to pass optional file styles
        """
        atomdnn_data = create_descriptors(self.elements,filename,self.descriptor,format,silent=True,remove_descriptors_folder=True,**kwargs)

        outputs = self.predict(atomdnn_data.get_input_dict())

        return outputs
                     
   
    def _predict(self, input_dict, compute_force=atomdnn.compute_force):
        """
        Only used for __call__ method and C API. 
        Same as :func:`~atomdnn.network.Network.predict`, but returns per-atom potential energy, force and stress for derivative pairs
        """

        fingerprints = input_dict['fingerprints']

        if self.scaling is not None:
            if tf.math.equal(self.scaling,'std'):  # standardize with the mean and deviation
                fingerprints = (fingerprints - self.scaling_factor[0])/self.scaling_factor[1]
            elif tf.math.equal(self.scaling,'norm'): # normalize with the minimum and maximum
                fingerprints = (fingerprints - self.scaling_factor[0])/(self.scaling_factor[1] - self.scaling_factor[0])
        
        if not compute_force:
            total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'])
            return {'pe':atom_pe}                       
        else: 
            with tf.GradientTape() as dEdG_tape:
                dEdG_tape.watch(fingerprints)
                total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'])          
            dEdG = dEdG_tape.gradient(atom_pe, fingerprints)
            force_block, stress_block = self._compute_force_stress(dEdG, input_dict)         
            return {'atom_pe':atom_pe, 'force':force_block, 'stress':stress_block}

        

    def evaluate(self, dataset, batch_size=None, return_eval_loss=False, compute_force=atomdnn.compute_force):
        """
        Do evaluation on trained model.
        
        Args:
            dataset: tensorflow dataset used for evaluation
            batch_size: default is total data size
            compute_force(bool): True for computing force and stress
        """

        if batch_size is None:
            batch_size = dataset.cardinality().numpy()
            
        for step, (input_dict, output_dict) in enumerate(dataset.batch(batch_size)):
            y_predict = self.predict(input_dict, compute_force=compute_force, evaluation=True)
            batch_loss = self.loss(output_dict,y_predict)
            if step==0:
                eval_loss = batch_loss
            else:
                for key in batch_loss:
                    eval_loss[key] = batch_loss[key] + eval_loss[key]
        for key in batch_loss:
            eval_loss[key] = eval_loss[key]/(step+1)

        print('Evaluation loss is:')    
        for key in eval_loss:
            print('%15s:  %15.4e' % (key, eval_loss[key]))
        if 'total_loss' in eval_loss:
            print('The total loss is computed using the loss weights', *['- %s: %.2f' % (key,value) for key, value in self.loss_weights.items()])

        if return_eval_loss:
            return eval_loss
                                                

    def loss_function (self,true, pred):
        """
        Build-in tensorflow loss functions can be used.
        Customized loss function can be also defined here.

        Args:
            true (tensorflow tensor): true output data
            pred (tensorflow tensor): predicted results
        """
        if self.loss_fun == 'rmse':
            tf_loss_fun = tf.keras.losses.get('mse')
        else:
            tf_loss_fun = tf.keras.losses.get(self.loss_fun)
        return tf_loss_fun(true, pred)
            
        #     return tf.sqrt(tf.maximum(tf_loss_fun(true,pred), 1e-9))
        # else:
        #     tf_loss_fun = tf.keras.losses.get(self.loss_fun)
        #     return tf_loss_fun(true, pred)
            
        
        
    def loss(self, true_dict, pred_dict, training=False, validation=False):
        """
        Compute losses for energy, force and stress.
        
        Args:
            true_dict(dictionary): dictionary of outputs data
            pred_dict(dictionary): dictionary of predicted results
            training(bool): True for loss calculation during training
            validation(bool): True for loss calculation during validation  

        Returns:
            if training is true, return total_loss and loss_dict(loss dictionary), otherwise(for evaluation) return loss_dict
        """

        loss_dict={}
        
        loss_dict['pe_loss'] = self.loss_function(true_dict['pe_per_atom'], pred_dict['pe_per_atom'])

        if 'force' in pred_dict and 'force' in true_dict: # compute force loss
            # when evaluation OR train/validation using force OR all losses are requested
            if ((not training and not validation) or ((training or validation) and self.loss_weights['force']!=0)) or self.compute_all_loss:
                loss_dict['force_loss'] = tf.reduce_mean(self.loss_function(true_dict['force'],pred_dict['force']))
                
        if 'stress' in pred_dict and 'stress' in true_dict: # compute stress loss
            # when evaluation OR train/validation using stress OR all losses are requested
            if ((not training and not validation) or ((training or validation) and (self.loss_weights['stress']!=0))) or self.compute_all_loss: 

                # select upper triangle elements (xx,yy,zz,yz,xz,xy) of the stress tensor
                indices = [[[0],[4],[8],[5],[2],[1]]] * len(pred_dict['stress'])
                stress = tf.gather_nd(pred_dict['stress'], indices=indices, batch_dims=1)
                loss_dict['stress_loss'] = tf.reduce_mean(self.loss_function(true_dict['stress'],stress))
                # print('pred_stress.numpy()[0][0]:', stress.numpy()[0][0], ' | true_stress.numpy()[0][0]:', true_dict['stress'].numpy()[0][0])
                #loss_dict['stress_loss'] = tf.reduce_mean(self.loss_function(true_dict['stress'],pred_dict['stress']))

        if self.loss_fun=='rmse':
            for key in loss_dict.keys():
                loss_dict[key] = tf.sqrt(loss_dict[key])


        # if self.first_epoch_passed:
        #     total_loss = (loss_dict['pe_loss'] * self.loss_weights['pe'] +  
        #                   loss_dict['force_loss'] * self.loss_weights['force'] +
        #                   loss_dict['stress_loss'] * self.loss_weights['stress'])
        #     loss_dict['total_loss'] = total_loss

        # else:
        #     total_loss = (loss_dict['pe_loss'] + loss_dict['force_loss'] + loss_dict['stress_loss'])
        #     loss_dict['total_loss'] = total_loss
        #     self.first_epoch_passed = True

                
        # ============= only pe used for training
        if 'force' not in self.loss_weights and 'stress' not in self.loss_weights:
            total_loss = loss_dict['pe_loss']

        elif 'force' in self.loss_weights and 'stress' in self.loss_weights: # pe, force and stress used for training
            total_loss = (loss_dict['pe_loss'] * self.loss_weights['pe'] +  
                          loss_dict['force_loss'] * self.loss_weights['force'] +
                          loss_dict['stress_loss'] * self.loss_weights['stress'])
            loss_dict['total_loss'] = total_loss

        elif 'force' in self.loss_weights: # pe and force used for training 
            total_loss = loss_dict['pe_loss'] * self.loss_weights['pe'] + loss_dict['force_loss'] * self.loss_weights['force']
            loss_dict['total_loss'] = total_loss
        elif self.loss_weights['stress']!=0: # pe and stress used for training
            total_loss = loss_dict['pe_loss'] * self.loss_weights['pe'] + loss_dict['stress_loss'] * self.loss_weights['stress']
            loss_dict['total_loss'] = total_loss
        else:
            raise ValueError('loss_weights is not set correctly.')

        if training:
            self.count_loss_calculations['train'] += 1
            return total_loss,loss_dict
        else:
            return loss_dict


    def update_loss_weights(self, epoch, loss_dict):

        use_stress = True
        if 'stress' not in self.loss_weights or self.loss_weights['stress']==0:
            use_stress = False

        if epoch%self.softadapt_params['n']==0:
            self.lw_per_group_of_epochs = {}
            for key in self.loss_weights:
                self.lw_per_group_of_epochs[key] = []

        if self.first_epoch_passed:
            b = self.softadapt_params['beta']
            f_iminus1_1 = self.f_iminus1_1
            f_iminus1_2 = self.f_iminus1_2
            if use_stress:
                f_iminus1_3 = self.f_iminus1_3

            s_i1 = loss_dict['pe_loss'] - f_iminus1_1
            s_i2 = loss_dict['force_loss'] - f_iminus1_2
            if use_stress:
                s_i3 = loss_dict['stress_loss'] - f_iminus1_3
                s_i_norm = tf.norm(tf.Variable(np.array([s_i1, s_i2, s_i3]), dtype=self.data_type), axis = None)
            else:
                s_i_norm = tf.norm(tf.Variable(np.array([s_i1, s_i2]), dtype=self.data_type), axis = None)
            
            s_i1 = (loss_dict['pe_loss'] - f_iminus1_1)/s_i_norm
            s_i2 = (loss_dict['force_loss'] - f_iminus1_2)/s_i_norm
            if use_stress:
                s_i3 = (loss_dict['stress_loss'] - f_iminus1_3)/s_i_norm

            if 'loss_weighted' in self.softadapt_params and self.softadapt_params['loss_weighted']==True:
                if use_stress:
                    alpha_normaliz_factor = loss_dict['pe_loss']*tf.exp(b*s_i1) +\
                                            loss_dict['force_loss']*tf.exp(b*s_i2) +\
                                            loss_dict['stress_loss']*tf.exp(b*s_i3) + self.epsilon
                else:
                    alpha_normaliz_factor = loss_dict['pe_loss']*tf.exp(b*s_i1) +\
                                            loss_dict['force_loss']*tf.exp(b*s_i2) + self.epsilon


                alpha_i1 = loss_dict['pe_loss']     * tf.exp(b*s_i1)/alpha_normaliz_factor
                alpha_i2 = loss_dict['force_loss']  * tf.exp(b*s_i2)/alpha_normaliz_factor
                if use_stress:
                    alpha_i3 = loss_dict['stress_loss'] * tf.exp(b*s_i3)/alpha_normaliz_factor
            else:
                if use_stress:
                    alpha_normaliz_factor = tf.exp(b*s_i1) + tf.exp(b*s_i2) + tf.exp(b*s_i3) + self.epsilon
                else:
                    alpha_normaliz_factor = tf.exp(b*s_i1) + tf.exp(b*s_i2) + self.epsilon

                alpha_i1 = tf.exp(b*s_i1)/alpha_normaliz_factor
                alpha_i2 = tf.exp(b*s_i2)/alpha_normaliz_factor
                if use_stress:
                    alpha_i3 = tf.exp(b*s_i3)/alpha_normaliz_factor

            self.loss_weights['pe'] = alpha_i1
            self.lw_per_group_of_epochs['pe'].append(alpha_i1.numpy())

            self.loss_weights['force'] = alpha_i2
            self.lw_per_group_of_epochs['force'].append(alpha_i2.numpy())

            if use_stress:
                self.loss_weights['stress'] = alpha_i3
                self.lw_per_group_of_epochs['stress'].append(alpha_i3.numpy())

            # tf.print(f'  epoch:{epoch} - loss_weights:', self.loss_weights['pe'], self.loss_weights['force'], self.loss_weights['stress'])

        else:
            self.first_epoch_passed = True
            self.prev_lw_per_group_of_epochs = {}
            for key in self.loss_weights:
                self.lw_per_group_of_epochs[key].append(self.loss_weights[key])
                self.prev_lw_per_group_of_epochs[key] = self.loss_weights[key] 
            # print('FIRST EPOCH! self.loss_weights:', self.loss_weights, flush=True)

        self.f_iminus1_1 = loss_dict['pe_loss']
        self.f_iminus1_2 = loss_dict['force_loss']
        if use_stress:
            self.f_iminus1_3 = loss_dict['stress_loss']
        # tf.print('  loss_weights:', self.loss_weights['pe'], self.loss_weights['force'], self.loss_weights['stress'])
        # print(60*'-')
        if (epoch+1)%self.softadapt_params['n']==0:
            # print('record_lw -> self.lw_per_group_of_epochs[\'pe_loss\']=', self.lw_per_group_of_epochs['pe_loss'], np.mean(self.lw_per_group_of_epochs['pe_loss']))
            # self.loss_weights['pe'] = np.mean(self.lw_per_group_of_epochs['pe_loss'])
            # print('  UPDATe EPOCH!')
            for key in self.loss_weights:    
                self.loss_weights[key] = np.mean(self.lw_per_group_of_epochs[key])
                self.prev_lw_per_group_of_epochs[key] = np.mean(self.lw_per_group_of_epochs[key])
                # print('  self.loss_weights[', key,']:', self.loss_weights[key], 'len=', len(self.lw_per_group_of_epochs[key]), flush=True)

        for key in self.loss_weights:    
            self.loss_weights_history[key].append(self.prev_lw_per_group_of_epochs[key])
            # print('  appending to loss_weights_history[',key,']:', self.prev_lw_per_group_of_epochs[key])


    def decay_lr(self, epoch):
        decayed_value = self.decay['initial_lr'] * math.pow(self.decay['decay_rate'],np.floor((1+epoch)/self.decay['decay_steps']))
        if decayed_value < self.decay['min_lr']:
            self.lr = self.decay['min_lr']
        else:
            self.lr = decayed_value
            
        K.set_value(self.tf_optimizer.learning_rate, self.lr)        

     
    def train_step(self, input_dict, output_dict):
        """
        A single training step.

        Args:
            input_dict: input dictionary data
            output_dict: output dictionary data

        Returns:
            loss dictionary
        """
        with tf.GradientTape() as tape: # automatically watch all tensorflow variables 
            if self.loss_weights['force']==0 and self.loss_weights['stress']==0 and not self.compute_all_loss:
                pred_dict = self.predict(input_dict, training=True, compute_force=False)
            else:
                pred_dict = self.predict(input_dict, training=True, compute_force=True)

            total_loss,loss_dict = self.loss(output_dict, pred_dict, training=True)

        grads = tape.gradient(total_loss, self.params)
        self.tf_optimizer.apply_gradients(zip(grads, self.params))

        # print('lw:')
        # for param_lw,lw in zip(self.params[-3:], self.loss_weights):
        #     print(f'self.params:{param_lw.numpy():0.5f}  self.loss_weights:{self.loss_weights[lw].numpy():0.5f}')
        # print('\n')

        return loss_dict


    
    def validation_step(self, input_dict, output_dict):
        """
        A single validation step.
 
        Args:
            input_dict: input dictionary data
            output_dict: output dictionary data

        Returns:
            loss dictionary
        """
        if self.loss_weights['force']==0 and self.loss_weights['stress']==0 and not self.compute_all_loss:
            pred_dict = self.predict(input_dict,training=True,compute_force=False)
        else:
            pred_dict = self.predict(input_dict,training=True,compute_force=True)
        loss_dict = self.loss(output_dict, pred_dict,validation=True)
        return loss_dict
    

    def compute_scaling_factors(self,train_dataset):
        """
        Compute scaling factors using training dataset.
        """
        self.scaling_factor = []
        fingerprints = get_input_dict(train_dataset)['fingerprints']
        if self.scaling == 'std':
            self.scaling_factor.append(tf.math.reduce_mean(fingerprints,axis=[0,1]))  
            self.scaling_factor.append(tf.math.reduce_std(fingerprints,axis=[0,1]))
        if self.scaling == 'norm':
            self.scaling_factor.append(tf.math.reduce_min(fingerprints,axis=[0,1]))
            self.scaling_factor.append(tf.math.reduce_max(fingerprints,axis=[0,1]))
        self.scaling_factor = tf.Variable(self.scaling_factor) # convert to tf.Variable for saving

        
    def scaling_dataset(self,dataset):
        """
        Scaling a dataset with the scaling factors calculated with :func:`~atomdnn.network.Network.compute_scaling_factors`.
        """
        def map_fun(input_dict,output_dict):
            if self.scaling == 'std':
                input_dict['fingerprints'] = (input_dict['fingerprints'] - self.scaling_factor[0])/self.scaling_factor[1]
            if self.scaling == 'norm':
                input_dict['fingerprints'] = (input_dict['fingerprints'] - self.scaling_factor[0])/(self.scaling_factor[1] - self.scaling_factor[0])
            return input_dict,output_dict
        return dataset.map(map_fun)


    
    
    def train(self, train_dataset, valid_dataset=None, regularization=None, regularization_param=None, cache=True, early_stop=None, dropout=None, nepochs_checkpoint=None, figfolder=None, savefolder=None, scaling=None, batch_size=None, epochs=None, loss_fun=None, \
              optimizer=None, lr=None, decay=None, loss_weights=None, initial_loss_weights=None, softadapt_params=None, compute_all_loss=False, use_stress=True, shuffle=True, buffer_size=None, append_loss=False, output_freq=1):
        """
        Train the neural network.

        Args:
            train_dataset(tfdataset): dataset used for training
            valid_dataset(tfdataset): dataset used for validation
            cache:  if dataset can fit into RAM, cache the dataset after the initial loading and preprocessing steps can reduce training time
            early_stop(dictionary): condition for stop training, \
                                    e.g. {'train_loss':[0.01,3]} means stop when train loss is less than 0.01 for 3 times
            scaling(string): = None, 'std' or 'norm'
            batch_size: the training batch size
            epochs: training epochs
            loss_fun(string): loss function, 'mae', 'mse', 'rmse' and others
            opimizer(string): any Tensorflow optimizer such as 'Adam'
            lr(float): learning rate, it is ignored if decay is provided
            decay(dictionary): parameters for exponentialDecay, keys are: 'initial_lr','decay_steps', 'decay_rate' and 'min_lr'
            loss_weights(dictionary): weights assigned to loss function, e.g. {'pe':1,'force':1,'stress':0.1}  

            softadapt_params(dictiorary): parameters for adaptive loss weights, e.g. {'beta':0.1,'loss_weighted':True}
                - 'beta'>0: More weight to worst performing term in loss function
                - 'beta'<0: Favor best performing term in loss function
                - 'loss_weighted': True/False. Activate/deactivate Loss Weighted approach.
                - 'n': Update loss weights every `n` number of epochs.
                For more, see: https://doi.org/10.48550/arXiv.1912.12355

            compute_all_loss(bool): compute loss for force and stress even when they are not used for training
            shuffle(bool): shuffle training dataset during training
            buffer_size: dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, \
                         defautl is train_dataset.cardinality().numpy()
            append_loss(bool): append loss history 
        """

        if loss_weights is None:
            if hasattr(self,'loss_weights'):
                print('Loss weights are set to the value from imported model:', self.loss_weights, flush=True)
            else:
                if softadapt_params is None:
                    if use_stress:
                        self.loss_weights = {'pe':1.0,'force':0,'stress':0}
                    else:
                        self.loss_weights = {'pe':1.0,'force':0}

                    print('Loss weights are set to default fixed value:', self.loss_weights, flush=True)
                    self.softadapt_params['n'] = 1
                    print('Softadapt param `n` has been set to default value:', self.softadapt_params['n'], flush=True)

                elif initial_loss_weights:
                    self.loss_weights = initial_loss_weights
                    self.softadapt_params = softadapt_params
                    
                    if 'n' not in self.softadapt_params:
                        self.softadapt_params['n'] = 1

                    print('Softadapt parameters have been set for adaptive weights.', self.softadapt_params, flush=True)
                else:
                    if use_stress:
                        self.loss_weights = {'pe' : 0.33, 'force' : 0.33, 'stress': 0.33}
                        # self.loss_weights = {'pe' : 0.05, 'force' : 0.90, 'stress': 0.05}
                    else:
                        self.loss_weights = {'pe' : 0.5, 'force' : 0.5}
                    self.softadapt_params = softadapt_params
                    if 'n' not in self.softadapt_params:
                        self.softadapt_params['n'] = 1
                    print('Softadapt parameters have been set for adaptive weights.', self.softadapt_params, flush=True)
        else:
            if softadapt_params is not None:
                raise ValueError('Please use either `loss_weights` for fixed loss weights or `softadapt_params` for adaptive loss weights.')
            self.loss_weights = loss_weights


        print('self.loss_weights:', self.loss_weights)
        if not append_loss:
            self.loss_weights_history = {}
            for key in self.loss_weights:
                # self.loss_weights[key] = tf.Variable(self.loss_weights[key],dtype=self.data_type)
                self.loss_weights_history[key] = []
        
        if regularization:
            if isinstance(regularization, str):
                self.regularization = regularization
                self.regularization_tf = tf.keras.regularizers.get(regularization)
                if regularization_param:
                    self.regularization_tf = self.regularization_tf.from_config({self.regularization:regularization_param})
                print('Regularization has been set to:', self.regularization_tf.get_config())
            else:
                raise ValueError('Regularizer name not recognized.')


        self.count_loss_calculations = {'train':0, 'valid':0}

            
        if not self.built:
            self._build()
            self.built = True
                        
        if scaling:
            if scaling != 'std' and scaling != 'norm':
                raise ValueError('Scaling needs to be \'std\' (standardization) or \'norm\'(normalization).')
            else:
                self.scaling = tf.Variable(scaling)
        elif hasattr(self,'scaling'):
            print('Scaling is set to %s from imported model.'%self.scaling.numpy().decode(),flush=True)
        else:
            self.scaling = None
            print('Scaling is not applied to data.',flush=True)

        if not hasattr(self,'lr_history'):
            self.lr_history = []
            
        if decay:
            for key in {'initial_lr','decay_steps','decay_rate'}:
                if decay.get(key)==None:
                    raise ValueError(key + ' is not in decay.')
            if decay.get('min_lr') == None:
                decay['min_lr'] = 1e-5
                print('Minimum lr is set to 1e-5 by default for decaying lr.')
            self.decay = decay
            self.lr = decay['initial_lr']
            self.lr_history.append(decay['initial_lr'])
            print("Initial learning rate is set to %.3e."%self.lr,flush=True)
        else:
            if lr:
                self.lr = lr
                print ("Learning rate is set to %.3e."%self.lr,flush=True)
            elif hasattr(self,'lr'):
                print("Learning rate is set to %.3e from imported model."%self.lr,flush=True)
            else:
                self.lr = 0.001
                print ("Learning rate is set to 0.001 by default.",flush=True)                        

        self.lr_history.append(self.lr)

        if optimizer:
            self.optimizer = optimizer
            self.tf_optimizer = tf.keras.optimizers.get(optimizer)
            #self.tf_optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
            K.set_value(self.tf_optimizer.learning_rate, self.lr)
        elif hasattr(self,'optimizer'):
            print("Optimizer is set to %s from imported model."%self.optimizer,flush=True)
        else:
            self.optimizer = 'Adam'
            self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
            print ("Optimizer is set to Adam by default.",flush=True)
    
            
        if loss_fun:
            self.loss_fun = loss_fun 
        elif hasattr(self,'loss_fun'):
            print("Loss function is set to %s from imported model."%self.loss_fun,flush=True)
        else:
            self.loss_fun = 'rmse'
            print ("Loss function is set to rmse by default.",flush=True)
                                   
        if not epochs:
            epochs = 1
            print ("epochs is set to 1 by default.",flush=True)

        if batch_size is None:
            batch_size = 30
            print ("batch_size is set to 30 by default.",flush=True)    


        if dropout:
            self.dropout = tf.Variable(dropout, dtype=self.data_type)
        else:
            self.dropout = tf.Variable(0., dtype=self.data_type)

        if early_stop:
            if 'val_loss' in early_stop and not valid_dataset:
                raise ValueError('No validation data for early stopping.')
           
        if not hasattr(self,'train_loss') or not append_loss:
            self.train_loss={}
         
        if valid_dataset and (not hasattr(self,'val_loss') or not append_loss):
            self.val_loss={}
             


        self.compute_all_loss=compute_all_loss

        if self.loss_weights['pe']!=0 or self.compute_all_loss:
            if 'pe_loss' not in self.train_loss or not append_loss:
                self.train_loss['pe_loss']=[]
            if valid_dataset and ('pe_loss' not in self.val_loss or not append_loss):
                self.val_loss['pe_loss']=[]
            
        if self.loss_weights['force']!=0 or self.compute_all_loss: # when force is used for training/validation OR compute force_loss is requested     
            if 'force_loss' not in self.train_loss or not append_loss:
                self.train_loss['force_loss']=[]
            if valid_dataset and ('force_loss' not in self.val_loss or not append_loss):
                self.val_loss['force_loss']=[]
                
        if self.loss_weights['force']!=0:
            print ("Forces are used for training.",flush=True)
        else:
            print ("Forces are not used for training.",flush=True)
            
        if 'stress' in self.loss_weights and self.loss_weights['stress']!=0 or self.compute_all_loss: # when stress is used for training/validation OR compute stress_loss is requested
            if 'stress_loss' not in self.train_loss or not append_loss:
                self.train_loss['stress_loss']=[]
            if valid_dataset and ('stress_loss' not in self.val_loss or not append_loss):
                self.val_loss['stress_loss']=[]
            print ("Stresses are used for training.",flush=True)
        else:
            print ("Stresses are not used for training.",flush=True)

        if ('force' in self.loss_weights and self.loss_weights['force']!=0) or \
            ('stress' in self.loss_weights and self.loss_weights['stress']!=0) or self.compute_all_loss: 
            if 'total_loss' not in self.train_loss or not append_loss:
                self.train_loss['total_loss']=[]
            if valid_dataset and ('total_loss' not in self.val_loss or not append_loss):
                self.val_loss['total_loss']=[]
        
        # normalize or starndardize input data    
        if self.scaling:
            self.compute_scaling_factors(train_dataset)
            print('Scaling factors are computed using training dataset.',flush=True)
            train_dataset = self.scaling_dataset(train_dataset)
            print('Training dataset are %s.' % ('standardized' if self.scaling =='std' else 'normalized'),flush=True)
            if valid_dataset:
                valid_dataset = self.scaling_dataset(valid_dataset)
                print('Validation dataset are %s.' % ('standardized' if self.scaling =='std' else 'normalized'),flush=True)


        if cache is True:
            train_dataset = train_dataset.cache()
            if valid_dataset:
                valid_dataset = valid_dataset.cache()
                
                
        if shuffle:
            if buffer_size is None:
                buffer_size = train_dataset.cardinality().numpy()

            # reshuffle train_dataset each epoch during trainin    
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)
            print('Training dataset will be shuffled during training.',flush=True)


        n_train_examples = len(train_dataset)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        n_structs_per_train_batch = batch_size*np.ones((len(train_dataset)))
        if n_train_examples%batch_size!=0:
            n_structs_per_train_batch[-1] = n_train_examples%batch_size

        # train dataset is mandatory, valid dataset is optional
        if valid_dataset:
            n_valid_examples = len(valid_dataset)
            valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            n_structs_per_valid_batch = batch_size*np.ones((len(valid_dataset)))
            if n_valid_examples%batch_size!=0:
                n_structs_per_valid_batch[-1] = n_valid_examples%batch_size


        train_start_time = time.time()
        early_stop_repeats = 0
        self.first_epoch_passed = False
        
        # start training
        for epoch in range(epochs):
            epoch_start_time = time.time()
            # iterate over the batches of the training dataset
            for step, [(input_dict, output_dict), n_structs_train] in enumerate(zip(train_dataset, n_structs_per_train_batch)):
                batch_loss = self.train_step(input_dict, output_dict)
                if step==0:
                    train_epoch_loss = batch_loss
                else:
                    for key in batch_loss:
                        train_epoch_loss[key] += batch_loss[key] * n_structs_train

            for key in batch_loss:
                train_epoch_loss[key] = train_epoch_loss[key]/n_train_examples
                self.train_loss[key].append(train_epoch_loss[key])


            if hasattr(self, 'softadapt_params'):
                # if epoch==0:
                #     print('epocchhhhhhhh=0, self.first_epoch_passed:', self.first_epoch_passed, flush=True)
                self.update_loss_weights(epoch, train_epoch_loss)
                # for key in self.loss_weights:    
                #     self.loss_weights_history[key].append(self.loss_weights[key])


            # Iterate over the batches of the validation dataset
            if valid_dataset is not None:  
                for step, [(input_dict, output_dict), n_structs_valid] in enumerate(zip(valid_dataset, n_structs_per_valid_batch)):
                    batch_loss = self.validation_step(input_dict, output_dict)
                    if step==0:
                        val_epoch_loss = batch_loss
                    else:
                        for key in batch_loss:
                            val_epoch_loss[key] += batch_loss[key] * n_structs_valid

                for key in batch_loss:
                    val_epoch_loss[key] = val_epoch_loss[key]/n_valid_examples
                    self.val_loss[key].append(val_epoch_loss[key])
                    

            epoch_end_time = time.time()
            time_per_epoch = (epoch_end_time - epoch_start_time)
            
            if (epoch+1)%output_freq == 0:
                print('\n===> Epoch %i/%i - %.3fs/epoch' % (epoch+1, epochs, time_per_epoch),flush=True)
                print('     training_loss   ',*["- %s: %5.3e" % (key,value) for key,value in train_epoch_loss.items()],flush=True)
            if valid_dataset:
                if (epoch+1)%output_freq == 0:
                    print('     validation_loss ',*["- %s: %5.3e" % (key,value) for key,value in val_epoch_loss.items()],flush=True)

            if early_stop:
                if 'train_loss' in early_stop:
                    if 'total_loss' in train_epoch_loss:
                        if train_epoch_loss['total_loss']<=early_stop['train_loss'][0]:
                            early_stop_repeats += 1
                        else:
                            early_stop_repeats = 0
                    elif 'pe_loss' in train_epoch_loss:
                        if train_epoch_loss['pe_loss']<=early_stop['train_loss'][0]:
                            early_stop_repeats += 1
                        else:
                            early_stop_repeats = 0
                    if early_stop_repeats == early_stop['train_loss'][1]:
                        print('\nTraining is stopped when train_loss <= %.3e for %i time.'%(early_stop['train_loss'][0],early_stop['train_loss'][1]),flush=True)
                        break
                elif 'val_loss' in early_stop:
                    if 'total_loss' in val_epoch_loss:
                        if val_epoch_loss['total_loss']<=early_stop['val_loss'][0]:
                            early_stop_repeats += 1
                        else:
                            early_stop_repeats = 0
                    elif 'pe_loss' in val_epoch_loss:
                        if val_epoch_loss['pe_loss']<=early_stop['val_loss'][0]:
                            early_stop_repeats += 1
                        else:
                            early_stop_repeats = 0
                    if early_stop_repeats == early_stop['val_loss'][1]:
                        print('\nTraining is stopped when val_loss <= %.3e for %i time.'%(early_stop['val_loss'][0],early_stop['val_loss'][1]),flush=True)
                        break

            if nepochs_checkpoint:
                if (epoch+1) % nepochs_checkpoint ==0:
                    if savefolder:
                        self.save(savefolder, save_training_history=True, checkpoint=True)
                    else:
                        self.save(save_training_history=True, checkpoint=True)
                    self.plot_loss(saveplot=True, figfolder=figfolder, showplot=False)

            if decay:
                if (epoch+1) % decay['decay_steps']==0 and (epoch+1)<epochs:
                    if self.lr > decay['min_lr']:
                        self.decay_lr(epoch)
                        print('\nLearning rate is decreased to %.3e'% self.lr,flush=True)

                    self.lr_history.append(self.lr)

        elapsed_time = (epoch_end_time - train_start_time)
        print('\nEnd of training, elapsed time: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),flush=True)




    def save(self, model_dir=None, save_training_history=False, checkpoint=False):
        """
        Save a trained model.
        
        Args:
            model_dir: directory for the saved neural network model
        """
        print('Saving network variables ...',flush=True)
                
        self.saved_descriptor = {}
        for key, value in self.descriptor.items():
            self.saved_descriptor[key] = tf.Variable(value)
        print('  symmetry function parameters',flush=True)

        self.saved_num_fingerprints = tf.Variable(self.num_fingerprints)
        print('  number of fingerprints')

        self.saved_elements = tf.Variable(self.elements)
        print('  elements',flush=True)
        
        self.saved_arch = tf.Variable(self.arch)
        print('  network architecture',flush=True)
        self.saved_activation_function = tf.Variable(self.activation_function)
        print('  activation function',flush=True)

        self.saved_data_type = tf.Variable(self.data_type)
        print('  data type',flush=True)

        self.saved_optimizer = tf.Variable(self.optimizer)
        print('  optimizer',flush=True)
        self.saved_loss_fun = tf.Variable(self.loss_fun)

        self.saved_loss_weights = {}
        for key in self.loss_weights:
            self.saved_loss_weights[key] = tf.Variable(self.loss_weights[key], dtype=self.data_type)
        print('  loss function',flush=True)

        if hasattr(self,'softadapt_params'):
            self.saved_softadapt_params = {}
            for key, value in self.softadapt_params.items():
                self.saved_softadapt_params[key] = tf.Variable(value)
            print('  softadapt_params', flush=True)

        if hasattr(self,'decay'):
            self.saved_decay = {}
            for key, value in self.decay.items():
                self.saved_decay[key] = tf.Variable(value)
            print('  decay learning parameters', flush=True)
                
        self.saved_lr = tf.Variable(self.lr)
        print('  learning rate at the last epoch', flush=True)
      
        self.save_training_history = tf.Variable(save_training_history)
        
        if save_training_history:
            current_epochs = str(len(self.train_loss['pe_loss']))
            if checkpoint:
                if model_dir:
                    model_dir = os.path.join(model_dir, 'saved_model_at_epoch_'+current_epochs+'.dnn')
                else:
                    model_dir = 'saved_model_at_epoch_'+current_epochs+'.dnn'

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            self.saved_loss_weights_history = {}
            for key in self.loss_weights_history:
                # print('  loss_weights_history[',key,']:', self.loss_weights_history[key])
                self.saved_loss_weights_history[key] = tf.Variable(self.loss_weights_history[key], dtype=self.data_type)
            print('  loss weights history has been saved.')

            if hasattr(self,'decay'):
                if hasattr(self, 'val_loss'):
                    history = [self.lr_history,self.train_loss,self.val_loss]
                else:
                    history = [self.lr_history,self.train_loss]
                msg = 'Decayed learning rates, training and validation loss are saved as binary data in %s'%(model_dir+'/training_history_data')
            else:
                if hasattr(self, 'val_loss'):
                    history = [self.train_loss,self.val_loss]
                else:
                    history = [self.train_loss]
                msg = 'Training and validation loss are saved as binary data in %s'%(model_dir+'/training_history_data')
            with open(model_dir+'/training_history_data', 'wb') as out_: 
                pickle.dump(history, out_)
                print(msg,flush=True)

        tf.saved_model.save(self, model_dir)
           
        input_tag, input_name, output_tag, output_name = get_signature(model_dir)
        file = open(model_dir+'/parameters','w')

        file.write('%-20s ' % 'element')
        file.write('  '.join(j for j in self.elements))
        file.write('\n\n')

        file.write('%-20s %d\n' % ('input',len(input_tag)))
        for i in range(len(input_tag)):
            file.write('%-20s %-20s\n' % (input_tag[i],input_name[i]))
        file.write('\n')

        file.write('%-20s %d\n' % ('output',len(output_tag)))
        for i in range(len(output_tag)):
            file.write('%-20s %-20s\n' % (output_tag[i],output_name[i]))
        file.write('\n')

        file.write('%-20s %f\n' % ('cutoff',self.descriptor['cutoff']))
        file.write('\n')

        file.write('%-20s %s  %d\n' % ('descriptor',self.descriptor['name'], len(self.descriptor)-2))    
        for i in range(len(self.descriptor)):
            key  = list(self.descriptor.keys())[i]
            if key=='etaG2' or key=='etaG4' or key=='zeta' or key=='lambda':
                file.write('%-20s '% key)
                file.write('  '.join(str(j) for j in self.descriptor[key]))
                file.write('\n')
        file.close()
        print('Network signatures and descriptor are written to %s for LAMMPS simulation'% (model_dir+'/parameters'),flush=True)
 

    def do_print_shapes(self):
        '''Print shapes of trainable parameters - Just for testing purposes'''
        nlayer = 1
        count = 0
        nparams_prev_w = 0
        for param in self.params:
            # print('param:',param.shape)
            if nlayer > (len(self.arch)+1)*2:
                print('loss_weights:', param.numpy())
            else:
                if count%2==0:
                    print('W'+str(nlayer), param.shape, end=' ')
                    nparams_prev_w = tf.shape(param)[0]*tf.shape(param)[1]
                else:
                    print('b'+str(nlayer), param.shape, end=' ')
                    # print('tf.shape(param):', tf.shape(param))
                    print('len(tf.shape(param)):',len(tf.shape(param)))
                    if len(tf.shape(param))==2:
                        print('\t\t# params:', nparams_prev_w.numpy() + tf.shape(param)[0].numpy()*tf.shape(param)[1].numpy())
                    else:
                        print('\t\t# params:', nparams_prev_w.numpy() + tf.shape(param)[0].numpy())
                    nlayer+=1


            count+=1

        print('\n')
        

    def plot_loss(self,start_epoch=1,saveplot=False,showplot=True,**kwargs):
        """
        Plot the losses.
        
        Args:
            start_epoch: plotting starts from start_epoch 
            figsize: set the fingersize, e.g. (8,4)
            saveplot(bool): if true, save the plots to "plot_loss" folder  
            kwargs: optional parameters for figures, default values are:
                    figfolder = './loss_figures', the folder name for saving figures 
                    figsize = (8,5)
                    linewidth = [1,1] for train and validation loss plot
                    color =['blue','darkgreen'] 
                    label = ['train loss','validation loss']
                    linestyle = ['-','-']
                    markersize = [5,5] 
                    xlabel = 'epoch'
                    ylabel = {'pe':'loss(eV)', 'force':'loss(eV/A),'stress':'loss(GPa)'}         
                    format = 'pdf'
        """

        if 'figfolder' in kwargs.keys():
            figfolder = kwargs['figfolder']
        else:
            figfolder = './loss_figures'
        if 'figsize' in kwargs.keys():
            figsize = kwargs['figsize']
        else:
            figsize = (15,10)
        if 'linewidth' in kwargs.keys():
            linewidth = kwargs['linewidth']
        else:
            linewidth = [1,1]
        if 'color' in kwargs.keys():
            color = kwargs['color']
        else:
            color = ['blue','darkgreen']
        if 'label' in kwargs.keys():
            label = kwargs['label']
        else:
            label =['train loss','validation loss']
        if 'linestyle' in kwargs.keys():
            linestyle = kwargs['linestyle']
        else:
            linestyle = ['-','-']
        if 'markersize' in kwargs.keys():
            markersize = kwargs['markersize']
        else:
            markersize = [5,5]
        if 'xlabel' in kwargs.keys():
            xlabel = kwargs['xlabel']
        else:
            xlabel = 'epoch'
        if 'ylabel' in kwargs.keys():
            ylabel = kwargs['ylabel']
        else:
            ylabel = {'pe_loss':'pe_per_atom loss', 'force_loss':'force loss','stress_loss':'stress loss','total_loss':'total loss'}
        if 'format' in kwargs.keys():
            format = kwargs['format']
        else:
            format = 'pdf'
        
        
        matplotlib.rc('legend', fontsize=15) 
        matplotlib.rc('xtick', labelsize=15) 
        matplotlib.rc('ytick', labelsize=15) 
        matplotlib.rc('axes', labelsize=15) 
        matplotlib.rc('axes', titlesize=15)


        fig, axs = plt.subplots(2, 2 ,figsize=figsize)
        fig.tight_layout(pad=5.0)

        i=0
        j=0
        for key in self.train_loss:
            end_epoch = len(self.train_loss[key])
            epoch = np.arange(start_epoch,end_epoch+1)
            axs[i][j].plot(epoch,self.train_loss[key][start_epoch-1:],linestyle[0], markersize=markersize[0], \
                     fillstyle='none', linewidth=linewidth[0], color= color[0], label=label[0])
                
            if hasattr(self,'val_loss'):
                axs[i][j].plot(epoch,self.val_loss[key][start_epoch-1:],linestyle[1], markersize=markersize[1], \
                         fillstyle='none', linewidth=linewidth[1], color=color[1], label=label[1])
            axs[i][j].set_xlabel(xlabel)
            axs[i][j].set_ylabel(ylabel[key])
            axs[i][j].set_yscale('log')
            axs[i][j].legend(loc='upper right',frameon=True,borderaxespad=1)
            #axs[i][j].title.set_text(key)
            axs[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[i][j].grid(True)
            j = j+1
            if j>1:
                i=1
                j=0
                
        end_epoch = len(self.train_loss['pe_loss'])
        if not os.path.isdir(figfolder):
            os.mkdir(figfolder)
        figname = 'loss_at_epoch_'+str(end_epoch+1)+'.'+format
        if saveplot:
            fig.savefig(os.path.join(figfolder,figname), bbox_inches='tight', format=format, dpi=500)
            plt.close(fig)
        if showplot:
            plt.show()




        

            
                                    
def get_signature(model_dir):
    """
    Run shell command 'saved_model_cli show --dir  model_dir  --tag_set serve --signature_def serving_default', 
    and get the signature of the network from the shell outputs.
    SavedModel Command Line Interface (CLI) is a Tensorflow tool to inspect a SavedModel.

    Args:
        model_dir: directory for a saved neural network model
    """
    stream = os.popen('saved_model_cli show --dir '+ model_dir +' --tag_set serve --signature_def serving_default')
    output = stream.read()
    lines = output.split('\n')
    input_tag=[]
    output_tag=[]
    input_name=[]
    output_name=[]
    check_in = 0
    check_out = 0
    for line in lines[0:-2]:
        if 'inputs' in line:
            input_tag.append(line.split('[')[1].split(']')[0].split("'")[1])
            check_in = 1
            check_out = 0
        if 'outputs' in line:
            output_tag.append(line.split('[')[1].split(']')[0].split("'")[1])
            check_out = 1
            check_in = 0
        if 'name' in line:
            if check_in:
                input_name.append(line.split()[1])
            if check_out:
                output_name.append(line.split()[1])
    return input_tag, input_name, output_tag, output_name


def print_signature(model_dir):
    """
    Print the neural network signature.

    Args:
        model_dir: directory for a saved neural network model
    """
    stream = os.popen('saved_model_cli show --dir '+ model_dir +' --tag_set serve --signature_def serving_default')
    output = stream.read()
    print(output)

    

    
# def save(obj, model_dir):
#     """
#     Save a trained model.

#     Args:
#         model_dir: directory for the saved neural network model
#         descriptor(dictionary): descriptor parameters, used for LAMMPS prediction
#     """
#     tf.saved_model.save(obj, model_dir)

#     input_tag, input_name, output_tag, output_name = get_signature(model_dir)
#     file = open(model_dir+'/parameters','w')
    
#     file.write('%-20s ' % 'element')
#     file.write('  '.join(j for j in obj.elements))
#     file.write('\n\n')

#     file.write('%-20s %d\n' % ('input',len(input_tag)))
#     for i in range(len(input_tag)):
#         file.write('%-20s %-20s\n' % (input_tag[i],input_name[i]))
#     file.write('\n')
        
#     file.write('%-20s %d\n' % ('output',len(output_tag)))
#     for i in range(len(output_tag)):
#         file.write('%-20s %-20s\n' % (output_tag[i],output_name[i]))
#     file.write('\n')

#     file.write('%-20s %f\n' % ('cutoff',obj.descriptor['cutoff']))
#     file.write('\n')

#     file.write('%-20s %s  %d\n' % ('descriptor',obj.descriptor['name'], len(obj.descriptor)-2))    
#     for i in range(2,len(obj.descriptor)):
#         key  = list(obj.descriptor.keys())[i]
#         file.write('%-20s '% key)
#         file.write('  '.join(str(j) for j in obj.descriptor[key]))
#         file.write('\n')
        
#     file.close()
# #    print('Network signatures and descriptor are written to %s for LAMMPS simulation.'% (model_dir+'/parameters'))
    

        
def load(model_dir):
    """
    Load a saved model.

    Args:
        model_dir: directory of a saved neural network model
    """
    if model_dir:
        return Network(import_dir=model_dir)
    else:
        raise ValueError('Load function has no model directory.')


def print_output(output_dict):
    num_images = output_dict['pe'].shape[0]
    for i in range(num_images):
        print ('-----------------------------------------------------------------------------')
        print ('image %i: potential energy = %.3e'%(i+1,output_dict['pe'][i]))
        stress_str = ('stress_xx','stress_yy','stress_zz','stress_yz','stress_xz','stress_xy')
        stress_value = (output_dict['stress'][i][0],output_dict['stress'][i][4],output_dict['stress'][i][8],\
                        output_dict['stress'][i][5],output_dict['stress'][i][2],output_dict['stress'][i][1])
        msg_str = '%-12s %-12s %-12s %-12s %-12s %-12s' % stress_str
        msg_value ='%-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e' % stress_value
        print(msg_str)
        print(msg_value)
        if 'atom_pe' in output_dict.keys():
            force_str = ('atom_id', 'fx','fy','fz', 'atom_pe')
            msg_str = '%-12s %-12s %-12s %-12s %-12s' % force_str
        else:
            force_str = ('atom_id', 'fx','fy','fz')
            msg_str = '%-12s %-12s %-12s %-12s' % force_str
        print(msg_str)
        num_atoms = output_dict['force'].shape[1]
        for j in range(num_atoms):
            if 'atom_pe' in output_dict.keys():
                force_value = (j+1, output_dict['force'][i][j][0], output_dict['force'][i][j][1], output_dict['force'][i][j][2], output_dict['atom_pe'][i][j])
                msg_value = '%-12d %-12.4e %-12.4e %-12.4e %-12.4e' % force_value
            else:
                force_value = (j+1, output_dict['force'][i][j][0], output_dict['force'][i][j][1], output_dict['force'][i][j][2])
                msg_value = '%-12d %-12.4e %-12.4e %-12.4e' % force_value
            print(msg_value)



def save_output(output_dict,filename):
    f = open(filename, 'w')
    num_images = output_dict['pe'].shape[0]
    for i in range(num_images):
        f.write('-----------------------------------------------------------------------------\n')
        f.write('image %i: potential energy = %.3e\n'%(i+1,output_dict['pe'][i]))
        stress_str = ('stress_xx','stress_yy','stress_zz','stress_xy','stress_yz','stress_xz')
        stress_value = (output_dict['stress'][i][0],output_dict['stress'][i][4],output_dict['stress'][i][8],\
                        output_dict['stress'][i][1],output_dict['stress'][i][3],output_dict['stress'][i][2])
        msg_str = '%-12s %-12s %-12s %-12s %-12s %-12s\n' % stress_str
        msg_value ='%-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n' % stress_value
        f.write(msg_str)
        f.write(msg_value)
        if 'atom_pe' in output_dict.keys():
            force_str = ('atom_id', 'fx','fy','fz', 'atom_pe')
            msg_str = '%-12s %-12s %-12s %-12s %-12s\n' % force_str
        else:
            force_str = ('atom_id', 'fx','fy','fz')
            msg_str = '%-12s %-12s %-12s %-12s\n' % force_str
        f.write(msg_str)
        num_atoms = output_dict['force'].shape[1]
        for j in range(num_atoms):
            if 'atom_pe' in output_dict.keys():
                force_value = (j+1, output_dict['force'][i][j][0], output_dict['force'][i][j][1], output_dict['force'][i][j][2], output_dict['atom_pe'][i][j])
                msg_value = '%-12d %-12.4e %-12.4e %-12.4e %-12.4e\n' % force_value
            else:
                force_value = (j+1, output_dict['force'][i][j][0], output_dict['force'][i][j][1], output_dict['force'][i][j][2])
                msg_value = '%-12d %-12.4e %-12.4e %-12.4e\n' % force_value
            f.write(msg_value)
    f.close()


    


