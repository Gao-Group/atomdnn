import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import os
import atomdnn
from atomdnn.data import *
import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt

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
    The nueral network is built based on tf.Module.

    Args:
            elements(python list): list of element in the system, e.g. [C,O,H]
            num_fingerprints(int): number of fingerprints in dataset
            arch: network architecture, e.g. [10,10,10], 3 dense layers each with 10 neurons
            activation_function: any avialiable Tensorflow activation function, e.g. 'relu'
            weighs_initializer: the way to initialize weights, default one is tf.random.normal
            bias_initializer: the way to initialize bias, default one is tf.zeros
            import_dir: the directory of a saved model to be loaded
    """
    def __init__(self, elements=None, num_fingerprints=None, arch=None, \
                 activation_function=None, weights_initializer=tf.random.normal, bias_initializer=tf.zeros, import_dir=None):
        
        super(Network,self).__init__()
        
        if import_dir:
            imported = tf.saved_model.load(import_dir)
            self.inflate_from_file(imported)
        else:
            if arch:
                self.arch = arch
            else:
                self.arch = [10]
                print ('network arch is set to [10] by default.')
                
            if activation_function:
                self.activation_function = activation_function
            else:
                self.activation_function = 'tanh'
                print ('activation function is set to tanh by default.')
            self.tf_activation_function = tf.keras.activations.get(self.activation_function)

            if num_fingerprints:
                self.num_fingerprints = num_fingerprints 
            else:
                raise ValueError ('Network has no num_fingerprints input.')
                            
            if elements is not None:
                self.elements = elements 
            else:
                raise ValueError ('Network has no elements input.')

            self.data_type = atomdnn.data_type
            self.weights_initializer = weights_initializer
            self.bias_initializer = bias_initializer
                    
            self.built = False
            self.params = [] # layer and node parameters
        
            # training parameter
            self.loss_fun=None # string
            self.optimizer=None # string
            self.tf_optimizer=None
            self.lr=None

                
        self.saved_num_fingerprints = tf.Variable(self.num_fingerprints)
        self.saved_arch = tf.Variable(self.arch)
        self.saved_activation_function = tf.Variable(self.activation_function)
        self.saved_data_type = tf.Variable(self.data_type)
        self.saved_elements = tf.Variable(self.elements)

        
        
    def inflate_from_file(self, imported):
        '''
        Inflate network object from a SavedModel.

        Args:
            imported: a saved tensorflow neural network model
        '''
        self.built = True
        self.arch = imported.saved_arch.numpy()
        self.num_fingerprints = imported.saved_num_fingerprints.numpy()

        if hasattr(imported,'scaling'):
            self.scaling = imported.scaling
            self.scaling_factor = imported.scaling_factor
        else:
            self.scaling = None
            
        self.activation_function = imported.saved_activation_function.numpy().decode()
        self.tf_activation_function = tf.keras.activations.get(self.activation_function)
        self.data_type = imported.saved_data_type.numpy().decode()
        self.elements = [imported.saved_elements.numpy()[i].decode() for i in range(imported.saved_elements.shape[0])]
           
        self.lr = imported.saved_lr.numpy()
        self.loss_fun = imported.saved_loss_fun.numpy().decode()

        if hasattr(imported, 'dropout'):
            self.dropout = imported.dropout
            print('Dropout rate has been set to ', self.dropout.numpy())
        else:
            self.dropout = tf.Variable(0., dtype=self.data_type)
        
        self.optimizer = imported.saved_optimizer.numpy().decode()
        self.tf_optimizer = tf.keras.optimizers.get(self.optimizer)
        K.set_value(self.tf_optimizer.learning_rate, self.lr)

        self.train_loss = imported.train_loss
        self.val_loss = imported.val_loss
        self.loss_weights = imported.loss_weights
        
        try:
            self.params = []
            for param in imported.params:
                self.params.append(param)
        except AttributeError:
            raise AttributeError('imported_object does not have params attribute.')
 
        print('Network has been inflated! self.built:',self.built)
        
        
        
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

        self.built = True

            
    
    def compute_pe (self, fingerprints, atom_type, verbose=False):
        """
        Forward pass of the network to compute potential energy. Parallel networks are used for multiple atom types.
        
        Args:
            fingerprints: 3D array *[batch_size,atom_num,fingerprints_num]*
            atom_type: 2D array *[batch_size, atom_num]*

        Returns:
            total potential energy and per-atom potential energy
        """
        nimages = tf.shape(atom_type)[0]
        natoms = tf.shape(atom_type)[1]
        
        atom_pe = tf.zeros([nimages,natoms,1],dtype=self.data_type)
        params_iter = 0
        nlayers = len(self.arch) + 1 # include one linear layer at the end
        nelements = len(self.elements)
            
        if verbose:
            print('nlayers:', nlayers, '    # inlcuding linear layer')

        for i in range(nelements):
            type_id = i + 1
            type_array = tf.ones([nimages, natoms], dtype='int32')*type_id
            type_mask = tf.cast(tf.math.equal(atom_type, type_array), dtype=self.data_type)
            type_onehot = tf.linalg.diag(type_mask)
            fingerprints_i = tf.matmul(type_onehot, fingerprints)

                            
            # first dense layer
            params_iter = nlayers*2*i

            # if dropout, it happens before first dense layer
            dropped = tf.nn.dropout(fingerprints_i, rate=self.dropout)
                
            # sequential dense layer
            Z = tf.matmul(dropped, self.params[params_iter]) + self.params[params_iter+1]
            Z = self.tf_activation_function(Z)
                        
            # sequential dense layer
            for j in range(1,nlayers-1):    
                Z = tf.matmul(Z,self.params[params_iter+j*2]) + self.params[params_iter+j*2+1]
                Z = self.tf_activation_function(Z)                
                
            Z = tf.matmul(Z, self.params[nlayers*2*(i+1)-2]) + self.params[nlayers*2*(i+1)-1]

            # apply the mask
            mask = tf.reshape(type_mask,[nimages,natoms,1])
                
            atom_pe += Z * mask
            
        total_pe = tf.reshape(tf.math.reduce_sum(atom_pe, axis=1),[nimages])        
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
        if hasattr(self,'scaling'):
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

        # Form dEdG matrix corresponding to the blocks in dGdr data file
        center_atom_id_reshape = tf.reshape(center_atom_id, shape=[num_image,-1,1])
        dEdG_block = tf.gather_nd(dEdG, center_atom_id_reshape, batch_dims=1)     
        dEdG_block = tf.reshape(dEdG_block, [num_image, max_block, 1, self.num_fingerprints])

        
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


    
    def predict(self, input_dict, compute_force=atomdnn.compute_force, training=False, verbose=False, evaluation=False, debug=False):
        """
        Predict energy, force and stress.

        Args:
            input_dict: input dictionary data
            compute_force(bool): True to compute force and stress
            training: True when used during training

        Returns:
            dictionary: potential energy, force and stress
        """
        if not self.built:
            self._build()
            self.built = True

        predict_start_time = time.time()
        
        fingerprints = input_dict['fingerprints']
        num_atoms = np.count_nonzero(input_dict['atom_type'], axis=1)
        if not tf.is_tensor(fingerprints):
            fingerprints = tf.convert_to_tensor(fingerprints)

        if not training:
            if hasattr(self, 'scaling'):
                if self.scaling == 'std':  # standardize with the mean and deviation
                    fingerprints = (fingerprints - self.scaling_factor[0])/self.scaling_factor[1]
                elif self.scaling == 'norm': # normalize with the minimum and maximum
                    fingerprints = (fingerprints - self.scaling_factor[0])/(self.scaling_factor[1] - self.scaling_factor[0])
                
        if not compute_force:
            # print('Number of atoms computed by my code:', num_atoms)
            total_pe, atom_pe = self.compute_pe(fingerprints, input_dict['atom_type'], verbose=verbose)

            if evaluation:
                predict_stop_time = (time.time() - predict_start_time)*1000
                print('\n    End of prediction, elapsed time: ', time.strftime('%H:%M:%S:{}'.format(predict_stop_time%1000), time.gmtime(predict_stop_time/1000.0)))
            
            if training:
                return {'pe':total_pe, 'pe/atom':total_pe/num_atoms} 
            else:
                return {'pe':total_pe.numpy(), 'pe/atom':total_pe/num_atoms}

        else: 
            with tf.GradientTape() as dEdG_tape:
                dEdG_tape.watch(fingerprints)
                total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'], verbose=verbose)
            dEdG = dEdG_tape.gradient(atom_pe, fingerprints)
                
            force, stress = self.compute_force_stress(dEdG, input_dict)

            if evaluation:
                predict_stop_time = (time.time() - predict_start_time)*1000
                print('\n    End of prediction, elapsed time: ', time.strftime('%H:%M:%S:{}'.format(predict_stop_time%1000), time.gmtime(predict_stop_time/1000.0)))

            if training:
                if debug:
                    print(f'DEBUG1:\ntotal_pe:{total_pe} - num_atoms:{num_atoms}')
                return {'pe':total_pe, 'pe/atom':total_pe/num_atoms, 'force':force, 'stress':stress}
            else:
                return {'pe':total_pe.numpy(), 'pe/atom':total_pe.numpy()/num_atoms, 'force':force.numpy(), 'stress':stress.numpy()}

        
                     
    
    def _predict(self, input_dict,compute_force=atomdnn.compute_force):
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

        

    def evaluate(self,dataset,batch_size=None,return_prediction=True,compute_force=atomdnn.compute_force):
        """
        Do evaluation on trained model.
        
        Args:
            dataset: tensorflow dataset used for evaluation
            batch_size: default is total data size
            return_prediction(bool): True for returning prediction resutls
            compute_force(bool): True for computing force and stress
        """

        if batch_size is None:
            batch_size = dataset.cardinality().numpy()
            
        for step, (input_dict, output_dict) in enumerate(dataset.batch(batch_size)):
            y_predict = self.predict(input_dict, compute_force=compute_force)
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
            # try to add loss values to the return dictionary ggg
            y_predict[key] = eval_loss[key]
            print('%15s:  %15.4e' % (key, eval_loss[key]))
        if 'total_loss' in eval_loss:
            print('The total loss is computed using the loss weights', *['- %s: %.2f' % (key,value) for key, value in self.loss_weights.items()])

        if return_prediction:
            print('\nThe prediction is returned.')
            return y_predict
                                                

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
            return tf.sqrt(tf.maximum(tf_loss_fun(true,pred), 1e-9))
        else:
            tf_loss_fun = tf.keras.losses.get(self.loss_fun)
            return tf_loss_fun(true, pred)
            
        
        
    def loss(self, true_dict, pred_dict, training=False, validation=False, train_by_pe=False, verbose=False, debug=False):
        """
        Compute losses for energy, force and stress.
        
        Args:
            true_dict(dictionary): dictionary of outputs data
            pred_dict(dictionary): dictionary of predicted results
            training(bool): True for loss calculation during training
            validation(bool): True for loss calculation during validation  
            train_by_pe(boolean): False by default. If true, then loss function for PE is calculated using pe/atom; otherwise, it is calculated using total PE. 
        Returns:
            if training is true, return total_loss and loss_dict(loss dictionary), otherwise(for evaluation) return loss_dict
        """
        if debug:
            temp1 = true_dict['pe/atom']
            temp2 = pred_dict['pe/atom']
            print(f'DEBUG2:\ntrue_dict:{temp1} - pred_dict:{temp2}')

        loss_dict = {}
        if train_by_pe:
            loss_dict['pe_loss'] = self.loss_function(true_dict['pe'], pred_dict['pe'])
            if verbose:
                tmp1 = true_dict['pe']
                tmp2 = pred_dict['pe']
                print(f'true_dict[\'pe\']:{tmp1}\npred_dict[\'pe\']:{tmp2}')
        else:
            loss_dict['pe_atom_loss'] = self.loss_function(true_dict['pe/atom'], pred_dict['pe/atom'])
            if verbose:
                tmp1 = true_dict['pe/atom']
                tmp2 = pred_dict['pe/atom']
                print(f'true_dict[\'pe/atom\']:{tmp1}\npred_dict[\'pe/atom\']:{tmp2}')

                if 'force' in true_dict:
                    tmp3 = true_dict['force']
                    tmp4 = pred_dict['force']
                    print(f'true_dict[\'force\']:{tmp3}\npred_dict[\'force\']:{tmp4}')

        
        if 'force' in pred_dict and 'force' in true_dict: # compute force loss
            # when evaluation OR train/validation using force OR all losses are requested
            if ((not training and not validation) or ((training or validation) and self.loss_weights['force']!=0)) or self.compute_all_loss:
                # loss_value = tf.reduce_mean(self.loss_function(true_dict['force'], pred_dict['force']))
                if verbose:
                    print('force_loss->tf.reduce_mean(self.loss_function(true_force, pred_force))')
                loss_value = tf.reduce_mean(self.loss_function(true_dict['force'], pred_dict['force']))
                loss_dict['force_loss'] = loss_value
                
                
        if 'stress' in pred_dict and 'stress' in true_dict: # compute stress loss
            # when evaluation OR train/validation using stress OR all losses are requested
            if ((not training and not validation) or ((training or validation) and (self.loss_weights['stress']!=0))) or self.compute_all_loss: 
                indices = [[[0],[4],[8],[1],[2],[5]]] * len(pred_dict['stress']) # select upper triangle elements (pxx,pyy,pzz,pxy,pxz,pyz) of the stress tensor
                stress = tf.gather_nd(pred_dict['stress'], indices=indices, batch_dims=1)
                loss_dict['stress_loss'] = tf.reduce_mean(self.loss_function(true_dict['stress'],stress))


        total_loss = None
        if self.loss_weights['force']==0 and self.loss_weights['stress']==0: # only pe used for training
            if train_by_pe:
                total_loss = loss_dict['pe_loss']
            else:
                total_loss = loss_dict['pe_atom_loss']
        elif 'stress' in self.loss_weights and\
            self.loss_weights['force']!=0 and self.loss_weights['stress']!=0: # pe, force and stress used for training
            if train_by_pe:
                total_loss = loss_dict['pe_loss'] * self.loss_weights['pe'] + loss_dict['force_loss'] * self.loss_weights['force']\
                    + loss_dict['stress_loss'] * self.loss_weights['stress']
            else:
                total_loss = loss_dict['pe_atom_loss'] * self.loss_weights['pe'] + loss_dict['force_loss'] * self.loss_weights['force']\
                    + loss_dict['stress_loss'] * self.loss_weights['stress']
        elif self.loss_weights['force']!=0: # pe and force used for training
            if train_by_pe:
                total_loss = loss_dict['pe_loss'] * self.loss_weights['pe'] + loss_dict['force_loss'] * self.loss_weights['force']
                loss_dict['total_loss'] = total_loss
            else:
                total_loss = loss_dict['pe_atom_loss'] * self.loss_weights['pe'] + loss_dict['force_loss'] * self.loss_weights['force']

        elif self.loss_weights['stress']!=0: # pe and stress used for training
            if train_by_pe:
                total_loss = loss_dict['pe_loss'] * self.loss_weights['pe'] + loss_dict['stress_loss'] * self.loss_weights['stress']
            else:
                total_loss = loss_dict['pe_atom_loss'] * self.loss_weights['pe'] + loss_dict['stress_loss'] * self.loss_weights['stress']
        else:
            raise ValueError('loss_weights is not set correctly.')

        loss_dict['total_loss'] = total_loss
        if training:
            return total_loss,loss_dict
        else:
            return loss_dict



     
    def train_step(self, input_dict, output_dict, step=None, train_by_pe=False, verbose=False, debug=False):
        """
        A single training step.

        Args:
            input_dict(Python dictionary): input dictionary data
            output_dict(Python dictionary): output dictionary data
            train_by_pe(boolean): False by default. If true, then loss function for PE is calculated using pe/atom; otherwise, it is calculated using total PE. 
        Returns:
            loss dictionary
        """
        with tf.GradientTape() as tape: # automatically watch all tensorflow variables 
            if self.loss_weights['force']==0 and self.loss_weights['stress']==0 and not self.compute_all_loss:
                pred_dict = self.predict(input_dict, training=True, compute_force=False, verbose=verbose, debug=debug)
            else:
                pred_dict = self.predict(input_dict, training=True, compute_force=True, verbose=verbose)
            total_loss,loss_dict = self.loss(output_dict, pred_dict, training=True, train_by_pe=train_by_pe, verbose=verbose, debug=debug)

        grads = tape.gradient(total_loss, self.params)

        lr = self.tf_optimizer.learning_rate
        self.tf_optimizer.apply_gradients(zip(grads, self.params))
        return loss_dict


    
    def validation_step(self, input_dict, output_dict, valid_by_pe=False):
        """
        A single validation step.
 
        Args:
            input_dict: input dictionary data
            output_dict: output dictionary data

        Returns:
            loss dictionary
        """
        if self.loss_weights['force']==0 and self.loss_weights['stress']==0 and not self.compute_all_loss:
            pred_dict = self.predict(input_dict, training=True, compute_force=False)
        else:
            pred_dict = self.predict(input_dict, training=True, compute_force=True)
        loss_dict = self.loss(output_dict, pred_dict, validation=True, train_by_pe=valid_by_pe)
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


    def update_lr(self, epoch):
        lr = self.step_decay(epoch)
        K.set_value(self.tf_optimizer.learning_rate, lr)
        self.lr_history.append(lr)
        

    def step_decay(self, epoch):
        import math
        return self.starting_lr * math.pow(self.decay_power,  np.floor((1+epoch)/self.decay_steps))
    
    
    def train(self, train_dataset, validation_dataset=None, train_by_pe=False, dropout=None, early_stop=None, scaling=None, batch_size=None, epochs=None, loss_fun=None, \
              optimizer=None, lr=None, decay_lr=False, starting_lr=None, decay_steps=None, decay_power=None, loss_weights=None, compute_all_loss=False,\
              shuffle=True, append_loss=False, descriptors=None, folder_id=None, nepochs_checkpoint=None, save_lossplots_on_checkpoint=True, verbose=False, debug=False):
        """
        Train the neural network.

        Args:
            train_dataset(tfdataset): dataset used for training
            validation_dataset(tfdataset): dataset used for validation
            train_by_pe(boolean): False by default. If true, then loss function for PE is calculated using pe/atom; otherwise, it is calculated using total PE. 
            early_stop(dictionary): condition for stop training, \
                                    e.g. {'train_loss':[0.01,3]} means stop when train loss is less than 0.01 for 3 times
            scaling(string): = None, 'std' or 'norm'
            batch_size: the training batch size
            epochs: training epochs
            loss_fun(string): loss function, 'mae', 'mse', 'rmse' and others
            opimizer(string): any Tensorflow optimizer such as 'Adam'
            lr(float): learning rate
            loss_weight(dictionary): weights assigned to loss function, e.g. {'pe':1,'force':1,'stress':0.1}  
            compute_all_loss(bool): compute loss for force and stress even when they are not used for training
            shuffle(bool): shuffle training dataset during training
            append_loss(bool): append loss history 
        """

        if not self.built:            
            self._build()
            self.built = True

        if nepochs_checkpoint:
            import os
            import shutil
            if os.path.isdir('plots'+str(folder_id)):
                shutil.rmtree('plots'+str(folder_id))

        if decay_lr:
            self.lr_history = []
            self.lr = starting_lr
            if starting_lr:
                self.starting_lr = starting_lr
            else:
                self.starting_lr = 0.01
                print(f'Decaying learning rate - starting_lr:{self.starting_lr}')
            if decay_steps:
                self.decay_steps = decay_steps
            else:
                self.decay_steps = 20
                print(f'Decaying learning rate - decay_steps:{self.decay_steps}')
            if decay_power:
                self.decay_power = decay_power
            else:
                self.decay_power = 0.5
                print(f'Decaying learning rate = decay_power:{self.decay_power}')
        else:            
            if lr:
                self.lr = lr
            elif not self.lr: 
                self.lr = 0.01
                print ("learning rate is set to 0.01 by default.")
            
        if scaling:
            if scaling != 'std' and scaling != 'norm':
                raise ValueError('Scaling needs to be \'std\' (standardization) or \'norm\'(normalization).')
            else:
                self.scaling = tf.Variable(scaling)
        else:
            self.scaling = None
            
        if optimizer:
            if isinstance(optimizer, str):
                self.optimizer = optimizer
                self.tf_optimizer = tf.keras.optimizers.get(optimizer)
            else:
                self.tf_optimizer = optimizer
                self.optimizer    = self.tf_optimizer.get_config()['name'].lower() 
            K.set_value(self.tf_optimizer.learning_rate, self.lr)
        elif not self.optimizer and not self.tf_optimizer:
            self.optimizer = 'Adam'
            self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) 
            print ("optimizer is set to Adam by default.")
            
        if loss_fun:
            self.loss_fun = loss_fun
        elif self.loss_fun is not None:
            self.loss_fun = 'rmse'
            print ("loss_fun is set to rmse by default.")
                                   
        if not epochs:
            epochs = 1
            print ("epochs is set to 1 by default.")

        if batch_size is None:
            batch_size = 30
            print ("batch_size is set to 30 by default.")    

        if dropout:
            self.dropout = tf.Variable(dropout, dtype=self.data_type)
        else:
            self.dropout = tf.Variable(0., dtype=self.data_type)
            
        if early_stop:
            if 'val_loss' in early_stop and not validation_dataset:
                raise ValueError('No validation data for early stopping.')
           
        if not hasattr(self,'train_loss') or not append_loss:
            self.train_loss={}
         
        if validation_dataset and (not hasattr(self,'val_loss') or not append_loss):
            self.val_loss={}
             
        if loss_weights is None:
            self.loss_weights = {'pe':1.,'force':0,'stress':0}
            print('loss_weights is set to default value:', self.loss_weights)
        else:
            self.loss_weights = loss_weights

        for key in self.loss_weights:
            self.loss_weights[key] = tf.Variable(self.loss_weights[key], dtype=self.data_type)

        if compute_all_loss:
            self.compute_all_loss=True
        else:
            self.compute_all_loss=False        

        
        if self.loss_weights['pe']!=0 or self.compute_all_loss:
            if train_by_pe:
                if 'pe_loss' not in self.train_loss or not append_loss:
                    self.train_loss['pe_loss'] = []
                if validation_dataset and ('pe_loss' not in self.val_loss or not append_loss):
                    self.val_loss['pe_loss'] = []
            else:
                if 'pe_atom_loss' not in self.train_loss or not append_loss:
                    self.train_loss['pe_atom_loss'] = []
                    self.train_loss['total_loss']   = []
                if validation_dataset and ('pe_atom_loss' not in self.val_loss or not append_loss):
                    self.val_loss['pe_atom_loss'] = []
            
        if self.loss_weights['force']!=0 or self.compute_all_loss: # when force is used for training/validation OR compute force_loss is requested     
            if 'force_loss' not in self.train_loss or not append_loss:
                self.train_loss['force_loss']=[]
            if validation_dataset and ('force_loss' not in self.val_loss or not append_loss):
                self.val_loss['force_loss']=[]
                
        if self.loss_weights['force']!=0:
            print ("Forces are used for training.")
        else:
            print ("Forces are not used for training.")
            
        if ('stress' in self.loss_weights and self.loss_weights['stress']!=0) or self.compute_all_loss: # when stress is used for traning/validation OR compute stress_loss is requested
            if 'stress_loss' not in self.train_loss or not append_loss:
                self.train_loss['stress_loss']=[]
            if validation_dataset and ('stress_loss' not in self.val_loss or not append_loss):
                self.val_loss['stress_loss']=[]

        if 'stress' in self.loss_weights and self.loss_weights['stress']!=0:
            print ("Stresses are used for training.")
        else:
            print ("Stresses are not used for training.")

        if self.loss_weights['force']!=0 or self.loss_weights['stress']!=0 or self.compute_all_loss: 
            if 'total_loss' not in self.train_loss or not append_loss:
                self.train_loss['total_loss']=[]
            if validation_dataset and ('total_loss' not in self.val_loss or not append_loss):
                self.val_loss['total_loss']=[]

        # convert to tf.variable so that they can be saved to network
        self.saved_optimizer = tf.Variable(self.optimizer)
        self.saved_lr = tf.Variable(self.lr)
        self.saved_loss_fun = tf.Variable(self.loss_fun)
                                                    
        if self.scaling:
            self.compute_scaling_factors(train_dataset)
            print('Scaling factors are computed using training dataset.')
            train_dataset = self.scaling_dataset(train_dataset)
            print('Training dataset are %s.' % ('standardized' if self.scaling =='std' else 'normalized'))
            if validation_dataset:
                validation_dataset = self.scaling_dataset(validation_dataset)
                print('Validation dataset are %s.' % ('standardized' if self.scaling =='std' else 'normalized'))

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality().numpy())
            print('Training dataset will be shuffled during training.')
            
        train_start_time = time.time()
        early_stop_repeats = 0
        for epoch in range(epochs):

            epoch_start_time = time.time()                
            
            # Iterate over the batches of the training dataset.
            for step, (input_dict, output_dict) in enumerate(train_dataset.batch(batch_size)):
                if decay_lr:
                    self.update_lr(epoch)
                    
                batch_loss = self.train_step(input_dict, output_dict, step=step, train_by_pe=train_by_pe, verbose=verbose, debug=debug)
                if step==0:
                    train_epoch_loss = batch_loss
                else:
                    for key in batch_loss:
                        train_epoch_loss[key] = batch_loss[key] + train_epoch_loss[key]

            print(30*'-')
            print(f'self.train_loss:{self.train_loss}')
            print(f'train_epoch_loss:{train_epoch_loss}')
            print(f'batch_loss:{batch_loss}')
            for key in batch_loss:
                if verbose:
                    print(f'key:{key}')
                train_epoch_loss[key] = train_epoch_loss[key]/(step+1)
                self.train_loss[key].append(tf.Variable(train_epoch_loss[key]))

            # Iterate over the batches of the validation dataset
            if validation_dataset is not None:  
                for step, (input_dict, output_dict) in enumerate(validation_dataset.batch(batch_size)):
                    batch_loss = self.validation_step(input_dict, output_dict, valid_by_pe=train_by_pe)
                    if step==0:
                        val_epoch_loss = batch_loss
                    else:
                        for key in batch_loss:
                            val_epoch_loss[key] = batch_loss[key] + val_epoch_loss[key]
                for key in batch_loss:
                    val_epoch_loss[key] = val_epoch_loss[key]/(step+1)
                    self.val_loss[key].append(tf.Variable(val_epoch_loss[key]))
                    
            epoch_end_time = time.time()
            time_per_epoch = (epoch_end_time - epoch_start_time)

            
            print('\n===> Epoch %i/%i - %.3fs/epoch - lr: %.5f' % (epoch+1, epochs, time_per_epoch, self.tf_optimizer.learning_rate))

            print('     training_loss   ',*["- %s: %5.3f" % (key,value) for key,value in train_epoch_loss.items()])
            if validation_dataset:
                print('     validation_loss ',*["- %s: %5.3f" % (key,value) for key,value in val_epoch_loss.items()])

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
                        print('\nTraining is stopped when train_loss <= %.3f for %i time.'%(early_stop['train_loss'][0],early_stop['train_loss'][1]))
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
                        print('\nTraining is stopped when val_loss <= %.3f for %i time.'%(early_stop['val_loss'][0],early_stop['val_loss'][1]))
                        break

            if nepochs_checkpoint:
                if epoch % nepochs_checkpoint == 0:
                    print('    Saving model - nepoch:', epoch)
                    save(self, 'saved_model_'+str(folder_id), descriptors)

                    if save_lossplots_on_checkpoint:
                        self.plot_loss(saveplots=True, folder_id=folder_id, pe_atom_loss_units=f'{loss_fun.upper()} - Potential Energy $[eV/atom]$', force_loss_units=f'{loss_fun.upper()} - Force Norm  $[eV/Å]$')
                    
        elapsed_time = (epoch_end_time - train_start_time)
        print('\nEnd of training, elapsed time: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        # self.plot_loss(saveplots=True, folder_id=folder_id, pe_atom_loss_units=f'{loss_fun.upper()} - Potential Energy $[eV/atom]$', force_loss_units=f'{loss_fun.upper()} - Force Norm  $[eV/Å]$')


        

    def plot_loss(self, start_epoch=2, figsize=(8,5), saveplots=False, folder_id=None, pe_atom_loss_units=None, force_loss_units=None):
        """
        Plot the losses.
        
        Args:
            start_epoch: plotting starts from start_epoch 
            figsize: set the fingersize, e.g. (8.5)
        """

        mpl.rc('legend', fontsize=15) 
        mpl.rc('xtick', labelsize=15) 
        mpl.rc('ytick', labelsize=15) 
        mpl.rc('axes', labelsize=15) 
        mpl.rc('figure', titlesize=25)

        for key in self.train_loss:
            fig, axs = plt.subplots(1, 1 ,figsize=figsize)
            epoch = np.arange(start_epoch,len(self.train_loss[key])+1)
            axs.plot(epoch,self.train_loss[key][start_epoch-1:],'-', markersize=5, fillstyle='none', linewidth=1, color= 'blue', label='train_loss', zorder=1)
            # axs.semilogy(epoch,self.train_loss[key][start_epoch-1:],'-', markersize=5, fillstyle='none', linewidth=1, color= 'blue', label='train_loss')
            axs.plot(epoch,self.val_loss[key][start_epoch-1:],'-', markersize=5, fillstyle='none', linewidth=1, color= 'darkgreen', label='val_loss', zorder=0)
            # axs.semilogy(epoch,self.val_loss[key][start_epoch-1:],'-', markersize=5, fillstyle='none', linewidth=1, color= 'darkgreen', label='val_loss')
            axs.set_title(key)
            axs.set_xlabel('epoch')
            if key=='pe_atom_loss' and pe_atom_loss_units:
                axs.set_ylabel(pe_atom_loss_units)
            elif key=='force_loss' and force_loss_units:
                axs.set_ylabel(force_loss_units)
            else:
                axs.set_ylabel('loss')
            axs.set_yscale('log')
            plt.legend(loc='upper right', frameon=True, borderaxespad=1)
            plt.tight_layout()
            axs.grid(True)
            if saveplots:
                print('key:', key)
                if not isinstance(folder_id, str):
                    folder_id = str(folder_id)
                if not os.path.isdir('plots'+folder_id):
                    os.mkdir('plots'+folder_id)
                plt.savefig('plots'+folder_id+'/'+key)
            plt.show()

        plt.close('all')
        


                                    
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
    
   
def save(obj, model_dir, descriptor=None):
    """
    Save a trained model.

    Args:
        model_dir: directory for the saved neural network model
        descriptor(dictionary): descriptor parameters, used for LAMMPS prediction
    """
    tf.saved_model.save(obj, model_dir)
    if descriptor is not None:
        input_tag, input_name, output_tag, output_name = get_signature(model_dir)
        file = open(model_dir+'/parameters','w')
        
        file.write('%-20s ' % 'element')
        file.write('  '.join(j for j in obj.elements))
        file.write('\n\n')
        
        file.write('%-20s %d\n' % ('input',len(input_tag)))
        for i in range(len(input_tag)):
            file.write('%-20s %-20s\n' % (input_tag[i],input_name[i]))
        file.write('\n')
        
        file.write('%-20s %d\n' % ('output',len(output_tag)))
        for i in range(len(output_tag)):
            file.write('%-20s %-20s\n' % (output_tag[i],output_name[i]))
        file.write('\n')

        file.write('%-20s %f\n' % ('cutoff',descriptor['cutoff']))
        file.write('\n')

        file.write('%-20s %s  %d\n' % ('descriptor',descriptor['name'], len(descriptor)-2))    
        for i in range(2,len(descriptor)):
            key  = list(descriptor.keys())[i]
            file.write('%-20s '% key)
            file.write('  '.join(str(j) for j in descriptor[key]))
            file.write('\n')
            
        file.close()
        print('Network signatures and descriptor are written to %s for LAMMPS simulation.'% (model_dir+'/parameters'))
    

        
def load(model_dir):
    """
    Load a saved model.

    Args:
        model_dir: directory of a saved neural network model
    """
    if model_dir:
        return Network(import_dir = model_dir)
    else:
        raise ValueError('Load function has no model directory.')


# def printout(eval_dict):


#             for i in range(len(pe)):
#                  print ("\n============================  image %i  =============================\n"%i+1)
#                  print ('potential energy = %15.8f' % pe[i])
#                  if compute_force:
#                      print ("%s %5s %15s %15s"%("atom_id","f_x","f_y","f_z"))
#                      for j in range(len(force[i][j])):
#                          print("%d %15.8f %15.8f %15.8f" % (j+1,force[i][j][0].numpy(), force[i][j][1].numpy(), force[i][j][2].numpy()))
                         
#                      print ("%s: %15.8f" %(pxx, stress[i][0]))
#                      print ("%s: %15.8f" %(pxy, stress[i][1]))
#                      print ("%s: %15.8f" %(pxz, stress[i][2]))
#                      print ("%s: %15.8f" %(pyx, stress[i][3]))
#                      print ("%s: %15.8f" %(pyy, stress[i][4]))
#                      print ("%s: %15.8f" %(pyz, stress[i][5]))
#                      print ("%s: %15.8f" %(pzx, stress[i][6]))
#                      print ("%s: %15.8f" %(pzy, stress[i][7]))
#                      print ("%s: %15.8f" %(pzz, stress[i][8]))

    
#                 print ("\n============================  image %i  =============================\n"%i+1)
#                 print ('%s %15s %15s %15s' % ('item','prediction','True','Error'))
#                 pe_error.append((pe-output_dict['pe'])/output_dict['pe'])
#                 print ('%s %15.6e %15.6e %15.6e' % ('pe',pe,output_dict['pe'],pe_error))
#                 if compute_force:
#                     for j in range(len(force[i])):
#                         print ('%s %15.6e %15.6e %15.6e' % ('',pe,output_dict['pe'],(pe-output_dict['pe'])/output_dict['pe']))
# maverick version
