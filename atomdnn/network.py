import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import time
import os
import atomdnn
from atomdnn.data import *

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
            self.loss_fn=None # string
            self.tf_loss_fn=None
            self.optimizer=None # string
            self.tf_optimizer=None
            self.lr=None
#            self.loss_weights=None
                
        self.saved_num_fingerprints = tf.Variable(self.num_fingerprints)
        self.saved_arch = tf.Variable(self.arch)
        self.saved_activation_function = tf.Variable(self.activation_function)
        self.saved_data_type = tf.Variable(self.data_type)
        self.saved_elements = tf.Variable(self.elements)

        
        
    def inflate_from_file(self, imported):
        '''
        Inflate network object from SavedModel 
        :param imported_object: Imported object from `tf.saved_model.load(saved_model_export_dir)`
        '''
        self.built = True
        self.arch = imported.saved_arch.numpy()
        self.num_fingerprints = imported.saved_num_fingerprints.numpy()

        if imported.scaling:
            self.scaling = imported.scaling
            self.scaling_factor = imported.scaling_factor
        else:
            self.scaling = None
            
        self.activation_function = imported.saved_activation_function.numpy().decode()
        self.tf_activation_function = tf.keras.activations.get(self.activation_function)
        self.data_type = imported.saved_data_type.numpy().decode()
        self.elements = [imported.saved_elements.numpy()[i].decode() for i in range(imported.saved_elements.shape[0])]
           
        self.lr = imported.saved_lr.numpy()
        self.loss_fn = imported.saved_loss_fn.numpy().decode()
        self.tf_loss_fn = tf.keras.losses.get(self.loss_fn)
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
        if len(x)==0:
            print('__call__ method - input is empty.')
            return {'pe':None,'force': None, 'stress':None}
        else:
            return self._predict(x)
            
        
    def _build(self):
        '''
        Initializes weights for each layer
        '''
        
        # Declare layer-wise weights and biases
        self.W1 = tf.Variable(
            self.weights_initializer(shape=(self.num_fingerprints,self.arch[0]),dtype=self.data_type), name='W1')
        self.b1 = tf.Variable(self.bias_initializer(shape=(1,self.arch[0]), dtype=self.data_type),name='b1')
        self.params.append(self.W1)
        self.params.append(self.b1)
        nlayer = 1
        
        for nneuron_layer in self.arch[1:]:
            self.params.append(tf.Variable(self.weights_initializer(shape=(self.arch[nlayer-1], self.arch[nlayer]),dtype=self.data_type)))
            self.params.append(tf.Variable(self.bias_initializer([1,self.arch[nlayer]],dtype=self.data_type)))
            nlayer+=1
        
        self.params.append(tf.Variable(self.weights_initializer(shape=(self.arch[-1],1),dtype=self.data_type)))
        self.params.append(tf.Variable(self.bias_initializer([1, 1],dtype=self.data_type)))

     
    
    def compute_pe (self, fingerprints, atom_type):
        '''
        Forward pass of the network
        '''
            
        mask_1 = tf.ones([tf.shape(atom_type)[0],tf.shape(atom_type)[1]],dtype='int32')*1
        
        type_onehot_1 = tf.linalg.diag(tf.cast(tf.math.equal(atom_type,mask_1),dtype=self.data_type))
        
#         mask_2 = tf.ones([tf.shape(atom_type)[0],tf.shape(atom_type)[1]],dtype='int32')*2
        
#         type_onehot_2 = tf.linalg.diag(tf.cast(tf.math.equal(atom_type,mask),dtype='float32'))
        
        
        fingerprints_1 = tf.matmul(type_onehot_1,fingerprints)
#         fingerprints_2 = tf.matmul(type_onehot_2,fingerprints)
        
        Z = tf.matmul(fingerprints_1,self.params[0]) + self.params[1]
        Z = self.tf_activation_function(Z)
        
        nlayer = 1
        params_iter = 0

        for nneuron_layer in self.arch[1:]:
            nlayer+=1
            params_iter+=2
            Z = tf.matmul(Z,self.params[params_iter]) + self.params[params_iter+1]
            Z = self.tf_activation_function(Z)

        atom_pe = tf.matmul(Z, self.params[-2]) + self.params[-1]

        total_pe = tf.reshape(tf.math.reduce_sum(atom_pe, axis=1),[tf.shape(atom_pe)[0]])

        return total_pe, atom_pe
    
    
    
    def compute_force_stress (self, dEdG, input_dict):
                
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



    # only used for __call__ method
    def _compute_force_stress (self, dEdG, input_dict):

        if hasattr(self,'scaling'):
            if self.scaling==tf.Variable('std'):
                dEdG = dEdG/self.scaling_factor[1]
            elif self.scaling==tf.Variable('norm'):
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


    
    def predict(self, input_dict,compute_force=atomdnn.compute_force,training=False):

        if not self.built:
            self._build()
            self.built = True

        fingerprints = input_dict['fingerprints']
        
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
                return {'pe':total_pe} 
            else:
                return {'pe':total_pe.numpy()} 
        else: 
            with tf.GradientTape() as dEdG_tape:
                dEdG_tape.watch(fingerprints)
                total_pe, atom_pe = self.compute_pe(fingerprints,input_dict['atom_type'])
            dEdG = dEdG_tape.gradient(atom_pe, fingerprints)
            force, stress = self.compute_force_stress(dEdG, input_dict)
            if training:
                return {'pe':total_pe, 'force':force, 'stress':stress}
            else:
                return {'pe':total_pe.numpy(), 'force':force.numpy(), 'stress':stress.numpy()}

        
                     
    # only used for __call__ method
    def _predict(self, input_dict,compute_force=atomdnn.compute_force):

        fingerprints = input_dict['fingerprints']

        if hasattr(self,'scaling'):
            if self.scaling == tf.Variable('std'):  # standardize with the mean and deviation
                fingerprints = (fingerprints - self.scaling_factor[0])/self.scaling_factor[1]
            elif self.scaling == tf.Variable('norm'): # normalize with the minimum and maximum
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

        

    def evaluate(self,dataset,batch_size=None,compute_force=atomdnn.compute_force):

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
            print('%15s:  %15.4e' % (key, eval_loss[key]))

        print ('\nThe prediction is:')
        return y_predict
                                                
        
        
    def loss(self, true_dict, pred_dict, training=False, validation=False):

        if self.tf_loss_fn is None:
            raise ValueError('Need to set up loss function.')
        else:
            pe_loss = self.tf_loss_fn(true_dict['pe'], pred_dict['pe'])

        if 'force' in pred_dict and 'force' in true_dict: # compute force loss
            # when evaluation OR train/validation using force OR all losses are requested 
            if (not training and not validation) or ((training or validation) and self.loss_weights['force']!=0) or self.compute_all_loss: 
                force_loss = tf.reduce_mean(self.tf_loss_fn(true_dict['force'],pred_dict['force']))
                
        if 'stress' in pred_dict and 'stress' in true_dict: # compute stress loss
            # when evaluation OR train/validation using stress OR all losses are requested
            if (not training and not validation) or ((training or validation) and (self.loss_weights['stress']!=0)) or self.compute_all_loss: 
                mask = [True,True,True,False,True,True,False,False,True] # select upper triangle elements of the stress tensor
                stress = tf.reshape(tf.boolean_mask(pred_dict['stress'], mask,axis=1),[-1,6]) # reduce to 6 components
                stress_loss = tf.reduce_mean(self.tf_loss_fn(true_dict['stress'],stress))

        if self.loss_weights['force']==0 and self.loss_weights['stress']==0: # only pe used for training
            total_loss = pe_loss
            loss_dict = {'pe_loss':pe_loss.numpy()}
        elif self.loss_weights['force']!=0 and self.loss_weights['stress']!=0: # pe, force and stress used for training
            total_loss = pe_loss * self.loss_weights['pe'] + force_loss * self.loss_weights['force'] + stress_loss * self.loss_weights['stress']
            loss_dict = {'pe_loss':pe_loss.numpy(), 'force_loss':force_loss.numpy(), 'stress_loss':stress_loss.numpy(), 'total_loss':total_loss.numpy()}
        elif self.loss_weights['force']!=0: # pe and force used for training 
            total_loss = pe_loss * self.loss_weights['pe'] + force_loss * self.loss_weights['force']
            loss_dict = {'pe_loss':pe_loss.numpy(), 'force_loss':force_loss.numpy(), 'total_loss':total_loss.numpy()}
        elif self.loss_weights['stress']!=0: # pe and stress used for training
            total_loss = pe_loss * self.loss_weights['pe'] + stress_loss * self.loss_weights['stress']
            loss_dict = {'pe_loss':pe_loss.numpy(), 'stress_loss':stress_loss.numpy(), 'total_loss':total_loss.numpy()}
        else:
            raise ValueError('loss_weights is not set correctly.')

        if (not training and not validation) or self.compute_all_loss:
            loss_dict = {'pe_loss':pe_loss.numpy(), 'force_loss':force_loss.numpy(), 'stress_loss':stress_loss.numpy(), 'total_loss':total_loss.numpy()}
            
        if training:
            return total_loss,loss_dict
        else:
            return loss_dict



     
    def train_step(self, input_dict, output_dict):    
        with tf.GradientTape() as tape: # automatically watch all tensorflow variables 
            if self.loss_weights['force']==0 and self.loss_weights['stress']==0:
                pred_dict = self.predict(input_dict, training=True,compute_force=False)
            else:
                pred_dict = self.predict(input_dict, training=True,compute_force=True)
            total_loss,loss_dict = self.loss(output_dict, pred_dict,training=True)
        grads = tape.gradient(total_loss, self.params)
        self.tf_optimizer.apply_gradients(zip(grads, self.params))
        return loss_dict


    
    def validation_step(self, input_dict, output_dict):
        if self.loss_weights['force']==0 and self.loss_weights['stress']==0:
            pred_dict = self.predict(input_dict,training=True,compute_force=False)
        else:
            pred_dict = self.predict(input_dict,training=True,compute_force=True)
            loss_dict = self.loss(output_dict, pred_dict,validation=True)
        return loss_dict
    

    def compute_scaling_factors(self,train_dataset):
        self.scaling_factor = []
        fingerprints = get_input_dict(train_dataset)['fingerprints']
        if self.scaling == 'std':
            self.scaling_factor.append(tf.math.reduce_mean(fingerprints,axis=[0,1]))  
            self.scaling_factor.append(tf.math.reduce_std(fingerprints,axis=[0,1]))
        if self.scaling == 'norm':
            self.scaling_factor.append(tf.math.reduce_max(fingerprints,axis=[0,1]))
            self.scaling_factor.append(tf.math.reduce_min(fingerprints,axis=[0,1]))
        self.scaling_factor = tf.Variable(self.scaling_factor) # convert to tf.Variable for saving

        
    def scaling_dataset(self,dataset):
        def map_fun(input_dict,output_dict):
            if self.scaling == 'std':
                input_dict['fingerprints'] = (input_dict['fingerprints'] - self.scaling_factor[0])/self.scaling_factor[1]
            if self.scaling == 'norm':
                input_dict['fingerprints'] = (input_dict['fingerprints'] - self.scaling_factor[0])/(self.scaling_factor[1] - self.scaling_factor[0])
            return input_dict,output_dict
        return dataset.map(map_fun)

    
    def train(self, train_dataset, validation_dataset=None, scaling=None, batch_size=None, epochs=None, loss_fn=None, \
              optimizer=None, lr=None, loss_weights=None, compute_all_loss=False, shuffle=True):

        if not self.built:            
            self._build()
            self.built = True
            
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
            self.optimizer = optimizer
            self.tf_optimizer = tf.keras.optimizers.get(optimizer)
            K.set_value(self.tf_optimizer.learning_rate, self.lr)
        elif not self.optimizer and not self.tf_optimizer:
            self.optimizer = 'Adam'
            self.tf_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) 
            print ("optimizer is set to Adam by default.")
            
        if loss_fn:
            self.loss_fn = loss_fn
            self.tf_loss_fn = tf.keras.losses.get(self.loss_fn)
        elif not self.loss_fn and not self.tf_loss_fn:
            self.loss_fn = 'mae'
            self.tf_loss_fn = tf.keras.losses.get(self.loss_fn)
            print ("loss_fn is set to mae by default.")
                                   
        if not epochs:
            epochs = 1
            print ("epochs is set to 1 by default.")

        if batch_size is None:
            batch_size = 30
            print ("batch_size is set to 30 by default.")    

        if not hasattr(self,'train_loss'):
            self.train_loss={}
            self.train_loss['epoch']=[]
        
        if validation_dataset and not hasattr(self,'val_loss'):
            self.val_loss={}
            self.val_loss['epoch']=[]
            
        if loss_weights is None:
            self.loss_weights = {'pe':1,'force':0,'stress':0}
            print('loss_weights is set to default value:',self.loss_weights)
        else:
            self.loss_weights = loss_weights
        for key in self.loss_weights:
            self.loss_weights[key] = tf.Variable(loss_weights[key],dtype=self.data_type)

        if compute_all_loss:
            self.compute_all_loss=True
        else:
            self.compute_all_loss=False        

        if self.loss_weights['pe']!=0 or self.compute_all_loss:
            if 'pe_loss' not in self.train_loss:
                self.train_loss['pe_loss']=[]
            if validation_dataset and 'pe_loss' not in self.val_loss:
                self.val_loss['pe_loss']=[]
            
        if self.loss_weights['force']!=0 or self.compute_all_loss: # when force is used for training/validation OR compute force_loss is requested     
            if 'force_loss' not in self.train_loss:
                self.train_loss['force_loss']=[]
            if validation_dataset and 'force_loss' not in self.val_loss:
                self.val_loss['force_loss']=[]
                
        if self.loss_weights['force']!=0:
            print ("Forces are used for training.")
        else:
            print ("Forces are not used for training.")
            
        if self.loss_weights['stress']!=0 or self.compute_all_loss: # when stress is used for traning/validation OR compute stress_loss is requested
            if 'stress_loss' not in self.train_loss:
                self.train_loss['stress_loss']=[]
            if validation_dataset and 'stress_loss' not in self.val_loss:
                self.val_loss['stress_loss']=[]
                
        if self.loss_weights['stress']!=0:
            print ("Stresses are used for training.")
        else:
            print ("Stresses are not used for training.")

        if self.loss_weights['force']!=0 or self.loss_weights['stress']!=0 or self.compute_all_loss: 
            if 'total_loss' not in self.train_loss:
                self.train_loss['total_loss']=[]
            if validation_dataset and 'total_loss' not in self.val_loss:
                self.val_loss['total_loss']=[]

        # convert to tf.variable so that they can be saved to network
        self.saved_optimizer = tf.Variable(self.optimizer)
        self.saved_lr = tf.Variable(self.lr)
        self.saved_loss_fn = tf.Variable(self.loss_fn)
                                                    
        if self.scaling:
            self.compute_scaling_factors(train_dataset)
            print('Scaling factors are computed using training dataset.')
            train_dataset = self.scaling_dataset(train_dataset)
            print('Training dataset are %s.' % 'standardized' if self.scaling =='std' else 'normalized')
            if validation_dataset:
                validation_dataset = self.scaling_dataset(validation_dataset)
                print('Validation dataset are %s.' % 'standardized' if self.scaling =='std' else 'normalized')

        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=train_dataset.cardinality().numpy())
            print('Training dataset will be shuffled during training.')

        if self.train_loss['epoch']:
            pre_train_epochs = self.train_loss['epoch'][-1].numpy()
        else:
            pre_train_epochs = 0
        if self.val_loss['epoch']:
            pre_val_epochs = self.train_loss['epoch'][-1].numpy()
        else:
            pre_val_epochs = 0

        train_start_time = time.time()
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Iterate over the batches of the training dataset.
            for step, (input_dict, output_dict) in enumerate(train_dataset.batch(batch_size)):
                batch_loss = self.train_step(input_dict, output_dict)
                if step==0:
                    train_epoch_loss = batch_loss
                else:
                    for key in batch_loss:
                        train_epoch_loss[key] = batch_loss[key] + train_epoch_loss[key]
            for key in batch_loss:
                train_epoch_loss[key] = train_epoch_loss[key]/(step+1)
                self.train_loss[key].append(tf.Variable(train_epoch_loss[key]))
            self.train_loss['epoch'].append(tf.Variable(pre_train_epochs+epoch+1))
                
            # Iterate over the batches of the validation dataset
            if validation_dataset is not None:  
                for step, (input_dict, output_dict) in enumerate(validation_dataset.batch(batch_size)):
                    batch_loss = self.validation_step(input_dict, output_dict)
                    if step==0:
                        val_epoch_loss = batch_loss
                    else:
                        for key in batch_loss:
                            val_epoch_loss[key] = batch_loss[key] + val_epoch_loss[key]
                for key in batch_loss:
                    val_epoch_loss[key] = val_epoch_loss[key]/(step+1)
                    self.val_loss[key].append(tf.Variable(val_epoch_loss[key]))
                self.val_loss['epoch'].append(tf.Variable(pre_val_epochs+epoch+1))
                    
            epoch_end_time = time.time()
            time_per_epoch = (epoch_end_time - epoch_start_time)
            print('\n===> Epoch %i/%i - %.3fs/epoch' % (epoch+1, epochs, time_per_epoch))

            print('     training_loss   ',*["- %s: %5.3f" % (key,value) for key,value in train_epoch_loss.items()])
            if validation_dataset:
                print('     validation_loss ',*["- %s: %5.3f" % (key,value) for key,value in val_epoch_loss.items()])

        elapsed_time = (epoch_end_time - train_start_time)
        print('\nEnd of training, elapsed time: ',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


        
def get_signature(model_dir):
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
    stream = os.popen('saved_model_cli show --dir '+ model_dir +' --tag_set serve --signature_def serving_default')
    output = stream.read()
    print(output)
    
   
def save(obj, model_dir, descriptor=None):
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
