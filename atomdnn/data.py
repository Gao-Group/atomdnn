import tensorflow as tf
import numpy as np
import os.path
from itertools import chain, repeat, islice
import time
import atomdnn
import random
import shutil
from ase.io import read,write        

# slice dictionary
def slice_dict (data_dict, start, end):
    keys = list(data_dict.keys())
    return {keys[i]: list(islice(data_dict[keys[i]],start, end)) for i in range(len(keys))}
    

# used to pad zeros to data 
def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))
    
    
def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)

    
class Data(object):
    
    def __init__(self,fp_filename=None, der_filename=None, image_num=None):
                
        self.data_type = atomdnn.data_type
        
        if self.data_type == 'float32':
            self.str2float = np.float32
        elif self.data_type == 'float64':
            self.str2float = np.float64
        else:
            raise ValueError('data_type has to be \'float32\' or \'float64\'.')
            
        self.num_images = None # number of fingerprints files
        self.num_fingerprints = None # number of fingerprints
        self.num_atoms = None # max number of atoms among all fingerprints files
        self.natoms_in_fpfiles = None # number of atoms per image in fingerprints files
        self.num_atoms_in_forcefiles = None # number of atoms per image in force files

        self.num_types = None        
        self.num_images_der = None # number of derivative files
        self.num_fingerprints_der = None # number of fingerprints der
        self.maxnum_blocks = None # max number of blocks among all derivative files
        self.num_blocks_list = None # number of data block per image

        self.input_dict = {}
        self.output_dict = {}

        
        if fp_filename is not None and der_filename is not None:
            self.read_inputdata(fp_filename,der_filename,image_num)  
        elif fp_filename is not None:
            self.read_inputdata(fp_filename,image_num)

        
    def read_inputdata(self,fp_filename=None, der_filename=None, image_num=None, append=False, read_der=atomdnn.compute_force):

        self.read_fingerprints_from_lmpdump(fp_filename,image_num, append)

        if read_der:
            self.read_der_from_lmpdump(der_filename,image_num, append)


        
    def read_fingerprints_from_lmpdump(self, fp_filename=None, image_num=None, append=False):

        if not append:
            self.input_dict['fingerprints'] = []
            self.input_dict['atom_type'] = []
            self.input_dict['volume'] = []
            self.natoms_in_fpfiles = []
            start_image = 0
        else:
            start_image = len(self.natoms_in_fpfiles)
            
            
        fd = start_image # count file
        maxnum_atom = 0 # max number of atoms among all the images
            
        if fp_filename.split('.')[-1] == '*':
            batch_mode = 1
            fp_filename = fp_filename[:-2]
        else:
            batch_mode = 0

        if not batch_mode:
            image_num = 1
            print("\nReading fingerprints data from LAMMPS dump files %s" % (fp_filename))
        else:
            print("\nReading fingerprints data from LAMMPS dump files %s" % (fp_filename+'.i'))
            
        while (1):
            if image_num:
                if fd-start_image+1>image_num:
                    break
            if batch_mode:
                filename = fp_filename + '.' + str(fd-start_image)
            else:
                filename = fp_filename

            print('filenamee:', filename)
            if os.path.isfile(filename):
                try:
                    file = open(filename)
                except OSError:
                    raise OSError('Could not open file %s' % filename)
                
                lines = file.readlines()
                file.close()
                print('fd:', fd)
                count_atom = 0
                self.input_dict['fingerprints'].append([])
                self.input_dict['atom_type'].append([])
                self.input_dict['volume'].append([])
                lx = self.str2float(lines[5].split()[1]) - self.str2float(lines[5].split()[0])
                ly = self.str2float(lines[6].split()[1]) - self.str2float(lines[6].split()[0])
                lz = self.str2float(lines[7].split()[1]) - self.str2float(lines[7].split()[0])
                volume = lx*ly*lz
                self.input_dict['volume'][fd].append(volume)
                for line in lines[9:]:
                    count_atom += 1
                    line = line.split()
                    self.input_dict['atom_type'][fd].append(int(line[1]))
                    self.input_dict['fingerprints'][fd].append([self.str2float(x) for x in line[2:]])
                    
                self.natoms_in_fpfiles.append(count_atom)
                fd += 1
                if fd-start_image > 0 and int((fd-start_image)%50)==0:
                    print ('  so far read %d images ...' % fd,flush=True)
            else:
                break
            
                
        print('  Finish reading fingerprints from total %i images.\n' % len(self.input_dict['fingerprints']), flush=True)
        maxnum_atom = max(self.natoms_in_fpfiles)

        # pad zeros if the atom number is less than maxnum_atom
        for i in range(len(self.input_dict['fingerprints'])):
            if len(self.input_dict['fingerprints'][i]) < maxnum_atom:
                print('  Pad image %i with zeros fingerprints.'%(i+1))
                zeros = [0] * len(self.input_dict['fingerprints'][i][0])
                self.input_dict['fingerprints'][i] = list(pad(self.input_dict['fingerprints'][i],maxnum_atom,zeros))
        
        self.num_images = np.shape(self.input_dict['fingerprints'])[0]
        self.num_atoms = np.shape(self.input_dict['fingerprints'])[1]
        self.num_fingerprints = np.shape(self.input_dict['fingerprints'])[2]
        self.num_types = max(self.input_dict['atom_type'][0])

        # self.input_dict['fingerprints'] = tf.convert_to_tensor(self.input_dict['fingerprints'],dtype=self.data_type)
        # self.input_dict['atom_type'] = tf.convert_to_tensor(self.input_dict['atom_type'],dtype='int32')
        # self.input_dict['volume'] = tf.convert_to_tensor(self.input_dict['volume'],dtype=self.data_type)
        
        print('  image number = %d' % self.num_images)
        print('  max number of atom = %d' % self.num_atoms)
        print('  number of fingerprints = %d' % self.num_fingerprints)
        print('  type of atoms = %d' % self.num_types)            



                
    def read_der_from_lmpdump(self,der_filename=None,image_num=None,append=False):
       
        # ==============================================================================
        #
        # variable used for force and stress calcualtion 
        #
        # dGdr: the derivative of fingerprints w.r.t atom coordiantes
        #
        # neighbor_atom_coord: 4D array to store the neighbor atom coordinates 
        #                      e.g. "neighbor_atom_coord[0][1][2][0]" means that: in the first image, 
        #                      in the second derivative data block, the z coordiate of the neighbor atom. 
        #                      Note that the last dimension of the array is 1 which is added for matrix 
        #                      multiplication in neural network force_stress_layer.
        #
        # center_atom_id: 2D array to store the center atom ID
        #                 e.g. "center_atom_id[0][1]=3" means that: in the first image, 
        #                 in the second derivative data block, the center atom ID is 3. 
        #                 The center atoms are non ghost atoms.
        #
        # neighbor_atom_id: 2D array to store the neighbor atom ID
        #                   e.g. "neighbor_atom_id[0][1]=5" means that: in the first image, 
        #                   in the second derivatve data block, the center atom's neigbor atom ID is 5. 
        #                   Note that the neighbors could be ghost atoms.
        #
        #===================================================================================

        if not append:
            self.input_dict['dGdr']=[]
            self.input_dict['neighbor_atom_coord']=[]
            self.input_dict['center_atom_id']=[]
            self.input_dict['neighbor_atom_id']=[]
            self.num_blocks_list=[]
            start_image = 0
        else:
            start_image = len(self.input_dict['dGdr'])

        if der_filename.split('.')[-1] == '*':
            batch_mode = 1
            der_filename = der_filename[:-2]
        else:
            batch_mode = 0

        if not batch_mode:
            image_num = 1
            print("\nReading derivative data from one file %s" % der_filename)
        else:
            print("\nReading derivative data from a series of files %s" % (der_filename+'.i'))

        print('This may take a while for large data set ...')
        
        start_time = time.time()    
        fd = start_image # count image
            
        while(1):
            if image_num:
                if fd-start_image+1>image_num:
                    break
            if batch_mode:
                filename = der_filename +'.'+ str(fd-start_image)
            else:
                filename = der_filename

            print('der_filename:', filename)
            if os.path.isfile(filename):
                self.input_dict['dGdr'].append([])
                self.input_dict['neighbor_atom_coord'].append([])
                self.input_dict['center_atom_id'].append([])
                self.input_dict['neighbor_atom_id'].append([])
                in_file = open(filename, 'r')
                lines = in_file.readlines()[9:]
                in_file.close()
                lines = [lines[i].split() for i in range(len(lines))]
                for i in range(0,len(lines),3):
                    block = lines[i:i+3]
                    self.input_dict['center_atom_id'][fd].append(int(block[1][0])-1) # center atom id, -1 to help with idexing
                    self.input_dict['neighbor_atom_id'][fd].append(int(block[1][1])-1) # neighbor atom id, -1 to help with idexing
                    self.input_dict['neighbor_atom_coord'][fd].append([[self.str2float(row[2])] for row in block[0:]])
                    self.input_dict['dGdr'][fd].append([[self.str2float(x[i]) for x in block[0:]]for i in range(3,len(block[0]))])  
                self.num_blocks_list.append(int(len(lines)/3))
                fd += 1
                if fd-start_image > 0 and int((fd-start_image)%50)==0:
                    print ('  so far read %d images ...' % fd,flush=True)
            else:
                break
        print('  Finish reading dGdr derivatives from total %i images.\n'% len(self.input_dict['dGdr']),flush=True)
        maxnum_blocks = max(self.num_blocks_list)

        print('  Pad zeros to derivatives data if needed ...',flush=True)
        # pad zeros if the blocks is less than maxnum_blocks
        count_pad = 0
        for i in range(len(self.input_dict['dGdr'])):
            if len(self.input_dict['dGdr'][i]) < maxnum_blocks:
                self.input_dict['center_atom_id'][i] = list(pad(self.input_dict['center_atom_id'][i],maxnum_blocks,0))
                self.input_dict['neighbor_atom_id'][i] = list(pad(self.input_dict['neighbor_atom_id'][i],maxnum_blocks,-1))
                zeros = [[0]]*3
                self.input_dict['neighbor_atom_coord'][i] = list(pad(self.input_dict['neighbor_atom_coord'][i],maxnum_blocks,zeros))
                zeros = [[0]*3]*len(self.input_dict['dGdr'][i][0])
                self.input_dict['dGdr'][i] = list(pad(self.input_dict['dGdr'][i],maxnum_blocks,zeros))
                count_pad += 1
                if int((count_pad)%50)==0:
                    print ('  so far paded %d images ...' % count_pad,flush=True)
 
        if count_pad ==0:
            print('  Pading is not needed.')
        else:
            print('  Pading finished: %i images derivatives have been padded with zeros.'%count_pad,flush=True)
        self.num_images_der = np.shape(self.input_dict['dGdr'])[0]
        self.maxnum_blocks = maxnum_blocks
        self.num_fingerprints_der = np.shape(self.input_dict['dGdr'])[2]
        
        print('\n  image number = %d' % self.num_images_der,flush=True)
        print('  max number of derivative pairs = %d' % self.maxnum_blocks,flush=True)
        print('  number of fingerprints = %d' % self.num_fingerprints_der,flush=True)
        
        end_time = time.time()
        print('\n  It took %.2f seconds to read the derivatives data.'%(end_time-start_time),flush=True)


        
        
    def read_outputdata(self, xyzfile_path=None, xyzfile_name=None, read_force=True, read_stress=True, image_num=None, append=False, verbose=False):

        if not append:
            self.output_dict['pe'] = []
            if read_stress:
                self.output_dict['stress'] = []
            if read_force:
                self.output_dict['force'] = []
                self.natoms_in_force = []
            start_image = 0
        else:
            start_image = len(self.output_dict['pe'])
        
        print("Reading outputs from extxyz files ...")

        if xyzfile_name.split('.')[-1] == '*':
            batch_mode = 1
            xyzfile_name = xyzfile_name[:-2]
        else:
            batch_mode = 0    
        if not batch_mode:
            image_num = 1

        fd = start_image # count image
        while (1):
            if image_num:
                if fd-start_image+1>image_num:
                    break
            if batch_mode:
                filename = xyzfile_path + '/' + xyzfile_name +'.'+ str(fd-start_image)
            else:
                filename = xyzfile_path + '/' + xyzfile_name

            if verbose:
                print('filename:', filename)

            if os.path.isfile(filename):
                patom = read(filename, index=':', format='extxyz')[0]
                pe = patom.get_potential_energy()
                self.output_dict['pe'].append(pe)
                if read_force:
                    force = patom.get_forces(patom)
                    self.output_dict['force'].append(force)
                    self.natoms_in_force.append(len(force))
                if read_stress:
                    stress = patom.get_stress()
                    
                    self.output_dict['stress'].append(stress)
                fd += 1
                if fd-start_image > 0 and int((fd-start_image)%50)==0:
                    print ('  so far read %d images ...' % fd,flush=True)
            else:
                break
            
        print('  Finish reading outputs from total %i images.\n' % len(self.output_dict['pe']),flush=True)
        if read_force:
            maxnum_atom = max(self.natoms_in_force)

        # pad zeros if the atom number is less than maxnum_atom
        if read_force:
            for i in range(len(self.output_dict['force'])):
                if len(self.output_dict['force'][i]) < maxnum_atom:
                    print('  Pad image %i with zeros forces.'%(i+1))
                    zeros = [0] * 3
                    self.output_dict['force'][i] = list(pad(self.output_dict['force'][i],maxnum_atom,zeros))
                

                    
    def shuffle(self):
        zipped = list(zip(*[self.input_dict[keys] for keys in list(self.input_dict.keys())],*[self.output_dict[keys] \
                                                                                for keys in list(self.output_dict.keys())]))
        random.shuffle(zipped)
        unzipped= list(zip(*zipped))
        i=0
        for keys in list(self.input_dict.keys()):
            self.input_dict[keys] = unzipped[i]
            i+=1
        for keys in list(self.output_dict.keys()):
            self.output_dict[keys] = unzipped[i]
            i+=1
        del zipped,unzipped


        
        
    def create_tf_dataset(self, train_pct=None, val_pct=None, test_pct=None, shuffle=False, data_size=None):

        self.check_data() # check the data
        
        if not data_size:
            data_size = self.num_images

        if not train_pct:
            raise ValueError('Need to set the percentage of data used for tranining.')
        
        if not val_pct:
            val_pct = 0
            print ("No data are used for validatiaon by default." % val_pct)
            
        if not test_pct:
            test_pct = 0
            print ("No data are used for test by default." % test_pct)

        if train_pct+val_pct+test_pct>1:
            raise ValueError('Percentages are not correct.')
            
        train_size = int(train_pct * data_size)
        val_size = int(val_pct * data_size)
        test_size = int(test_pct * data_size)

        print('Traning data: %d images'% train_size)
        print('Validation data: %d images'% val_size )
        print('Test data: %d images'% test_size)

        if not shuffle:
            print ('Data are not shuffled by default, set shuffle=True if needed.')

        print("Creating tensorflow datasets, this may take more than 10 minites for large dataset ...")
        start_time = time.time()
        full_dataset = tf.data.Dataset.from_tensor_slices((self.input_dict, self.output_dict))
        if shuffle:
            full_dataset.shuffle(buffer_size= full_dataset.cardinality().numpy())
        train_dataset = full_dataset.take(train_size)
        val_dataset = full_dataset.skip(train_size).take(val_size)
        test_dataset = full_dataset.skip(train_size+val_size).take(test_size)
        del full_dataset
        end_time = time.time()
        print('It took %.3f seconds to create tensorflow datasets' % (end_time-start_time))
        
        return train_dataset,val_dataset,test_dataset


    
        
    def slice(self, start=None, end=None):   
        # check data
        self.check_data()            
        input_dict = slice_dict(self.input_dict, start, end)
        output_dict = slice_dict(self.output_dict, start, end)
        return input_dict, output_dict
    

        
    def check_data(self):
        if 'fingerprints' in list(self.input_dict.keys()):            
            if 'dGdr' in list(self.input_dict.keys()):
                if len(self.input_dict['fingerprints']) != len(self.input_dict['dGdr']):
                    raise ValueError("The image numbers of fingerprints files and derivative files are not consistant.")
                if self.num_fingerprints != len(self.input_dict['dGdr'][0][0]):
                    raise ValueError("The fingerprints numbers of fingerprints files and derivative files are not consistant.")
            if 'pe' in list(self.output_dict.keys()):
                if self.num_images != len(self.output_dict['pe']):
                    raise ValueError("The image numbers of fingerprints files and pe files are not consistant.")
            else:
                raise ValueError("No potential energy in output data.")    
            if 'force' in list(self.output_dict.keys()):
                if self.num_images != len(self.output_dict['force']):
                    raise ValueError("The image numbers of fingerprints files and force files are not consistant.")
                if self.natoms_in_fpfiles != self.natoms_in_force:
                    raise ValueError("The atom numbers in fingerprints files and force files are not consistant.")
            if 'stress' in list(self.output_dict.keys()):
                if self.num_images != len(self.output_dict['stress']):
                    raise ValueError("The image numbers of fingerprints files and stress files are not consistant.")            
        else:
            raise ValueError('No fingerprints in input data.')



    def convert_data_to_tensor(self):
        #self.check_data()
        print('Conversion may take a while for large datasets...',flush=True)
        start_time = time.time()
        for key in self.output_dict:
            self.output_dict[key] = tf.convert_to_tensor(self.output_dict[key],dtype=self.data_type)

        for key in self.input_dict:
            if key=='fingerprints' or key=='volume' or key=='dGdr' or key=='neighbor_atom_coord':
                self.input_dict[key] = tf.convert_to_tensor(self.input_dict[key],dtype=self.data_type)
            else:
                self.input_dict[key] = tf.convert_to_tensor(self.input_dict[key],dtype='int32')
        end_time = time.time()
        print('It took %.4f second.'%(end_time-start_time),flush=True)



#=================================================================================================================
# some functions to operate tensorflow dataset

def get_input_dict(dataset):
    for x,y in dataset.batch(len(dataset)):
        return x

def get_output_dict(dataset):
    for x,y in dataset.batch(len(dataset)):
        return y

# # get the dictionary element from tensorflow dataset
# def get_element(dataset,key):
#     if key in ['fingerprints','atom_type','volume','dGdr','center_atom_id','neighbor_atom_id','neighbor_atom_coord']:
#         element = dataset.map(lambda input_dict, output_dict: input_dict[key])
#     elif key in ['pe','force','stress']:
#         element = dataset.map(lambda input_dict, output_dict: output_dict[key])
#     else:
#         raise ValueError('Key is not found in the dataset dictionary')
#     element = [item for item in list(element.as_numpy_iterator())]
#     return element


def get_fingerprints_num (dataset):
    if dataset.element_spec[0]['fingerprints'].shape[1] is not None: 
        return dataset.element_spec[0]['fingerprints'].shape[1]
    else:
        return len(get_input_dict(dataset)['fingerprints'][0][0]) # for tensorflow version below 2.6
    

def slice_dataset(dataset, start, end):
    return dataset.skip(start).take(end-start)


def split_dataset(dataset, train_pct=None, val_pct=None, test_pct=None, shuffle=False,data_size=None):
    if not data_size:
        data_size = dataset.cardinality().numpy()
    if not train_pct:
        raise ValueError('Need to set the percentage of data used for tranining.')
    if not val_pct:
        val_pct = 0
        print ("No data are used for validatiaon by default." % val_pct)
    if not test_pct:
        test_pct = 0
        print ("No data are used for test by default." % test_pct)
    if train_pct+val_pct+test_pct>1:
        raise ValueError('Percentages are not correct.')
    train_size = int(train_pct * data_size)
    val_size = int(val_pct * data_size)
    test_size = int(test_pct * data_size)
    print('Traning data: %d images'% train_size)
    print('Validation data: %d images'% val_size )
    print('Test data: %d images'% test_size)
    if not shuffle:
        print ('Data are not shuffled by default, set shuffle=True if needed.')
    if shuffle: # note that: reshuffle_each_iteration has to be False
        dataset = dataset.shuffle(buffer_size=dataset.cardinality().numpy(),reshuffle_each_iteration=False)
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size+val_size).take(test_size)

    return train_dataset,val_dataset,test_dataset
