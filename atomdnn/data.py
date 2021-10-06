import tensorflow as tf
import numpy as np
import os.path
from itertools import chain, repeat, islice
import time
import atomdnn
import random
import shutil

    
# slice dictionary
def slice (data_dict, start, end):
    keys = list(data_dict.keys())
    return {keys[i]: list(islice(data_dict[keys[i]],start, end)) for i in range(len(keys))}
    

# used to pad zeros to data 
def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))
    
    
def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


def get_input(tf_dataset):
    return [item[0] for item in list(dataset.as_numpy_iterator())]

def get_output(tf_dataset):
    return [item[1] for item in list(dataset.as_numpy_iterator())]




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
        self.num_atoms_in_fpfiles = None # number of atoms per image in fingerprints files
        self.num_atoms_in_forcefiles = None # number of atoms per image in force files

        self.num_types = None        
        self.num_images_der = None # number of derivative files
        self.num_fingerprints_der = None # number of fingerprints der
        self.maxnum_blocks = None # max number of blocks among all derivative files
        self.num_blocks_list = None # number of data block per image

        self.input_dict = {}
        self.output_dict = {}

        self.mean_fp = None
        self.dev_fp = None
        self.max_fp = None
        self.min_fp = None

        
        if fp_filename is not None and der_filename is not None:
            self.read_inputdata(fp_filename,der_filename,image_num)  
        elif fp_filename is not None:
            self.read_inputdata(fp_filename,image_num)

        
    def read_inputdata(self,fp_filename=None, der_filename=None, image_num=None, read_der=atomdnn.compute_force):

        self.read_fingerprints_from_lmpdump(fp_filename,image_num)

        if read_der:
            self.read_der_from_lmpdump(der_filename,image_num)
            
        self.check_consistance()



        
    def read_fingerprints_from_lmpdump(self, fp_filename=None, image_num=None):
        
        self.input_dict['fingerprints'] = []
        self.input_dict['atom_type'] = []
        self.input_dict['volume'] = []
        self.num_atoms_in_fpfiles = []
       
        #file_name = '/dump_fingerprints.'
        fd = 0 # count file
        maxnum_atom = 0 # max number of atoms among all the images

        if fp_filename.split('.')[-1] == '*':
            batch_mode = 1
            fp_filename = fp_filename[:-2]
        else:
            batch_mode = 0
            
        if not batch_mode:
            image_num = 1
            print("\nReading fingerprints data from LAMMPS dump files %s ..." % (fp_filename))
        else:
            print("\nReading fingerprints data from LAMMPS dump files %s ..." % (fp_filename+'.i'))
            
        while (1):
            if image_num:
                if fd+1>image_num:
                    break
            if batch_mode:
                filename = fp_filename + '.' + str(fd+1)
            else:
                filename = fp_filename
            if os.path.isfile(filename):
                try:
                    file = open(filename)
                except OSError:
                    raise OSError('Could not open file %s' % filename)
                lines = file.readlines()
                file.close()
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
                self.num_atoms_in_fpfiles.append(count_atom)
                fd += 1
                if fd > 0 and int(fd%50)==0:
                    print ('  Progress: read %d images ...' % fd)
            else:
                break

        print('  Finish reading fingerprints from %i images.\n' % fd)
        maxnum_atom = max(self.num_atoms_in_fpfiles)

        # pad zeros if the atom number is less than maxnum_atom
        for i in range(len(self.num_atoms_in_fpfiles)):
            if self.num_atoms_in_fpfiles[i] < maxnum_atom:
                print('  Pad image %i with zeros fingerprints.\n'%(i+1))
                zeros = [0] * len(self.input_dict['fingerprints'][i][0])
                self.input_dict['fingerprints'][i] = list(pad(self.input_dict['fingerprints'][i],maxnum_atom,zeros))
        
        self.num_images = np.shape(self.input_dict['fingerprints'])[0]
        self.num_atoms = np.shape(self.input_dict['fingerprints'])[1]
        self.num_fingerprints = np.shape(self.input_dict['fingerprints'])[2]
        self.num_types = max(self.input_dict['atom_type'][0])
        
        print('  image number = %d' % self.num_images)
        print('  max number of atom = %d' % self.num_atoms)
        print('  number of fingerprints = %d' % self.num_fingerprints)
        print('  type of atoms = %d' % self.num_types)            
        
        self.input_dict['fingerprints'] = tf.convert_to_tensor(self.input_dict['fingerprints'],dtype=self.data_type)
        self.input_dict['atom_type'] = tf.convert_to_tensor(self.input_dict['atom_type'],dtype='int32')
        
        # compute the mean, standard deviation, max and min of each fingerprint
        self.mean_fp = tf.math.reduce_mean(self.input_dict['fingerprints'],axis=[0,1])  
        self.dev_fp = tf.math.reduce_std(self.input_dict['fingerprints'],axis=[0,1])
        self.max_fp = tf.math.reduce_max(self.input_dict['fingerprints'],axis=[0,1])
        self.min_fp = tf.math.reduce_min(self.input_dict['fingerprints'],axis=[0,1])

        # check data consistance
        if 'dGdr' in list(self.input_dict.keys()):
            if self.num_images != len(self.input_dict['dGdr']):
                print ("\nWarning: The image numbers of fingerprints files and derivative files are not consistant.\n")
            if self.num_fingerprints != len(self.input_dict['dGdr'][0][0]):
                print ("\nWarning: The fingerprints numbers of fingerprints files and derivative files are not consistant.\n")
        if 'pe' in list(self.output_dict.keys()):
            if self.num_images != len(self.output_dict['pe']):
                print ("\nWarning: The image numbers of fingerprints files and pe files are not consistant.\n")
        if 'force' in list(self.output_dict.keys()):
            if self.num_images != len(self.output_dict['force']):
                print ("\nWarning: The image numbers of fingerprints files and force files are not consistant.\n")
            if self.num_atoms_in_fpfiles != self.num_atoms_in_forcefiles:
                print ("\nWarning: The atom numbers in fingerprints files and force files are not consistant.\n")




                
    def read_der_from_lmpdump(self,der_filename=None,image_num=None):
       
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

        
        self.input_dict['dGdr']=[]
        self.input_dict['neighbor_atom_coord']=[]
        self.input_dict['center_atom_id']=[]
        self.input_dict['neighbor_atom_id']=[]
        self.num_blocks_list=[]

        if der_filename.split('.')[-1] == '*':
            batch_mode = 1
            der_filename = der_filename[:-2]
        else:
            batch_mode = 0

        if not batch_mode:
            image_num = 1
            print("\nReading derivative data from one file %s ..." % der_filename)
        else:
            print("\nReading derivative data from a series of files %s ..." % (der_filename+'.i'))
            
        fd = 0 # count file/image
        max_block = 0 # max number of atoms among all the images
            
        while(1):
            if image_num:
                if fd+1>image_num:
                    break
            if batch_mode:
                filename = der_filename +'.'+ str(fd+1)
            else:
                filename = der_filename
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
                if fd > 0 and int(fd%50)==0:
                    print ('  Progress: read %d images ...' % fd)
            else:
                break
            
        print('  Finish reading dGdr derivatives from %i images.\n'% fd)
        maxnum_blocks = max(self.num_blocks_list)
        
        # pad zeros if the blocks is less than maxnum_blocks
        for i in range(len(self.num_blocks_list)):
            if self.num_blocks_list[i] < maxnum_blocks:
                print('  Pad image %i with zeros derivatives.\n'%(i+1))
                self.input_dict['center_atom_id'][i] = list(pad(self.input_dict['center_atom_id'][i],maxnum_blocks,0))
                self.input_dict['neighbor_atom_id'][i] = list(pad(self.input_dict['neighbor_atom_id'][i],maxnum_blocks,-1))
                zeros = [[0]]*3
                self.input_dict['neighbor_atom_coord'][i] = list(pad(self.input_dict['neighbor_atom_coord'][i],maxnum_blocks,zeros))
                zeros = [[0]*3]*len(self.input_dict['dGdr'][i][0])
                self.input_dict['dGdr'][i] = list(pad(self.input_dict['dGdr'][i],maxnum_blocks,zeros))

        self.num_images_der = np.shape(self.input_dict['dGdr'])[0]
        self.maxnum_blocks = maxnum_blocks
        self.num_fingerprints_der = np.shape(self.input_dict['dGdr'])[2]


        self.input_dict['dGdr'] = tf.convert_to_tensor(self.input_dict['dGdr'],dtype=self.data_type)
        self.input_dict['center_atom_id'] = tf.convert_to_tensor(self.input_dict['center_atom_id'],dtype='int32')
        self.input_dict['neighbor_atom_id'] = tf.convert_to_tensor(self.input_dict['neighbor_atom_id'],dtype='int32')
        self.input_dict['neighbor_atom_coord'] = tf.convert_to_tensor(self.input_dict['neighbor_atom_coord'],dtype=self.data_type)
        
        print('  image number = %d' % self.num_images_der)
        print('  max number of blocks = %d' % self.maxnum_blocks)
        print('  number of fingerprints = %d' % self.num_fingerprints_der)
        
        # check data consistance
        if 'fingerprints' in list(self.input_dict.keys()):
            if self.num_images != len(self.input_dict['dGdr']):
                print ("\nWarning: The image numbers of fingerprints files and derivative files are not consistant.\n")
            if self.num_fingerprints != len(self.input_dict['dGdr'][0][0]):
                print ("\nWarning: The fingerprints numbers of fingerprints files and derivative files are not consistant.\n")
        if 'pe' in list(self.output_dict.keys()):
            if len(self.input_dict['dGdr'][0]) != len(self.output_dict['pe']):
                print ("\nWarning: The image numbers of fingerprints files and pe files are not consistant.\n")
        if 'force' in list(self.output_dict.keys()):
            if len(self.input_dict['dGdr']) != len(self.output_dict['force']):
                print ("\nWarning: The image numbers of fingerprints files and force files are not consistant.\n")
        




                

    def read_outputdata_from_lmp(self, pe_stress_filename=None, force_filename=None, image_num=None, lmp_stress_unit_convert=None):

        # read potential energy and stress from lammps output file
        try:
            pe_stress_file = open(pe_stress_filename)
        except OSError:
            raise OSError('Could not open file %s' % pe_stress_filename)

        print("\nReading potential energy and stress from %s ..." % pe_stress_filename)

        if image_num:
            lines = pe_stress_file.readlines()[0:image_num+1]
        else:
            lines = pe_stress_file.readlines()
        pe_stress_file.close()

        self.output_dict['pe'] = [self.str2float(lines[i].split()[1]) for i in range(1,len(lines))]

        if lmp_stress_unit_convert:
            unit_convert = lmp_stress_unit_convert
        else:
            unit_convert = 1.0

        self.output_dict['stress'] = [-self.str2float(lines[i].split()[2:8]) * unit_convert for i in range(1,len(lines))]

        print('  Finish reading pe and stress from %i images.\n'%len(self.output_dict['pe']))

        self.output_dict['pe'] = tf.convert_to_tensor(self.output_dict['pe'],dtype=self.data_type)
        self.output_dict['stress'] = tf.convert_to_tensor(self.output_dict['stress'],dtype=self.data_type)

        # check data consistance
        if 'fingerprints' in list(self.input_dict.keys()):
            if self.num_images != len(self.output_dict['pe']):
                print ("\nWarning: The image numbers of fingerprints and potentials are not consistant.\n")

                
        # read forces from lammps dump files
        if force_filename is None:
            return
        
        try:
            force_file = open(force_filename)
        except OSError:
            raise OSError('Could not open file %s' % force_filename)
        
        print("\nReading force from %s ..." % force_filename)

        lines = force_file.readlines()    
        force_file.close()
        
        self.output_dict['force'] = []
        self.num_atoms_in_forcefiles = []
        check = 0
        fd = 0
        natom = 0
        count_atom = 0
        for i in range(len(lines)):                
            if 'fx' and 'fy' and 'fz' in lines[i]:
                check = 1
                self.output_dict['force'].append([])
            elif 'ITEM: TIMESTEP' in lines[i] and check==1:    
                check = 0
                self.num_atoms_in_forcefiles.append(count_atom)
                fd += 1
                count_atom = 0
                if image_num==fd:
                    break
                if fd > 0 and int(fd%50)==0:
                    print ('  Progress: force read %d images ...' % fd)
            elif check==1:
                self.output_dict['force'][fd].append(self.str2float(lines[i].split()[5:8]))
                count_atom += 1
        self.num_atoms_in_forcefiles.append(count_atom)        
        print('  Finish reading force from %i images.\n'% (fd+1))
        
        maxnum_atom = max(self.num_atoms_in_forcefiles)
        
        # pad zeros if the atom number is less than maxnum_atom
        for i in range(len(self.num_atoms_in_forcefiles)):
            if self.num_atoms_in_forcefiles[i] < maxnum_atom:
                print('  Pad image %i with zeros forces.\n'%(i+1))
                zeros = [0] * 3
                self.input_dict['force'][i] = list(pad(self.input_dict['force'][i],maxnum_atom,zeros))

        self.output_dict['force'] = tf.convert_to_tensor(self.output_dict['force'],dtype=self.data_type)        

        # check data consistance
        if 'fingerprints' in list(self.input_dict.keys()):
            if self.num_images != len(self.output_dict['force']):
                print ("\nWarning: The image numbers of fingerprints and forces are not consistant.\n")
            if self.num_atoms_in_fpfiles != self.num_atoms_in_forcefiles:
                print ("\nWarning: The atom numbers in fingerprints files and force files are not consistant.\n")

                

    def shuffel(self):
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


        
        
    # def create_tf_dataset(self, train_pct=None,val_pct=None,test_pct=None,data_size=None,shuffle=False):
    #     if not data_size:
    #         data_size = self.num_images

    #     if not train_pct:
    #         train_pct = 0.7
    #         print ("%f percent of data are used for training by default." % train_pct)
        
    #     if not val_pct:
    #         val_pct = 0.2
    #         print ("%f percent of data are used for validatiaon by default." % val_pct)
            
    #     if not test_pct:
    #         test_pct = 0.1
    #         print ("%f percent of data are used for test by default." % test_pct)
            
    #     train_size = int(train_pct * data_size)
    #     val_size = int(val_pct * data_size)
    #     test_size = data_size - train_size - val_size

    #     print('Traning data: %d images'% train_size)
    #     print('Validation data: %d images'% val_size )
    #     print('Test data: %d images'% test_size)

    #     full_dataset = tf.data.Dataset.from_tensor_slices((self.input_dict, self.output_dict))
    #     if shuffle:
    #         full_dataset.shuffle(buffer_size= full_dataset.cardinality().numpy())

    #     train_dataset = full_dataset.take(train_size)
    #     val_dataset = full_dataset.skip(train_size).take(val_size)
    #     test_dataset = full_dataset.skip(train_size+val_size)

    #     del full_dataset
    #     return train_dataset,val_dataset,test_dataset


    
        
    def split(self, train_pct=None, val_pct=None, test_pct=None, data_size=None):   

        if not data_size:
            data_size = self.num_images
            
        if not train_pct:
            train_pct = 0.7
            print ("%f percent of data are used for training by default." % train_data_percent)
        
        if not val_pct:
            val_pct = 0.2
            print ("%f percent of data are used for validatiaon by default." % val_data_percent)
            
        if not test_pct:
            test_pct = 0.1
            print ("%f percent of data are used for test by default." % test_data_percent)  
            
        train_size = int(train_pct * data_size)
        val_size = int(val_pct * data_size)
        test_size = data_size - train_size - val_size
        
        print('Traning data: %d images'% train_size)
        print('Validation data: %d images'% val_size )
        print('Test data: %d images'% test_size)
            

        x_train = slice(self.input_dict, 0, train_size)
        x_val = slice(self.input_dict, train_size, train_size + val_size)
        x_test = slice(self.input_dict, train_size + val_size, data_size)
        
        y_train = slice(self.output_dict, 0, train_size)
        y_val = slice(self.output_dict, train_size, train_size + val_size)
        y_test = slice(self.output_dict, train_size + val_size, data_size)

        return (x_train,y_train),(x_val,y_val),(x_test,y_test)
    
    def clean_up(self):
        if self.input_dict:
            del self.input_dict
        if self.output_dict:
            del self.output_dict
            

    
    def check_consistance(self):

        # check if numbers are consistant with those in derivative file
        if self.num_images_der is not None and self.num_images is not None:
            if self.num_images != self.num_images_der:
                print ("Warning: The image numbers in fingerprints files and derivative files are not consistant.")
            
        if self.num_fingerprints_der is not None and self.num_fingerprints is not None:
            if self.num_fingerprints != self.num_fingerprints_der:
                print ("Warning: The fingerprints numbers in fingerprints files and derivative files are not consistant.")
                




            
