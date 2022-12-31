import tensorflow as tf
import numpy as np
import os.path
from itertools import chain, repeat, islice
import time
import atomdnn
import random
import shutil
from ase.io import read,write
from atomdnn.descriptor import get_filenames
import glob
from atomdnn import color

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
    """
    Create Data object, with an option to read inputs and outputs. \
    Parameters are explained in :func:`~atomdnn.data.Data.read_inputdata`.
    """

    
    def __init__(self, descriptors_path=None, fp_filename=None, der_filename=None, \
                 xyzfiles=None, format='extxyz',image_num=None, skip=0, \
                 verbose=False, silent=False, read_der=atomdnn.compute_force,**kwargs):

        self.data_type = atomdnn.data_type
        
        if self.data_type == 'float32':
            self.str2float = np.float32
        elif self.data_type == 'float64':
            self.str2float = np.float64
        else:
            raise ValueError('data_type has to be \'float32\' or \'float64\'.')
            
        self.print_inputinfo_together = False


        self.input_dict = {}
        self.output_dict = {}
        
        if descriptors_path is not None and fp_filename is not None:
            self.read_inputdata(descriptors_path,fp_filename,der_filename,verbose=verbose,silent=silent,read_der=read_der)
            
        if xyzfiles is not None:
            self.read_outputdata(xyzfiles,format,image_num,skip,verbose=verbose,silent=silent,**kwargs)


        
    def read_inputdata(self,descriptors_path, fp_filename, der_filename=None, image_num=None, skip=0, append=False, verbose=False, silent=False, read_der=atomdnn.compute_force):
        """
        Read input data from :func:`~atomdnn.data.Data.read_fingerprints_from_lmpdump` and :func:`~atomdnn.data.Data.read_der_from_lmpdump`.
        
        Args:
            descriptor_path: directory to descriptor files
            fp_filename: file names for descriptors, use '*' for multiple files order numerically
            der_filename: file names for derivatives, use '*' for multiple files order numerically
            image_num: None if read all files given by the fp_filename
            skip(int): skip some images 
            append(bool): True if append inputs to already existing data object
            verbose(bool): True to show all reading file names 
            read_der(bool): True if read derivatives

        """
        if append and len(self.input_dict)==0:
            print(color.RED + 'Warning: data is not appendable, append has been set to False.' + color.END)
            append = False

        self.print_inputinfo_together = True
        
        self.read_fingerprints_from_lmpdump(descriptors_path,fp_filename,image_num,skip,append,verbose,silent)
        if der_filename is not None and read_der:
            self.read_der_from_lmpdump(descriptors_path,der_filename,image_num,skip,append,verbose,silent)

        if silent is False:
            print('\ntotal images = %d' % self.num_images)
            print('max number of atoms = %d' % self.maxnum_atoms)
            print('number of fingerprints = %d' % self.num_fingerprints)
            print('number of atom types = %d' % self.num_types)
            print('max number of derivative pairs = %d' % self.maxnum_blocks)


        
    def read_fingerprints_from_lmpdump(self, descriptors_path, fp_filename, image_num=None, skip=0, append=False,verbose=False,silent=False):
        """
        Read descriptors(fingerprints), atom_type and volume from the descriptor files created with LAMMPS, and save them into data object.
        """
        if append and len(self.input_dict)==0:
            print(color.RED + 'Warning: data is not appendable, append has been set to False.' + color.END)
            append = False

        if not append:
            self.natoms_list=[]
            self.ntypes_list=[]
            
        files = get_filenames(descriptors_path,fp_filename)[skip:]
        
        if image_num is not None and image_num<len(files):
            nfiles = image_num
        else:
            nfiles = len(files)

        if nfiles==0:
            raise ValueError('Cannot find \'%s\' in \'%s\''%(fp_filename,descriptor_path))

        if silent is False:
            print ("\nReading fingerprints from \'%s\' for total %i files ..."%(fp_filename,nfiles))

        maxnum_atoms = 0 # max number of atoms among all the images
        # screen all the files, get the number of atoms and check data consistance
        for i in range(nfiles):
            f = open(files[i],'r')
            lines = f.readlines()[9:]
            f.close()
            natoms = len(lines)
            if natoms > maxnum_atoms:
                maxnum_atoms = natoms
            self.natoms_list.append(natoms)
            if i==0:
                num_fingerprints = len(lines[0].split())-2
                if append and num_fingerprints != self.num_fingerprints:
                    raise ValueError("The number of fingerprints not equall to the one in existing dataset.")
            else:
                if num_fingerprints != len(lines[0].split())-2:
                    raise ValueError("The number of fingerprints in \'%s\'(%i) is different from \'%s\' (%i)."\
                                     %(os.path.basename(files[0]),(len(lines[0].split())-2),os.path.basename(files[i]),num_fingerprints))
        if append:
            if self.maxnum_atoms > maxnum_atoms:
                maxnum_atoms = self.maxnum_atoms
                padding = False
            else:
                padding = True
                
        fingerprints = np.zeros([nfiles,maxnum_atoms,num_fingerprints],dtype=self.data_type)
        atom_type = np.zeros([nfiles,maxnum_atoms],dtype='int32')
        volume = np.zeros([nfiles,1],dtype=self.data_type)

        # read finterprints, atom_type and volumes
        for i in range(nfiles):
            f = open(files[i],'r')
            lines = f.readlines()
            f.close()

            lx = self.str2float(lines[5].split()[1]) - self.str2float(lines[5].split()[0])
            ly = self.str2float(lines[6].split()[1]) - self.str2float(lines[6].split()[0])
            lz = self.str2float(lines[7].split()[1]) - self.str2float(lines[7].split()[0])            
            volume[i] = [lx*ly*lz]
            
            for j,line in enumerate(lines[9:]): # loop inside each file
                line = line.split()
                atom_type[i][j] = int(line[1])
                fingerprints[i][j] = [self.str2float(x) for x in line[2:]]
            self.ntypes_list.append(max(atom_type[i]))
            
            if verbose and silent is False:
                print("  file-%i: read fingerprints from \'%s\'."%(i+1,os.path.basename(files[i])))
            if int((i+1)%50)==0 and silent is False:
                print ('  so far read %d images ...' % (i+1),flush=True)  

        if not append:
            self.input_dict['fingerprints'] = fingerprints
            self.input_dict['atom_type'] = atom_type
            self.input_dict['volume'] = volume
            self.maxnum_atoms = maxnum_atoms
            self.num_fingerprints = num_fingerprints
        else:
            if padding: # pad existing data
                if silent is False:
                    print('Pad existing dataset.')
                pad_size = maxnum_atoms - self.maxnum_atoms
                self.input_dict['fingerprints'] = \
                    np.pad(self.input_dict['fingerprints'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
                self.input_dict['atom_type'] = \
                    np.pad(self.input_dict['atom_type'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
                self.maxnum_atoms = maxnum_atoms

            self.input_dict['fingerprints'] = np.concatenate((self.input_dict['fingerprints'],fingerprints),axis=0)
            self.input_dict['atom_type'] = np.concatenate((self.input_dict['atom_type'],atom_type),axis=0)
            self.input_dict['volume'] = np.concatenate((self.input_dict['volume'],volume),axis=0)

        #if silent is False:
        #    print('  Finish reading fingerprints from total %i images.' % nfiles,flush=True)
        self.num_images = len(self.input_dict['fingerprints'])
        self.num_types = max(self.ntypes_list)

        if not self.print_inputinfo_together and silent is False:
            print('total images in dataset = %d' % self.num_images,flush=True)
            print('max number of atoms in dataset= %d' % self.maxnum_atoms,flush=True)
            print('number of fingerprints in dataset= %d' % self.num_fingerprints,flush=True)
            print('type of atoms in dataset= %d' % self.num_types)  

        


    def read_der_from_lmpdump(self,descriptor_path, der_filename, image_num=None,skip=0,append=False,verbose=False, silent=False):
        """
        Read derivatives of fingerprints w.r.t. coordinates (dGdr), neibhor_atom_coord, center_atom_id, neighbor_atom_id.        
        """

        if append and len(self.input_dict)==0:
            print(color.RED + 'Warning: data is not appendable, append has been set to False.' + color.END)
            append = False

        if not append:
            self.num_blocks_list=[]

        files = get_filenames(descriptor_path,der_filename)[skip:]

        if image_num is not None and image_num<len(files):
            nfiles = image_num
        else:
            nfiles = len(files)
        if nfiles==0:
            raise ValueError('Cannot find \'%s\' in \'%s\''%(der_filename,descriptor_path))

        if silent is False:
            print("\nReading derivatives from \'%s\' for total %i files (may take a while for large data set) ..."%(der_filename,nfiles))

        
        start_time = time.time()    

        maxnum_blocks = 0 # max number of block in the reading input dataset
        
        # screen all the derivative files, get the number of derivative pairs(blocks) and check data consistance
        for i in range(nfiles):
            in_file = open(files[i], 'r')
            lines = in_file.readlines()[9:]
            nlines = len(lines)
            if nlines%3 != 0:
                raise ValueError('The derivative file \'%s\' has the wrong line number.'%os.path.basename(files[i]))
            nblocks = int(nlines/3)
            if nblocks > maxnum_blocks:
                maxnum_blocks = nblocks
            self.num_blocks_list.append(nblocks)
            if i==0:
                num_fingerprints =  len(lines[0].split())-3
                if append and num_fingerprints != self.num_fingerprints:
                    raise ValueError("The number of fingerprints not equall to the one in existing dataset.")
            else:
                if num_fingerprints != len(lines[0].split())-3:
                    raise ValueError("The number of fingerprints in \'%s\'(%i) is different from \'%s\' (%i)."\
                                     %(os.path.basename(files[0]),(len(lines[0].split())-3),os.path.basename(files[i]),num_fingerprints))
        if append:
            if self.maxnum_blocks > maxnum_blocks:
                maxnum_blocks = self.maxnum_blocks
                padding = False # do not pad existing data
            else:
                padding = True # pad existing data

        dGdr = np.zeros([nfiles,maxnum_blocks,num_fingerprints,3],dtype=self.data_type)
        neighbor_atom_coord = np.zeros([nfiles,maxnum_blocks,3,1],dtype=self.data_type)
        center_atom_id = np.zeros([nfiles,maxnum_blocks],dtype='int32')
        neighbor_atom_id = np.ones([nfiles,maxnum_blocks],dtype='int32')*(-1)

        for i in range(nfiles):
            in_file = open(files[i], 'r')
            lines = in_file.readlines()[9:]
            in_file.close()
            lines = [lines[j].split() for j in range(len(lines))]
            ipair = 0
            for j in range(0,len(lines),3):
                block = lines[j:j+3]
                center_atom_id[i][ipair] = int(block[1][0])-1 # center atom id, -1 to help with idexing
                neighbor_atom_id[i][ipair] = int(block[1][1])-1 # neighbor atom id, -1 to help with idexing
                neighbor_atom_coord[i][ipair] = np.array([[self.str2float(row[2])] for row in block[0:]])
                dGdr[i][ipair] = np.array([[self.str2float(x[k]) for x in block[0:]]for k in range(3,len(block[0]))])
                ipair += 1
            if verbose and silent is False:
                print("  file-%i: read derivatives from \'%s\'."%(i+1,os.path.basename(files[i])))
            if int((i+1)%50)==0 and silent is False:
                print ('  so far read %d images ...' % (i+1),flush=True)
                
        if not append:
            self.input_dict['center_atom_id'] = center_atom_id
            self.input_dict['neighbor_atom_id'] = neighbor_atom_id
            self.input_dict['neighbor_atom_coord'] = neighbor_atom_coord
            self.input_dict['dGdr'] = dGdr
            self.maxnum_blocks = maxnum_blocks
            self.num_fingerprints = num_fingerprints
        else: 
            if padding: # pad existing data
                if silent is False:
                    print ('  \nPadding existing dataset because number of derivative pairs increased in new data.')
                pad_size = maxnum_blocks - self.maxnum_blocks 
                self.input_dict['center_atom_id'] = \
                    np.pad(self.input_dict['center_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
                self.input_dict['neighbor_atom_id'] = \
                    np.pad(self.input_dict['neighbor_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(-1))
                self.input_dict['neighbor_atom_coord'] = \
                    np.pad(self.input_dict['neighbor_atom_coord'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
                self.input_dict['dGdr'] = \
                    np.pad(self.input_dict['dGdr'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
                self.maxnum_blocks = maxnum_blocks
                                
            self.input_dict['center_atom_id'] = np.concatenate((self.input_dict['center_atom_id'],center_atom_id),axis=0)
            self.input_dict['neighbor_atom_id'] = np.concatenate((self.input_dict['neighbor_atom_id'],neighbor_atom_id),axis=0)
            self.input_dict['neighbor_atom_coord'] = np.concatenate((self.input_dict['neighbor_atom_coord'],neighbor_atom_coord),axis=0)
            self.input_dict['dGdr'] = np.concatenate((self.input_dict['dGdr'],dGdr),axis=0)

        #if silent is False:
        #    print('  Finish reading derivatives from total %i images.'% nfiles,flush=True)

        if not self.print_inputinfo_together and silent is False:
            #print('total images in dataset = %d' % len(self.input_dict['center_atom_id']),flush=True)
            print('  max number of derivative pairs = %d' % self.maxnum_blocks,flush=True)
            #print('number of fingerprints = %d' % self.num_fingerprints,flush=True)
        if verbose and silent is False:
            print('  It took %.2f seconds to read the derivatives data.'%(time.time()-start_time),flush=True)

        
        
    def read_outputdata(self, xyzfiles, format='extxyz',image_num=None, skip=0, append=False, verbose=False, silent=False,\
                        read_force=atomdnn.compute_force, read_stress=atomdnn.compute_force, **kwargs): 
        """
        Read outputs(energy, force and stress) from extxyz files

        Args:
            xyzfiles: the name and path to atomic structures, wildcard * is used for files numerically ordered
            format: 'lammp-data','extxyz','vasp' etc. See complete list on https://wiki.fysik.dtu.dk/ase/ase/io/io.html#ase.io.read. 'extxyz' is recommanded.
            read_force(bool): make sure extxyz files have force data if it's True 
            read_stress(bool): make sure extxyz files have stress data if it's True
            image_num: number of images that will be used, if it's None then read all files specified by xyzfile_name
            append(bool): append the reading to previous data object
            verbose(bool): set to True if want to print out the extxyz file names
            kwargs: used to pass optional file styles 
        """

        xyzfile_path = os.path.dirname(os.path.abspath(xyzfiles))
        xyzfile_name = os.path.basename(os.path.abspath(xyzfiles))

        
        if append and len(self.output_dict)==0:
            print(color.RED + 'Warning: data is not appendable, append has been set to False.' + color.END)
            append = False

        if not append:
            self.natoms_in_force=[]
            
        files = get_filenames(xyzfile_path,xyzfile_name)[skip:] # get filenames that match the patten given in xyzfile_name
        
        if image_num is not None and image_num<len(files):
            nfiles = image_num
        else:
            nfiles = len(files)

        if nfiles==0:
            raise ValueError('Cannot find \'%s\' in \'%s\''%(xyzfile_filename,xyzfile_path))

        maxnum_atoms = 0
        for i in range(nfiles):
            patom = read(files[i],format=format,**kwargs)
            natoms = patom.get_global_number_of_atoms()
            self.natoms_in_force.append(natoms)
            if natoms > maxnum_atoms:
                maxnum_atoms = natoms

        if append:
            if self.maxnum_atoms_output > maxnum_atoms:
                maxnum_atoms = self.maxnum_atoms_output
                padding = False
            else:
                padding = True

        pe = np.zeros(nfiles,dtype=self.data_type)
        if read_force:
            force = np.zeros([nfiles,maxnum_atoms,3],dtype=self.data_type)
        if read_stress:
            stress = np.zeros([nfiles,6],dtype=self.data_type)

        if silent is False:    
            print('\nReading outputs from \'%s\' ...' % xyzfile_name, flush=True)    
        for i in range(nfiles):
            patom = read(files[i],format=format,**kwargs)
            pad_size = maxnum_atoms - patom.get_global_number_of_atoms()
            try:
                pe[i] = patom.info['energy']
            except:
                if silent is False:
                    print('  There is no potential energy in \'%s\', check the file and use read_output() funciton to re-read potential energy.'%files[i], flush=True)
                return 
            if read_force:
                try:
                    f = patom.arrays['forces']
                    if pad_size>0:
                        f = np.pad(f, ((0,pad_size),(0,0)),'constant',constant_values=(0))
                    force[i] = f
                except:
                    if silent is False:
                        print('  There is no atomic forces in \'%s\', check the file and use read_output() funciton to re-read forces.'%files[i], flush=True)
                    return
            if read_stress:
                try:
                    stress[i] = patom.get_stress()
                except:
                    if silent is False:
                        print('  There is no stress in \'%s\', check the file and use read_output() funciton to re-read stress.'%files[i], flush=True)
                    return
            
            if verbose and silent is False:
                if read_force and read_stress:
                    print('  file-%i: read output potential energy, forces and stress from \'%s\''\
                          %(i+1, os.path.basename(files[i])))
                elif read_force:
                    print('  file-%i: read output potential energy and forces from \'%s\''\
                          %(i+1, os.path.basename(files[i])))
                elif read_stress:
                    print('  file-%i: read output potential energy and stress from \'%s\''\
                          %(i+1, os.path.basename(files[i])))
                else:
                    print('  file-%i: read output potential energy from \'%s\''\
                          %(i+1, os.path.basename(files[i])))
            if int((i+1)%50)==0 and silent is False:
                print ('  so far read %d images ...' % (i+1),flush=True)

        if not append:
            self.output_dict['pe'] = pe
            if read_force:
                self.output_dict['force'] = force
            if read_stress:
                self.output_dict['stress'] = stress
            self.maxnum_atoms_output = maxnum_atoms
        else:
            if padding and read_force:
                if silent is False:
                    print('Pad existing dataset force since the atom number in new data is increased.')
                pad_size = maxnum_atoms - self.maxnum_atoms_output
                self.output_dict['force'] = \
                    np.pad(self.output_dict['force'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
                
            self.output_dict['pe'] = np.concatenate((self.output_dict['pe'],pe),axis=0)
            if read_force:
                self.output_dict['force'] = np.concatenate((self.output_dict['force'],force),axis=0)
            if read_stress:
                self.output_dict['stress'] = np.concatenate((self.output_dict['stress'],stress),axis=0)

        #if silent is False:
            #print('  Finish reading outputs from total %i images.\n' % nfiles,flush=True)
            #print('total images = %d' % len(self.output_dict['pe']))
           # print('max number of atoms = %d' % self.maxnum_atoms_output)
        #if silent is False:
        #    print('---------------------------------------------------')

                    
    def shuffle(self):
        """
        Shuffle the data.
        """
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


        
    def slice(self, start=None, end=None):   
        """
        Slice the data between image start and image end, and return both the input and output dictionaries. Index starts from 1
        """
        self.check_data()            
        input_dict = slice_dict(self.input_dict, start-1, end-1)
        output_dict = slice_dict(self.output_dict, start-1, end-1)
        return input_dict, output_dict

    def get_input_dict(self,start=None, end=None):
        """
        Return the input dictionaries from image start to image end. Index starts from 1. If end is not privided, return only one dictionary of image start.
        """
        if start==None and end==None:
            start=0
            return slice_dict(self.input_dict, 0, 1)
        if end==None:
            return slice_dict(self.input_dict, start-1, start)
        else:
            return slice_dict(self.input_dict, start-1, end)
        
    def get_output_dict(self,start=None, end=None):
        """
        Return the output dictionaries from image start to image end. Index starts from 1. If end is not privided, return only one dictionary of image start.
        """
        if start==None and end==None:
            start = 0
            return slice_dict(self.output_dict, 0, 1)
        if end==None:
            return slice_dict(self.output_dict, start-1, start)
        else:
            return slice_dict(self.output_dict, start-1, end)
        

    def convert_data_to_tensor(self):
        """
        Convert the input and ouput data to Tensorflow tensors. This can speed up the data manipulation using Tensorflow functions.
        """
        self.check_data()
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

        

    def check_data(self):
        """
        Check consistance of input and output data.
        """
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
                if self.natoms_list != self.natoms_in_force:
                    raise ValueError("The atom numbers in fingerprints files and force files are not consistant.")
            if 'stress' in list(self.output_dict.keys()):
                if self.num_images != len(self.output_dict['stress']):
                    raise ValueError("The image numbers of fingerprints files and stress files are not consistant.")           
        else:
            raise ValueError('No fingerprints in input data.')




    def append(self,apdata,read_force=atomdnn.compute_force,read_stress=atomdnn.compute_force):
        """
        Append one dataset with a second dataset.
        """

        # append fingerprints and output data
        pad_size = np.absolute(self.maxnum_atoms - apdata.maxnum_atoms)
        if self.maxnum_atoms < apdata.maxnum_atoms:
            self.input_dict['fingerprints'] = \
                np.pad(self.input_dict['fingerprints'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
            self.input_dict['atom_type'] = \
                np.pad(self.input_dict['atom_type'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
            if read_force:
                self.output_dict['force'] = \
                    np.pad(self.output_dict['force'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
            self.maxnum_atoms = apdata.maxnum_atoms
        elif self.maxnum_atoms > apdata.maxnum_atoms:
            apdata.input_dict['fingerprints'] = \
                np.pad(apdata.input_dict['fingerprints'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
            apdata.input_dict['atom_type'] = \
                np.pad(apdata.input_dict['atom_type'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
            if read_force:
                apdata.output_dict['force'] = \
                    np.pad(apdata.output_dict['force'], ((0,0),(0,pad_size),(0,0)),'constant',constant_values=(0))
            apdata.maxnum_atoms = self.maxnum_atoms
        self.input_dict['fingerprints'] = np.concatenate((self.input_dict['fingerprints'],apdata.input_dict['fingerprints']),axis=0)
        self.input_dict['atom_type'] = np.concatenate((self.input_dict['atom_type'],apdata.input_dict['atom_type']),axis=0)
        self.input_dict['volume'] = np.concatenate((self.input_dict['volume'],apdata.input_dict['volume']),axis=0)
        self.output_dict['pe'] = np.concatenate((self.output_dict['pe'],apdata.output_dict['pe']),axis=0)
        if read_force:
            self.output_dict['force'] = np.concatenate((self.output_dict['force'],apdata.output_dict['force']),axis=0)
        if read_stress:
            self.output_dict['stress'] = np.concatenate((self.output_dict['stress'],apdata.output_dict['stress']),axis=0)
        
        # append derivatives data
        pad_size = np.absolute(self.maxnum_blocks - apdata.maxnum_blocks)
        if self.maxnum_blocks < apdata.maxnum_blocks:
            self.input_dict['center_atom_id'] = \
                np.pad(self.input_dict['center_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
            self.input_dict['neighbor_atom_id'] = \
                np.pad(self.input_dict['neighbor_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(-1))
            self.input_dict['neighbor_atom_coord'] = \
                np.pad(self.input_dict['neighbor_atom_coord'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
            self.input_dict['dGdr'] = \
                np.pad(self.input_dict['dGdr'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
            self.maxnum_blocks = apdata.maxnum_blocks
        elif self.maxnum_blocks > apdata.maxnum_blocks:
            apdata.input_dict['center_atom_id'] = \
                np.pad(apdata.input_dict['center_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(0))
            apdata.input_dict['neighbor_atom_id'] = \
                np.pad(apdata.input_dict['neighbor_atom_id'], ((0,0),(0,pad_size)),'constant',constant_values=(-1))
            apdata.input_dict['neighbor_atom_coord'] = \
                np.pad(apdata.input_dict['neighbor_atom_coord'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
            apdata.input_dict['dGdr'] = \
                np.pad(apdata.input_dict['dGdr'], ((0,0),(0,pad_size),(0,0),(0,0)),'constant',constant_values=(0))
            apdata.maxnum_blocks = self.maxnum_blocks
            
        self.input_dict['center_atom_id'] = np.concatenate((self.input_dict['center_atom_id'],apdata.input_dict['center_atom_id']),axis=0)
        self.input_dict['neighbor_atom_id'] = np.concatenate((self.input_dict['neighbor_atom_id'],apdata.input_dict['neighbor_atom_id']),axis=0)
        self.input_dict['neighbor_atom_coord'] = np.concatenate((self.input_dict['neighbor_atom_coord'],apdata.input_dict['neighbor_atom_coord']),axis=0)
        self.input_dict['dGdr'] = np.concatenate((self.input_dict['dGdr'],apdata.input_dict['dGdr']),axis=0)
    

#=================================================================================================================
# some functions to operate tensorflow dataset


def get_input_dict(dataset):
    """
    Args:
        dataset: Tensorflow dataset
    
    Returns:
        dictionary: input dictionary, see :class:`~atomdnn.data.Data` for the structure of the dictionary
    """
    for x,y in dataset.batch(len(dataset)):
        return x


    
def get_output_dict(dataset):
    """
    Args:
        dataset: Tensorflow dataset
    
    Returns:
        dictionary: output dictionary, see :class:`~atomdnn.data.Data` for the structure of the dictionary
    """
    for x,y in dataset.batch(len(dataset)):
        return y

    

def get_nfp_from_dataset (dataset):
    """
    Args:
        dataset: Tensorflow dataset
    
    Returns:
        number of fingerprints
    """
    if dataset.element_spec[0]['fingerprints'].shape[1] is not None: 
        return dataset.element_spec[0]['fingerprints'].shape[1]
    else:
        return len(get_input_dict(dataset)['fingerprints'][0][0]) # for tensorflow version below 2.6




def split_dataset(dataset, train_pct, val_pct=None, test_pct=None, shuffle=False,data_size=None):
    """
    Split the tensorflow dataset into training, validation and test.
    
    Args:
        dataset: tensorflow dataset
        train_pct: the percentage of data used for training
        val_pct: the percentage of data used for validation
        test_pct: the percentage of data used for testing
        shuffle(bool): shuffle the dataset
        data_size(int): if None, then use all data in the dataset 

    Returns:
        tensorflow dataset: training, validation and test dataset

    """
    if not data_size:
        data_size = dataset.cardinality().numpy()
    if val_pct is None:
        val_pct = 0
    if val_pct == 0:
        print ("No data are used for validatiaon by default.")
    if test_pct is None:
        test_pct = 0
    if test_pct == 0:
        print ("No data are used for test by default.")
    if train_pct + val_pct + test_pct > 1:
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



def slice_dataset(dataset, start, end):
    """
    Get a slice of the dataset.
    Args:
        dataset: input dataset
        start: starting index
        end: ending index
    Returns:
        tensorflow dataset
    """
    return dataset.skip(start).take(end-start)






