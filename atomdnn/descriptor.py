import os
import time
from ase.io import read,write
import atomdnn
import re
import glob

def sorted_alphanumeric(filenames):
    """
    Sort file names in alphanumeric order.
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(filenames, key=alphanum_key)



def get_filenames(file_path,file_name):
    """
    Get the names of a set of files that match the patten given by file_name. 
    Return the filenames having the format of 'string1*string2',
    in which string1 and string2 can be any strings amd * must be an integer.
    """
    true_path = os.path.abspath(os.path.join(file_path,file_name))
    if '*' in file_name:
        name_mask = file_name.split('*')
        if len(name_mask)!=2:
            raise ValueError('The file name \'%s\' can only has one *.'%file_name)
        filenames = glob.glob(true_path) # get the files matching the given pattern
      
        # make sure '*' only replace numbers, take out non-related files
        filenames = [filenames[i] for i in range(len(filenames)) \
                     if os.path.basename(filenames[i]).replace(name_mask[0],'').replace(name_mask[1],'').isdigit()]
        filenames = sorted_alphanumeric(filenames) # sort the files
        if len(filenames)==0:
            raise FileNotFoundError('Cannot find any files \'%s\' in %s'%(file_name,file_path))
        return filenames
    else:
        return [true_path]
    

    
def create_lmp_input(descriptor,descriptors_path):
    """
    Creates a lammps input file for computing descriptors and derivatives
    
    Args:
        descritpor (dictionary): the parameter dictionary of the descriptor 

    """
    infile = os.path.join(descriptors_path,'in.gen_descriptors')
    lmpinfile = open(infile,'w')    
    G2_parameters = ''
    for i in range(len(descriptor['etaG2'])):
        G2_parameters += str(descriptor['etaG2'][i])+' '
        
    G4_parameters = ''
    for i in range(len(descriptor['etaG4'])):
        G4_parameters += str(descriptor['etaG4'][i])+' '
        
    zeta_parameters = ''
    for i in range(len(descriptor['zeta'])):
        zeta_parameters += str(descriptor['zeta'][i])+' '
        
    lambda_parameters = ''
    for i in range(len(descriptor['lambda'])):
        lambda_parameters += str(descriptor['lambda'][i])+' '
        
    compute_fp_line = 'compute 1 all fingerprints ${cutoff} etaG2 ' + G2_parameters \
                       + 'etaG4 ' + G4_parameters \
                       + 'zeta ' + zeta_parameters \
                       + 'lambda ' + lambda_parameters + 'end\n'
    
    compute_der_line = 'compute 2 all derivatives ${cutoff} etaG2 '+ G2_parameters \
                       + 'etaG4 ' + G4_parameters \
                       + 'zeta ' + zeta_parameters \
                       + 'lambda ' + lambda_parameters + 'end\n'

    dump_fp_line =  'dump dump_fingerprints all custom 1 ${fp_filename} id type c_1[*]\n'+ \
                    'dump_modify dump_fingerprints sort id format float %20.10g\n'

    dump_der_line = 'dump dump_primes all local 1 ${der_filename} c_2[*]\n'+ \
                    'dump_modify dump_primes format float %20.10g\n'

    
    if atomdnn.compute_force:
        compute_line = compute_fp_line + compute_der_line
        dump_line = dump_fp_line + dump_der_line
    else:
        compute_line = compute_fp_line
        dump_line = dump_fp_line
        
    lmpinfile.writelines('clear\n'+
                         'dimension 3\n'+
                         'boundary p p p\n'+
                         'units metal\n'+
                         'atom_style atomic\n'+
                         'variable cutoff equal ' + str(descriptor['cutoff']) + '\n' +
                         'read_data ${lmpdatafile}\n' +
                         'mass * 1.0\n'+
                         'log ${logfile}\n'+
                         'pair_style zero ${cutoff} nocoeff\n'+
                         'pair_coeff * * 1.0 1.0\n'+
                         'neighbor 0.0 bin\n' +
                          compute_line + 
                          dump_line +
                         'fix NVE all nve\n'+
                         'run 0')

    

def create_descriptors(lmpexe, elements, xyzfile_path, xyzfile_name, descriptors_path, \
                       descriptor, descriptor_filename='dump_fp.*', der_filename='dump_der.*', \
                       start_file_id=1,image_num=None, skip=0, keep_lmpfiles=False, verbose=False):
    
    """
    Read extxyz files as inputs and create descriptors and their derivatives w.r.t. atom coordinates.

    Args:
       lmpexe: lammps executable 
       xyzfile_path: directory contains a serials of extxyz files of input atomic structures
       xyzfile_name: extxyz filename, wildcard * is used for files numerically ordered
       descriptors_path: a new directory where descirptors will be generated
       descritpor (dictionary): the parameter dictionary of the descriptor
       descriptor_filename: default is 'dump_fp.*', numerically ordered
       der_filename: default is 'dump_der.*', numerically ordered
       start_file_id(int): starting id for descriptor and derivative files
       image_num: number of images that will be used, if it's None then read all files specified by xyzfile_name
       skip(int): skip some images
       keep_lmpfiles(bool): set to True if want to keep the lammps input and datafiles used for creating descriptors 
       verbose(bool): set to True if want to print out the extxyz file names used for creating descriptors

    """
    
    os.makedirs(descriptors_path, exist_ok=True)

    if '*' in descriptor_filename:
        fp_name_mask = descriptor_filename.split('*')
        if len(fp_name_mask)!=2:
            raise ValueError('The descriptor_filename can only has one *.')  
    if '*' in der_filename:
        der_name_mask = der_filename.split('*')
        if len(der_name_mask)!=2:
            raise ValueError('The der_filename can only has one *.')

    xyzfiles = get_filenames(xyzfile_path,xyzfile_name)[skip:]

    if image_num is not None and image_num<len(xyzfiles):
        nfiles = image_num
    else:
        nfiles = len(xyzfiles)

    if nfiles >1 and '*' not in descriptor_filename:
        raise ValueError('Multiple extxyz files found, use * in descriptor_filename.')
    if nfiles >1 and '*' not in der_filename:
        raise ValueError('Multiple extxyz files found, use * in der_filename.')

    # check existing files in descriptor directory 
    if len(os.listdir(descriptors_path))>0:
        while True:
            del_files = input('There are existing files in %s, do you want to first delete the files, y/n? '%descriptors_path)
            if del_files=='y':
                for f in os.listdir(descriptors_path):
                    os.remove(os.path.join(descriptors_path,f))
                break
            elif del_files=='n':
                break
    
    if atomdnn.compute_force:
        print('Start creating fingerprints and derivatives for %i files ...'% nfiles)
    else:
        print('Start creating fingerprints for %i files (no derivatives, set atomdnn.compute_force to True for derivatives) ...'% nfiles)

    #os.chdir(descriptors_path) # switch to descriptor directory
    start_time = time.time()
    for i in range(nfiles):
        patom = read(xyzfiles[i],format='extxyz')
        
        create_lmp_input(descriptor,descriptors_path) # create lammps input file named 'in.gen_descriptors'

        if '*' in descriptor_filename:
            fp_fname = fp_name_mask[0] + str(i+start_file_id) + fp_name_mask[1]
        else:
            fp_fname = descriptor_filename
        if '*' in der_filename:
            der_fname = der_name_mask[0] + str(i+start_file_id) + der_name_mask[1]
        else:
            der_fname = der_filename

        lmpdatafile = 'lmpdatafile.'+str(i+start_file_id)
        lmpdatafile = os.path.join(descriptors_path,lmpdatafile)
        logfile = 'log.'+str(i+start_file_id)
        logfile = os.path.join(descriptors_path,logfile)

        # use specorder to make sure the type of atoms are consistant
        write(lmpdatafile, patom, specorder=elements, format='lammps-data',atom_style='atomic') # create lammps datafile
        
        # lammps run command
        fp_pfname = os.path.join(descriptors_path,fp_fname)
        der_pfname = os.path.join(descriptors_path,der_fname)
        infile = os.path.join(descriptors_path,'in.gen_descriptors')
        lmp_cmd = lmpexe + ' -in ' + infile \
                         + ' -var fp_filename ' + fp_pfname  \
                         + ' -var der_filename ' + der_pfname \
                         + ' -var lmpdatafile ' + lmpdatafile \
                         + ' -var logfile ' + logfile
      
        status = os.system(lmp_cmd) # run lammps
        if status!=0:
            raise RuntimeError('LAMMPS returns error, find error message in jupyter notebook terminal.' +
                               'To check problems, set keep_lmpfiles=True in create_descriptors function,' +
                               'and then check lammps input and data files in descriptor directory.')
        
        if not keep_lmpfiles:
            os.remove(lmpdatafile)
            os.remove(logfile)

        if verbose:
            if atomdnn.compute_force:
                print('  file-%i: read atoms from \'%s\' and created descriptors in \'%s\' and derivatives in \'%s\'' \
                      % (i+1,os.path.basename(xyzfiles[i]),fp_fname,der_fname))
            else:
                print('  file-%i: read atoms from \'%s\' and created descriptors in \'%s\'' \
                      % (i+1,os.path.basename(xyzfiles[i]),fp_fname))
        if i > 0 and int((i+1)%10)==0:
            print ('  so far finished for %d images ...' % (i+1),flush=True)

    if not keep_lmpfiles:
        os.remove(infile)
        os.remove('log.lammps')
        
    print('Finish creating descriptors and derivatives for total %i images.' % nfiles,flush=True)
    print('It took %.2f seconds.'%(time.time()-start_time),flush=True)



def get_num_fingerprints(descriptor,elements):
    """
    Compute the total number of fingerprints.
    
    Args:
        descritpor(dictionary): parameters that defines the descriptors
        elements(list): list of elements

    Returns:
        total number of fingerprints
    """

    ntypes = len(elements)

    if descriptor['name'] == 'acsf':
        ntypes_combinations = ntypes*(ntypes+1)/2;
        n_etaG2 = len(descriptor['etaG2'])
        n_etaG4 = len(descriptor['etaG4'])
        n_zeta = len(descriptor['zeta'])
        n_lambda = len(descriptor['lambda'])
        num_fingerprints = int(n_etaG2*ntypes + n_lambda*n_zeta*n_etaG4*ntypes_combinations + ntypes);

    return num_fingerprints
