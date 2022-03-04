import os
import time

def create_lmp_input(descriptor=None):

    lmpinfile = open('in.gen_descriptors','w')

    if 'cutoff' in descriptor:
        cutoff = descriptor['cutoff']
    else:
        raise ValueError('descriptor dictionary does not contain cutoff key.')

    G2_parameters = ''
    for i in range(len(descriptor['etaG2'])):
        G2_parameters += str(descriptor['etaG2'][i])+' '

        
    compute_fp_line = ''
    compute_der_line = ''
    
    if 'etaG4' in descriptor:
        G4_parameters = ''
        for i in range(len(descriptor['etaG4'])):
            G4_parameters += str(descriptor['etaG4'][i])+' '
        
            zeta_parameters = ''
            for i in range(len(descriptor['zeta'])):
                zeta_parameters += str(descriptor['zeta'][i])+' '
        
        lambda_parameters = ''
        for i in range(len(descriptor['lambda'])):
            lambda_parameters += str(descriptor['lambda'][i])+' '

        compute_fp_line  = f'compute 1 all fingerprints {cutoff} etaG2 {G2_parameters} etaG4 {G4_parameters} zeta {zeta_parameters} lambda {lambda_parameters} end\n'
        compute_der_line = f'compute 2 all derivatives  {cutoff} etaG2 {G2_parameters} etaG4 {G4_parameters} zeta {zeta_parameters} lambda {lambda_parameters} end\n'

    else:
        compute_fp_line  = f'compute 1 all fingerprints {cutoff} etaG2 1.0 end\n'
        compute_der_line = f'compute 2 all derivatives  {cutoff} etaG2 1.0 end\n'
        
        
    lmpinfile.writelines('clear\n'+
                         'dimension 3\n'+
                         'boundary p p p\n'+
                         'units metal\n'+
                         'atom_style atomic\n'+
                         'variable cutoff equal ' + str(descriptor['cutoff']) + '\n' +
                         'read_data lmpdatafile\n' +
                         'mass * 1.0\n'+
                         'pair_style zero ${cutoff} nocoeff\n'+
                         'pair_coeff * * 1.0 1.0\n'+
                         'neighbor 0.0 bin\n' +
                         compute_fp_line + 
                         compute_der_line +
                         'dump dump_fingerprints all custom 1 ${dp_filename}.${file_id} id type c_1[3*]\n'+
                         'dump_modify dump_fingerprints sort id format float %20.10g\n'+
                         'dump dump_primes all local 1 ${der_filename}.${file_id} c_2[*3] c_2[6*]\n'+
                         'dump_modify dump_primes format float %20.10g\n'+
                         'fix NVE all nve\n'+
                         'run 0')



def create_descriptors(xyzfile_path=None, xyzfile_name=None, descriptors_path=None, lmpexe=None, descriptor=None, descriptor_filename='dump_fp', der_filename='dump_der', image_num=None, fd=None, verbose=False):
    """
    fd => counter. Can be initialized to append to fingerprints.
    """
    
    from ase.io import read, write
    os.makedirs(descriptors_path, exist_ok=True)

    if verbose:
        print('xyzfile_name:', xyzfile_name)
        
    if ('.' in xyzfile_name and xyzfile_name.split('.')[-1] == '*'):
        ext_marker = '.' 
        batch_mode = 1
        xyzfile_name = xyzfile_name[:-2]        
    elif ('*' in xyzfile_name):
        ext_marker = '_'
        batch_mode = 1
        xyzfile_name = xyzfile_name[:-2]
    else:
        batch_mode = 0

    if not batch_mode:
        image_num = 1

    if fd is None:
        fd = 0

    print('xyzfile_path:', xyzfile_path, ' xyzfile_name:', xyzfile_name)
    xyzfile_name = xyzfile_path + '/' + xyzfile_name
    
    cwd = os.getcwd()
    print('Start creating fingerprints ...')
    start_time = time.time()
    while (1):
        if image_num:
            print('if image_num:\n\tfd:',fd,' image_num:', image_num)
            if fd > image_num:
                break
        if batch_mode:
            if verbose:
                print('    batch_mode: on')
            filename = xyzfile_name + ext_marker + str(fd)
        else:
            if verbose:
                print('    batch_mode: off')
            filename = xyzfile_name

        print('    filename::', filename)
        if os.path.isfile(filename):
            patom = read(filename,format='extxyz')
            write(descriptors_path+"/lmpdatafile", patom, format='lammps-data',atom_style='atomic')
            
            os.chdir(descriptors_path)
            
            create_lmp_input(descriptor) 
            
            lmp_cmd = lmpexe + ' -in in.gen_descriptors ' + ' -var file_id ' + str(fd) + ' -var dp_filename ' + descriptor_filename  + ' -var der_filename ' + der_filename 
            
            os.system(lmp_cmd)
            os.remove('lmpdatafile')
            os.chdir(cwd)
            
            fd += 1
            if fd > 0 and int(fd%10)==0:
                print ('  so far finished for %d images ...' % fd,flush=True)
        else:
            break
    print('Finish creating descriptors and their derivatives from total %i images.' % fd,flush=True)
    print('It took %.2f seconds.'%(time.time()-start_time),flush=True)
