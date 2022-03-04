import ase
import ase.io
import numpy as np

def sorted_alphanumeric(data):
    """
    Function to sort a given of file names in alphanumeric order.
    Author: Daniel Ocampo

    Typical example of use:
        sorted_alphanumeric(os.listdir(dataset_folder))
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)



def get_elem_name_by_atomic_num(atomic_numbers):
    ret = []
    if isinstance(atomic_numbers, np.ndarray):
        for entry in atomic_numbers:
            if (entry==1):
                ret.append('H')
            elif (entry==6):
                ret.append('C')
            elif (entry==10):
                ret.append('Ne')
            elif (entry==16):
                ret.append('O')
            elif (entry==42):
                ret.append('Mo')
            elif (entry==52):
                ret.append('Te')
            else:
                raise ValueError('Atomic number is not in the list.')
    
        return np.array(ret)
    else:
        if(atomic_numbers==1):
            return 'H'
        elif (atomic_numbers==6):
            ret.append('C')
        elif (atomic_numbers==10):
            ret.append('Ne')
        elif (atomic_numbers==16):
            ret.append('O')
        elif (atomic_numbers==42):
            ret.append('Mo')
        elif (atomic_numbers==52):
            ret.append('Te')
        else:
            raise ValueError('Atomic number is not in the list.')

    return ret


def read_images(filename, format, path=None, verbose=None, dtype=None):
    """
    Code for reading and visualizing Configuration file format files from N2P2[1]. This code is inspired
    in ASE's lists of atom objects, usually referred to as trajectory files[2].
    Author: Daniel Ocampo

    [1]. https://github.com/CompPhysVienna/n2p2 
         https://compphysvienna.github.io/n2p2/topics/cfg_file.html#cfg-file

    [2]. https://wiki.fysik.dtu.dk/ase/index.html

    returns: calling this function will return a python list filled with Atoms objects. Each instance of 
    the Atoms class is one atomic structure (image) containing the following attributes:
        - lattice (optional)		    - numpy array (floats) with the 9 components of the 3 lattice vectors
        - positions         		    - numpy array (floats) with each atom in the structure has coordinates x, y and z
        - atom types        		    - numpy array (string) with chemical formula of each atom.
        - atom charges (optional)      	    - numpy array (floats) with individual atom charges.
        - atom forces (optional)      	    - numpy array (floats) with fx, fy and fz.
        - img potential energy (optional)   - numpy array (float) with potential energy of atomic structure.
        - img charge (optional)       	    - numpy array (float) with total charge of atomic structure.
    """
    import re
    from os import listdir
    from os.path import join

    if dtype is None:
        dtype = np.float32

    if path is None:
        path='.'
    else:
        filename = join(path, filename)
    
    traj = []
    if format == 'n2p2':
        file = open(filename, 'r')

        while (True):
            next_line = file.readline().split()
            if verbose:
                print(next_line)

                if not next_line:
                    break

            current = re.sub('\n', '', next_line[0])
            current = re.sub(r"^\s+", '', current)
            
            img_lattices = []
            img_positions = []
            img_atom_types = []
            img_atom_charges = []
            img_zeros = []
            img_forces = []
            img_pot_energy = []
            img_charge = []
            if current=='begin':
                while not current=='end':
                    current_line = file.readline().split()
                    current = current_line[0]
                    if current:
                        if verbose:
                            print('    ', current)

                        if current_line[0].startswith('comment'):
                            continue
    
                        elif current_line[0].startswith('lattice'):
                            img_lattices.append(current_line[1:])
    
                        elif current_line[0].startswith('atom'):
                            # 2nd, 3rd and 4th columns are cartesian coords of atom
                            img_positions.append([current_line[1], current_line[2], current_line[3]])
                            # 5th column is atom_type
                            img_atom_types.append(current_line[4])
                            # 6th column is atom_charges
                            img_atom_charges.append(current_line[5])
                            # 7th column is not used at the moment
                            img_zeros.append(current_line[6])
                            # 8, 9 and 10 columns are atom forces
                            img_forces.append(current_line[7:])

                        elif current_line[0].startswith('energy'):
                            img_pot_energy.append(current_line[1])

                        elif current_line[0].startswith('charge'):
                            img_charge.append(current_line[1])

                        else:
                            continue

                # image = Atoms(img_lattices, img_positions, img_atom_types, img_atom_charges, img_zeros, img_forces, img_pot_energy, img_charge, dtype=dtype)
                # lattice=None, positions=None, atom_types=None, atom_charges=None, zeros=None,  forces=None, potential_energy=None, charge=None, dtype=None
                image   = Atoms( lattice          = img_lattices,
                                 positions        = img_positions,
                                 atom_types       = img_atom_types,
                                 atom_charges     = img_atom_charges,
                                 zeros            = img_zeros,
                                 forces           = img_forces,
                                 potential_energy = img_pot_energy,
                                 charge           = img_chage,
                                 dtype            = dtype)
                
                traj.append(image)
                continue
    elif(format=='xyz' or format=='extxyz'):
        import glob

        if '*' in filename:
            for fname in sorted_alphanumeric(glob.glob(filename)):
                ase_img = ase.io.read(fname, index=':', format='extxyz', parallel=True)[0]

                forces  = None
                try:
                    forces = ase_img.get_forces()
                except:
                    pass
                
                pot_eng = None
                try:
                    pot_eng = ase_img.get_potential_energy()
                except:
                    pass

                print('atomic numbers:', ase_img.get_atomic_numbers())

                our_img = Atoms(
                              lattice          = ase_img.get_cell().tolist(),
                              positions        = ase_img.get_positions().tolist(),
                              atom_types       = ase_img.get_atomic_numbers(),
                              forces           = forces,
                              potential_energy = pot_eng,
                              dtype            = dtype
                          )
                traj.append(our_img)
                
        else:
            traj = ase.io.read(filename, index=':', format='extxyz', parallel=True)

            for fname in sorted_alphanumeric(glob.glob(filename)):
                ase_img = ase.io.read(fname, index=':', format='extxyz', parallel=True)[0]

                forces  = None
                try:
                    forces = ase_img.get_forces()
                except:
                    pass

                pot_eng = None
                try:
                    pot_eng = ase_img.get_potential_energy()
                except:
                    pass

                our_img = Atoms(
                              lattice           = ase_img.get_cell().tolist(),
                              positions         = ase_img.get_positions().tolist(),
                              atom_types        = get_elem_name_by_atomic_num(ase_img.get_atomic_numbers()),
                              forces            = forces,
                              potential_energy  = pot_eng,
                              dtype             = dtype
                          )
                traj.append(our_img)

            
    elif(format=='lammps-dump-text'):
        
        ase_traj = ase.io.read(filename, index=':', format='lammps-dump-text', parallel=True)

        for ase_img in ase_traj:
            forces  = None
            try:
                forces = ase_img.get_forces()
            except:
                pass
            
            pot_eng = None
            try:
                pot_eng = ase_img.get_potential_energy()
            except:
                pass
            
            our_img = Atoms(
                            lattice           = ase_img.get_cell().tolist(),
                            positions         = ase_img.get_positions().tolist(),
		            atom_types        = ase_img.get_atomic_numbers(),
                            forces            = forces,
                            potential_energy  = pot_eng,
                            dtype             = dtype
                      )
            traj.append(our_img)
        
    else:
        raise ValueError('Format not recognized for read_images().')

    
    return traj


class Atoms():
    def __init__(self, lattice=None, positions=None, atom_types=None, atom_charges=None, zeros=None,  forces=None, potential_energy=None, charge=None, dtype=None):
        if dtype is None:
            self.dtype = np.float32
        else:
            self.dtype = dtype
        
        self.lattice = np.array(lattice, dtype=self.dtype)
        self.positions = np.array(positions, dtype=self.dtype)
        self.atom_types = np.array(atom_types, dtype=np.str)
        if not atom_charges is None:
            self.atom_charges = np.array(atom_charges, dtype=self.dtype)

        if not zeros is None:
            self.zeros = np.array(zeros, dtype=self.dtype)

        if not forces is None:
            self.forces = np.array(forces, dtype=self.dtype)

        if not potential_energy is None:
            self.potential_energy = np.array(potential_energy, dtype=self.dtype)

        if not charge is None:
            self.charge = np.array(charge, dtype=self.dtype)


    # def read_potential_energies(self, filename):
    #     """
    #     Read potential energies and stresses from external file. Organization of columns is as follows:
    #     # | MD step |         pe            |    pxx pyy pzz pxy pxz pyz    |
    #     # |    i    | atom potential energy | stress components (symmetric) |
    #     """
    #     file = open(filename, 'r')
    #     
    #     potential_energies = []
    #     stresses = []
    #     while (True):
    #         next_line = file.readline().split()
    #         if not next_line:
    #             break
    #             
    #         if next_line[0].startswith('#'):
    #             continue
    #         
    #         # mdstep? pe pxx pyy pzz pxy pxz pyz
    #         potential_energies.append(next_line[1])
    #         stresses.append(next_line[2:])
    #         
    #     file.close()
    #     
    #     self.potential_energies = np.array(potential_energies, dtype=np.float32)
    #     self.stresses = np.array(stresses, dtype=np.float32)
        
            
    def set_potential_energy(self, potential_energy):
        self.potential_energy = potential_energy

    def set_stresses(self, stress_components):
        """
        Set symmetric stress components
        `stress components`: list of len==6
        """
        self.stresses = np.array(stress_components, dtype=self.dtype)

        
        
        
    def write_to_file(self, filename=None, path=None, format=None, elements=None, verbose=None):
        """
        elements keyword: (i.e. elements=['H','O'] for the case of water)
            when converting `lammps-data` file to `extxyz`, there is no information about element type. Therefore, users
            can specify elements to put in the element type column
            
        """
        
        from os import mkdir
        from os.path import join,isdir
        
        if path is not None:
            if not isdir(path):
                mkdir(path)
            filename = join(path, filename)

        if format is None:
            raise ValueError('`format` argument must be specified.')

        if format=='n2p2' or format=='N2P2':
            outfile = open(filename, 'a+')

            outfile.write('begin\n')

            for i in self.lattice:
                outfile.write(f'lattice  {i[0]:9.6f} {i[1]:9.6f} {i[2]:9.6f}\n')

            # initialize variables that could be empty
            if not hasattr(self, 'atom_charges'):
                self.atom_charges = np.zeros((self.positions.shape[0],), dtype=self.dtype)

            if not hasattr(self, 'zeros'):
                self.zeros = np.zeros_like(self.atom_charges)

            if not hasattr(self, 'charge'):
                self.charge = np.array([0.])
            
                
            for pos,atyp,achrg,zrs,forc in zip(self.positions, self.atom_types, self.atom_charges, self.zeros, self.forces):
                atypes = get_elem_name_by_atomic_num(int(atyp))
                outfile.write(f'atom {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f}   {atypes[0]} {achrg:12.9f} {zrs:12.9f}  {forc[0]:12.9f} {forc[1]:12.9f} {forc[2]:12.9f}\n')

            outfile.write(f'energy {self.potential_energy:14.8f}\n')
            outfile.write(f'charge {self.charge[0]:14.8f}\n')
            outfile.write(f'end\n')
            outfile.close()



        elif(format=='xyz' or format=='extxyz'):
            if not filename.endswith('.xyz'):
                filename = filename + '.xyz'


            outfile = open(filename, 'w')

            num_atoms = len(self.positions)
            outfile.write(str(num_atoms)+'\n')

            properties_line = ''
            if not self.lattice.size==0:
                properties_line += 'Lattice="'
                for latt_line in self.lattice:
                    properties_line += f'{latt_line[0]} {latt_line[1]} {latt_line[2]} ' 
                properties_line += '" '

            if hasattr(self, 'stresses') and not self.stresses.size==0:
                # voigt representation of symmetric matrix:  pxx pyy pzz pxy pxz pyz
                # ASE requires 9-long vector. Still has to be symmetric. i.e. [[pxx,pxy,pxz],[pxy,pyy,pyz],[pxz,pyz,pzz]]
                properties_line += f'stress="{self.stresses[0]} {self.stresses[3]} {self.stresses[4]} {self.stresses[3]} {self.stresses[1]} {self.stresses[5]} {self.stresses[4]} {self.stresses[5]} {self.stresses[2]}" '
                
            properties_line += 'Properties=species:S:1:pos:R:3'

            if hasattr(self, 'forces') and not self.forces.size==0:
                properties_line += ':forces:R:3 '

            if hasattr(self, 'potential_energy') and self.potential_energy:
                properties_line += f'energy={self.potential_energy}'
                
            properties_line += '\n'
            
            atom_lines = ''
            if hasattr(self, 'forces'):
                for pos,atyp,forc in zip(self.positions, self.atom_types, self.forces):
                    if not elements:
                        raise ValueError('Writing to `xyz` file requires elements argument.')

                    elem_type = elements[int(atyp)-1]
                    atom_lines += f'{elem_type} {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f} {forc[0]:12.9f} {forc[1]:12.9f} {forc[2]:12.9f}\n'
            else:
                for pos,atyp in zip(self.positions, self.atom_types):
                    atom_lines += f'{atyp} {pos[0]:12.9f} {pos[1]:12.9f} {pos[2]:12.9f}\n'
            

            outfile.write(properties_line)
            outfile.write(atom_lines)
            outfile.close()
            

        elif(format=='lammps-data'):
            import warnings
            # raise ValueError('Exporting to lammps-data format is still not fully implemented.')

            if not filename.endswith('.data'):
                filename = filename + '.data'

            outfile = open(filename, 'w')

            # print('# LAMMPS data file written by GraoGroup\'s AtomDNN 1.0 - Atoms class')
            outfile.write('# LAMMPS data file written by GraoGroup\'s AtomDNN 1.0 - Atoms class\n')
            

            num_atoms = len(self.positions)
            # print(str(num_atoms)+' atoms')
            outfile.write(str(num_atoms)+' atoms\n')

            natom_types = np.unique(self.atom_types).size
            # print(str(natom_types)+ ' atom types')
            outfile.write(str(natom_types)+ ' atom types\n')

            ## lattice
            if not self.lattice.size==0:
                outfile.write(f'0.0 {self.lattice[0,0]:12.7f}   xlo xhi\n')
                outfile.write(f'0.0 {self.lattice[1,1]:12.7f}   ylo yhi\n')
                outfile.write(f'0.0 {self.lattice[2,2]:12.7f}   zlo zhi\n')

                if self.lattice[0,1] or self.lattice[0,2] or self.lattice[1,2]:
                    outfile.write(f'{self.lattice[0,1]:12.7f}  {self.lattice[0,2]:12.7f}  {self.lattice[1,2]:12.7f} xy xz yz\n')
                    
            ## Atoms
            outfile.write('\nAtoms\n')

            if hasattr(self, 'forces'):
                if not self.forces.size==0:
                    warnings.warn('Forces were ignored when writting lammps-data files.')


            for i,(pos,atype) in enumerate(zip(self.positions, self.atom_types)):
                outfile.write(f'{i}   {atype} {pos[0]:12.7f} {pos[1]:12.7f} {pos[2]:12.7f}\n')

            outfile.write('\n')
            outfile.close()

            if verbose:
                print(f'lammps-data file has been successfully written to {filename}')
            
            #latt_lines = ''
            #if not self.lattice.size==0:
                #for latt_line in self.lattice:
                    #latt_lines += f'{latt_line[0]} {latt_line[1]} {latt_line[2]} '
		    # CONVERT 9 PARAMETERS LATTICE SPECIFICATION TO 6 PARAMETERS LATTICE
            
           #0.0 19.1739553015 xlo xhi
           #0.0 10.3094040029 ylo yhi
           #0.0 29.999991153 zlo zhi
            
            
        else:
            raise ValueError('Formats other than N2P2 are not developed for write_to_file() method.')

