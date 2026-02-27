from ase import Atoms, Atom
from ase.calculators.vasp import Vasp
from ase.spacegroup import crystal
from ase.visualize import view
from ase.io import read, write
import numpy as np
import os
import time
from math import *
from functools import reduce
### Function for getting energy, volume and number of atoms from OUTCAR file of VASP
### taking [eV] and [A] units, returning [J] and [A]
### for QE, taking [Ry] and [Bohr] units, and returning [J] and [A]
def get_param(file_name,infile='VASP'):
    with open(file_name) as outf:
        data=outf.readlines()
    if infile=='VASP':
        for i in data:
            if 'free  energy   TOTEN' in i:
                energy=float(i.split()[-2]) # unti is eV
                #energy = energy * 1.602176565e-19 # unit is J
                #print energy
            elif 'energy  without entropy' in i:
                energy_wTS=float(i.split()[3]) # unti is eV
            elif 'volume of cell' in i:
                volume = float(i.split()[-1]) # unti Angstrom 
                #volume = volume*0.529177208**3 # unit is Angstrom
            elif 'NIONS' in i:
                atoms_n= float(i.split()[-1])
                #print atoms_n
    else:
        print("please set VASP or QE")
    return energy, energy_wTS, volume, atoms_n

### Function for writing vasp files from input structure and calculator ###
def set_vasp(structure, calc):
    structure.set_calculator(calc)
    calc.initialize(structure)
    calc.write_incar(structure)
    calc.write_kpoints()
    calc.write_potcar()
    write('POSCAR',structure, format='vasp', vasp5=True)
###
### Function for copieng potcars from specific directory, where potcars are written in format POTCAR_Li
def pot_write(struc_name,pot_dir, pot_name='POTCAR',write_dir='./'):
    struc=read(struc_name)
    symb_l=struc.get_chemical_symbols()
    symb=[]
    for i in symb_l:
        if i not in symb:
            symb.append(i)
    list_potcars=os.listdir(pot_dir)
    for i in list_potcars:
        if 'POTCAR' not in i:
            list_potcars.remove(i)
    with open('POTCAR','w') as pot:
        pass
    for i in symb:
        pot_i='POTCAR_'+i
        with open(pot_dir+pot_i,'r')as pot_j:
            data_j=pot_j.read()
        with open(write_dir+pot_name,'a') as pot_w:
            pot_w.write(data_j)

### Function for copying specific POTCAR to the current directiry
def potcar_cp(potcar_file,cp_tag=1):
    if cp_tag==1:
        cp_command='cp '+potcar_file+' ./POTCAR'
    os.system(cp_command)

# In[2]:
### Function for automatic K point generation for VASP POSCAR
def k_mesh(Rk,poscar_file):
    import math
    struc_file=poscar_file
    struc=read(struc_file)
    file = open(struc_file,'r')
    poscar = file.readlines()

    scale = float(poscar[1])

    line_a1 = poscar[2]
    line_a2 = poscar[3]
    line_a3 = poscar[4]

    a1 = scale*np.fromstring(line_a1, dtype=float, sep=' ')
    a2 = scale*np.fromstring(line_a2, dtype=float, sep=' ')
    a3 = scale*np.fromstring(line_a3, dtype=float, sep=' ')

    V= np.dot(a1,np.cross(a2,a3))

    pi = math.pi

    b1 = 2*pi*np.cross(a2,a3)/V
    b2 = 2*pi*np.cross(a3,a1)/V
    b3 = 2*pi*np.cross(a1,a2)/V

    mb = []
    mb.append(float(np.linalg.norm(b1)))
    mb.append(float(np.linalg.norm(b2)))
    mb.append(float(np.linalg.norm(b3)))

    N1 = int(max(1,Rk*mb[0]+0.5))
    N2 = int(max(1,Rk*mb[1]+0.5))
    N3 = int(max(1,Rk*mb[2]+0.5))
    return N1,N2,N3

### Function for calculating Energy gap from OUTCAR ###
def gap(file_name):
    with open(file_name) as outf:
        data=outf.readlines()
    # getting Number of Bands, electrons and Kpoints
    for i_lines in data:
        if 'NELECT' in i_lines:
            nelect=float(i_lines.split()[2])
        elif 'NBANDS=' in i_lines:
            nbands=int(i_lines.split()[-1])
            nkpts=int(i_lines.split()[3])
    # kpoint_id has  size of number kpoits, and each value correspond to place of Eingelvalues in OUTCAR
    kpoint_id=-1*np.ones(nkpts)
    # getting index of each kpoint, below which are eingenvalues
    k_i=0 #temporary counter
    # if number of K >1000 in outcar, k is written *** instead of 1000 (or above)numbers
    for i in np.arange(0,len(data)):
        if 'band No' in data[i]:
            k_id=(data[i-1].split()[1])
            if k_id=='1':
                k_id=1 
                k_i=1
            else:
                k_i=k_i+1
                k_id=k_i
            kpoint_id[k_id-1]=int(i)
    # array of homo and luma, first coloum is band number, second energy, third occupation
    homo=np.zeros((1,3))  
    lumo=np.zeros((1,3))
    band_occ=-1*np.ones((nbands,1)) # creating array of occupations
    band_energy=-1*np.ones((nbands,1)) # creating array for energy
    # Collecting HOMO and LUMO for each k-point
    for i in range(nkpts):
        kp_id=(int(kpoint_id[i])) # making integer
        band_e=np.loadtxt(data[kp_id+1:kp_id+nbands+1]) # loading data as numpy array
        band_occ_conc=band_e[:,2] # array of occupation level for each band
        band_energy_conc=band_e[:,1]  # array of energy for each band
        band_occ=np.concatenate((band_occ,band_occ_conc[:,np.newaxis]),axis=1) # total occupations array
        band_energy=np.concatenate((band_energy,band_energy_conc[:,np.newaxis]),axis=1) # total energy array
    # deleting first row of zeros number   
    band_occ=band_occ[:,1:]
    band_energy=band_energy[:,1:]
    ######
    ### main part ###
    occ_i=0 # indicator of occupied band
    unocc_i=0 # indicator vor fully unoccupied band
    for i in np.arange(nbands):
        band_o_i =band_occ[i]        
        if np.any(band_o_i>0) and np.any(band_o_i<=0): # looking if the band is half occupied
            E_gap=0
            occ_i=0
        elif np.all(band_o_i>0): # looking if the band is fully occupied
            occ_i=1
        elif np.all(band_o_i<=0) and occ_i==1: # looking if the band is unoccupied
            LUMO = np.min(band_energy[i])
            HOMO = np.max(band_energy[i-1])
            E_gap=LUMO-HOMO # calculating band gap for semiconductors and insulators
            occ_i=0
            #print HOMO
            #print LUMO
    return E_gap

### Function for finding maximum devidable from n and m, if both are > 0
def max_devid(n,m):
    if m==1 or n ==1:
        m_final=int(m)
        n_final=int(n)
    elif m%n==0:
        m_final=int(m/n)
        n_final=int(n/n)
    elif n%m==0:
        m_final=int(m/m)
        n_final=int(n/m)
    else:
        # find list of divisibles of m & n
        m_divs=filter(lambda i: m%i== 0, range(1,m//2+1))
        n_divs=filter(lambda i: n%i== 0, range(1,n//2+1))
        # finding maximum if matched divisibles
        div_max=max(list(set(m_divs).intersection(n_divs)))
        m_final=int(m/div_max)
        n_final=int(n/div_max)
        #print m, n, div_max
        
        
    return n_final,m_final

### Function for getting Elastic parameters form VASP OUTCAR ###
def get_C(filename):
    C=np.zeros((6,6))
    with open(filename,'r') as outcar:
        infile=outcar.readlines()
    for i in range(len(infile)):
        if "TOTAL ELASTIC MODULI" in infile[i]:
            indx = i
    for j in range(0,6):
        j_line=infile[indx+j+3].split()
        j_float=[float(k) for k in j_line[1:]]
        C[j]=j_float
    return C*0.1 # coverting to GPa


# calculating parts of free energy:zrero point energy & vibrational energy
def FreeE(file_name, atoms_n,temp=1000,in_type="VASP"):
    with open(file_name) as pdos:
        data_dos=pdos.readlines()
    # Constants #
    if in_type=="VASP":
        unit_op=10**12
    elif in_type=="QE":
        unit_op=2.998*10**10
    h=6.62607*10**-34    # Plank constant in J*s, don't use h-bar, because its unit is J*s/rad
    k=1.3806488*10**-23; # Boltzman constant in J/K
    ###
    omega=[]
    PDOS =[]
    modes_n=0
    for i in data_dos[1:]:
        if (float(i.split()[0])) > 0:  # use only positive modes, some negative modes exists
            omega.append(float(i.split()[0])*unit_op) # changing unic [cm^-1] to s^-1
            PDOS.append(float(i.split()[1])/(atoms_n*unit_op)) #1/cm-1/cell to 1/s^-1/atom
        else:
            modes_n=modes_n+1
    if modes_n>0:
        print('Warning: there are '+str(modes_n)+' negative modes')
    domega = omega[2]-omega[1]
    sum_Uzero=0
    sum_Fvib=0
    for i in range(len(omega)):
        sum_Uzero=sum_Uzero+0.5*h*PDOS[i]*omega[i]*domega
        sum_Fvib=sum_Fvib+k*temp*log(1-exp(-h*omega[i]/(k*temp)))*PDOS[i]*domega
    Fvib_ev=sum_Fvib*6.242e+18 # unit is eV
    Uzero_ev=sum_Uzero*6.242e+18 # unit is eV
    return Fvib_ev, Uzero_ev

# Function for reading poscar, and getting coposition, returning A, m, B, n in one list
def read_poscar(file_name):
    with open(file_name) as outf:
        data=outf.readlines()
    a_type=len(data[5].split())
    if a_type==2:
        A=data[5].split()[0]
        B=data[5].split()[1]
        m=int(data[6].split()[0])
        n=int(data[6].split()[1])
    elif a_type==1:
        A=data[5].split()[0]
        B=data[5].split()[0]
        m=int(data[6].split()[0])
        n=int(data[6].split()[0])
    else:
        print("Script works only for binary or unary materials")
    if m==1 or n ==1:
        m_final=int(m)
        n_final=int(n)
    elif m%n==0:
        m_final=int(m/n)
        n_final=int(n/n)
    elif n%m==0:
        m_final=int(m/m)
        n_final=int(n/m)
    else:
        # find list of divisibles of m & n
        m_divs=filter(lambda i: m%i== 0, range(1,m//2+1))
        n_divs=filter(lambda i: n%i== 0, range(1,n//2+1))
        # finding maximum if mathed divisibles
        div_max=max(list(set(m_divs).intersection(n_divs)))
        m_final=int(m/div_max)
        n_final=int(n/div_max)
        #print m, n, div_max


    return A,m_final,B,n_final

### Function for getting density of states from DOSCAR
def get_density(file_name, fermi):
    dos=np.loadtxt(file_name,skiprows=6)
    E=dos[:,0]                         # getting Energies for dos
    x1=E[ E- fermi<0][-1]            # getting the closest Energy below E-fermi
    y1=dos[len(E[ E- fermi<0])-1,1]  # getting the closest dos below DOS(E-fermi)
    x2=E[ E- fermi>0][0]             # getting the closest Energy above E-fermi
    y2=dos[len(E[ E- fermi<0]),1]    # getting the closest dos above DOS(E-fermi)
    dos_fermi= ((y2-y1)/(x2-x1))*(fermi-x1)+y1 # getting DOS at fermi level
    return dos_fermi


# Functions for claculating factorials and Combinations
def fact(num):
    if num>0:
        return  reduce((lambda x, y: x * y), range(1,num+1))
    else:
        return 1
C=lambda n,k: fact(n)/(fact(k)*fact(n-k))

### Function for getting E-energy, V-volume per atom, N-number of atoms and E-fermi  from OUTCAR file of VASP
### taking [eV] and [A] units, returning [J] and [A]

def get_param_fermi(file_name,infile='VASP'):
    with open(file_name) as outf:
        data=outf.readlines()
    E_fermi=np.array([])
    if infile=='VASP':
        for i in data:
            if 'free  energy   TOTEN' in i:
                energy=float(i.split()[-2]) # unti is eV
                #energy = energy * 1.602176565e-19 # unit is J
                #print energy
            elif 'volume of cell' in i:
                volume = float(i.split()[-1]) # unti Angstrom
            elif 'NIONS' in i:
                atoms_n= float(i.split()[-1])
            elif 'NELECT' in i:
                nelect=float(i.split()[2])
                #print atoms_n
            elif 'E-fermi' in i:
                E_fermi=np.append(E_fermi,float(i.split()[2]))
    else:
        print("please set VASP")
    return energy/atoms_n, volume/atoms_n, atoms_n,nelect, E_fermi[-1]

### Electronegativity using the Allen scale
X_list=     {'1' : 2.300    , 'H' : 2.300,
             '2': 4.160     , 'He' : 4.160,
             '3': 0.912     , 'Li': 0.912,
             '4': 1.576     , 'Be': 1.576,
             '5' : 2.051        , 'B' :         2.051,
             '6' : 2.544    , 'C' :     2.544,
             '7' :      3.066   , 'N' :         3.066,
             '8' :      3.610   , 'O' :         3.610,
             '9' :      4.193   , 'F' :         4.193,
             '10':      4.787   , 'Ne':         4.787,
             '11':      0.869   , 'Na':         0.869,
             '12':      1.293   , 'Mg':         1.293,
             '13': 1.613        , 'Al':         1.613,
             '14': 1.916        , 'Si':         1.916,
             '15': 2.253        , 'P' :         2.253,
             '16': 2.589        , 'S' :         2.589,
             '17': 2.869        , 'Cl':         2.869,
             '18': 3.242        , 'Ar':         3.242,
             '19': 0.734        , 'K' :         0.734,
             '20': 1.034        , 'Ca':         1.034,
             '21': 1.19     , 'Sc':     1.19,
             '22': 1.38     , 'Ti':     1.38,
             '23': 1.53     , 'V' :     1.53,
             '24': 1.65     , 'Cr':     1.65,
             '25': 1.75     , 'Mn':     1.75,
             '26': 1.80     , 'Fe':     1.80,
             '27': 1.84     , 'Co':     1.84,
             '28': 1.88     , 'Ni':     1.88,
             '29': 1.85     , 'Cu':     1.85,
             '30': 1.59     , 'Zn':     1.59,
             '31': 1.756        , 'Ga':         1.756,
             '32': 1.994        , 'Ge':         1.994,
             '33': 2.211        , 'As':         2.211,
             '34': 2.424        , 'Se':         2.424,
             '35': 2.685        , 'Br':         2.685,
             '36': 2.966        , 'Kr':         2.966,
             '37': 0.706        , 'Rb':         0.706,
             '38': 0.963        , 'Sr':         0.963,
             '39': 1.12     , 'Y' :     1.12,
             '40': 1.32     , 'Zr':     1.32,
             '41': 1.41     , 'Nb':     1.41,
             '42': 1.47     , 'Mo':     1.47,
             '43': 1.51     , 'Tc':     1.51,
             '44': 1.54     , 'Ru':     1.54,
             '45': 1.56     , 'Rh':     1.56,
             '46': 1.58     , 'Pd':     1.58,
             '47': 1.87     , 'Ag':     1.87,
             '48': 1.52     , 'Cd':     1.52,
             '49': 1.656        , 'In':         1.656,
             '50': 1.824        , 'Sn':         1.824,
             '51': 1.984        , 'Sb':         1.984,
             '52': 2.158        , 'Te':         2.158,
             '53': 2.359        , 'I' :         2.359,
             '54': 2.582        , 'Xe':         2.582,
             '55': 0.659        , 'Cs':         0.659,
             '56': 0.881        , 'Ba':         0.881,
             '71': 1.09     , 'Lu':     1.09,
             '72': 1.16     , 'Hf':     1.16,
             '73': 1.34     , 'Ta':     1.34,
             '74': 1.47     , 'W' :     1.47,
             '75': 1.60     , 'Re':     1.60,
             '76': 1.65     , 'Os':     1.65,
             '77': 1.68     , 'Ir':     1.68,
             '78': 1.72     , 'Pt':     1.72,
             '79': 1.92     , 'Au':     1.92,
             '80': 1.76     , 'Hg':     1.76,
             '81': 1.789        , 'Tl':         1.789,
             '82': 1.854        , 'Pb':         1.854,
             '83': 2.01     , 'Bi':     2.01,
             '84': 2.19     , 'Po':     2.19,
             '85': 2.39     , 'At':     2.39,
             '86': 2.60     , 'Rn':     2.60,
             '87': 0.67     , 'Fr':     0.67,
             '88': 0.89     , 'Ra':     0.89,
             }

### Class for Tags ###
class Tag():
    def __init__(self,tag_name,dir_path='./',file_name='Tag_File.txt'):
        self.tag_name=tag_name
        #print(len(self.tag_name.split()))
        assert len(self.tag_name.split())==1, 'Tag name or status should be one strig whithout any enter and space sign  '
        self.dir_path=dir_path
        self.file_name=file_name
    def set_status(self,status):
        assert len(status.split())==1, 'Tag name or status should be one strig whithout any enter and space sign  '
        self.status=status
        data=[]
        if os.path.isfile(self.dir_path+self.file_name):
            with open(self.dir_path+self.file_name,'r') as tag_file:
                data_1=tag_file.read()
                data=data_1.split('\n')
         #       print(data,'file')
        j=0
        for i in range(len(data)):
            i_list=data[i].split()
           # print(i_list,len(i_list))
            if len(i_list)==2:
            #    print('tag line')
                if i_list[0]==self.tag_name:
                    i_list[1]=status
                    data[i]=' '.join(i_list)
                    j+=1
        if j==0:
            data.append(self.tag_name+' '+status)
        data='\n'.join(data)
        with open(self.dir_path+self.file_name,'w') as tag_file:
            data=tag_file.write(data)

    def get_status(self):
        if os.path.isfile(self.dir_path+self.file_name):
            with open(self.dir_path+self.file_name,'r') as tag_file:
                data=tag_file.readlines()
            j=0
            current_status=''
            for i in range(len(data)):
                i_list=data[i].split()
                if len(i_list)==2:
                    if i_list[0]==self.tag_name:
                        current_status=i_list[1]
                        j+=1
            
            if j>0:
                return current_status
            else:
                return 'No STATUS for '+self.tag_name
        else:
            return 'No TAG file'
