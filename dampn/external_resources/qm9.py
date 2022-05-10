"""Download and format QM9"""
import os
import shutil
import tempfile
import urllib.request
import tarfile

import ase.io
import pandas as pd
import numpy as np
import csv

from joblib import Parallel, delayed

import logging
logger = logging.getLogger(__name__)

def qm9_file_to_atoms(filepath):
    '''
    Extract the information from file at self.filepath and store in self.data as pandas dataframe.
    
    Parameters
    ----------
    filepath : str
        path to qm9 file
    '''
    ##### FUNCTION IS ARCHAIC from pre object oriented transformation
    path = filepath
    molecule_file = filepath[-10:-4]

    ## need number of atoms to know where to pull values
    row_count = sum(1 for row in csv.reader(open(path)))
    na = row_count-5

    ## look frequencies
    freqs = pd.read_csv(path,sep=' |\t',engine='python',skiprows=row_count-3,nrows=1,header=None)
    sz = freqs.shape[1]
    is_linear = np.nan
    if 3*na - 5 == sz:
        is_linear = False
    elif 3*na - 6 == sz:
        is_linear = True
    else:
        is_linear = np.nan

    ## Get the molecular properties
    stats = pd.read_csv(path,sep=' |\t',engine='python',skiprows=1,nrows=1,header=None)
    stats = stats.loc[:,2:]
    stats.columns = ['rc_A','rc_B','rc_C','mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']

    ## get the xyz coordinates for each atom, and Mulliken charge
    xyz = pd.read_csv(path, sep='\t',engine='python', skiprows=3, skipfooter=3, names=['Atom', 'x', 'y', 'z', 'M'])
    m = xyz['M']
    xyz.drop('M', axis=1, inplace=True)
    stats['mulliken_charges'] = m
    stats['mulliken_min'] = m.min()
    stats['mulliken_max'] = m.max()
    stats['mulliken_mean'] = m.mean()

    ## grab the frequencies
    f = pd.read_csv(path,sep='\t',engine='python', skiprows=2+na, skipfooter=2,header=None).to_numpy()
    stats['frequencies'] = [f]

    ## grab smiles, InChI
    for i, line in enumerate(open(path).readlines()):
        if i == na+3:
            smline = line
        if i == na+4:
            InChEline = line

    sm_gdb, sm_opt = smline.split()
    InChE_gdb, InChE_opt = [i[6:] for i in InChEline.split()]

    stats['SMILES_GDB17'], stats['SMILES_Optimized'] = smline.split()
    stats['InChE_GDB17'], stats['InChE_Optimized'] = [i[6:] for i in InChEline.split()]


    ## save the file name
    stats['molecule_file'] = molecule_file
    stats.set_index('molecule_file', inplace=True)

    ## store the results
    positions = xyz[['x','y','z']].values
    try:
        positions = positions.astype(np.float64)
    except:
        # there are some weird strings in qm9
        logging.info(f'get_QM9: File {stats.index} Not all values in {positions} could be converted to float, remedying.')
        positions = positions.astype(str)
        func = np.vectorize(lambda val: val.replace('*^', 'e'))
        positions = func(positions)
    
    atoms = ase.Atoms(symbols=xyz['Atom'], positions=positions)
    atoms.info.update(dict(stats.reset_index().T[0]))
    return atoms

def get_QM9(destination: str, delete_raw: bool = True, n_jobs: int = -1):
    """Download and extract the QM9 dataset.
    
    Ramakrishnan, R., Dral, P. O., Rupp, M. & Von Lilienfeld, O. A. Quantum chemistry structures and properties of 134 kilo molecules. Sci. Data 1, 1–7 (2014).
    
    Parameters
    ----------
    destination : str
        Directory to extract the dataset to.
    delete_raw : bool, default True
        Whether to only keep formated data
    n_jobs : int
        number of cores to use
    """
    if not destination.endswith('/'):
        destination += '/'
    URL = "https://figshare.com/ndownloader/files/3195389"
    
    logging.info('get_QM9: Downloading archive file...')
    
    if not os.path.exists(destination):
        os.makedirs(destination+'raw')
    
    with tempfile.TemporaryDirectory() as tempdir:
        urllib.request.urlretrieve(URL, tempdir+'/archive.tar.bz2')
        logging.info(f'get_QM9: Extracting archive to {destination}')
        file = tarfile.open(tempdir+'/archive.tar.bz2')
        file.extractall(destination+'raw')

    def do_one(filename):
        if filename.endswith('xyz'):
            try:
                atoms = qm9_file_to_atoms(destination+'raw/'+filename)
                ase.io.write(destination+f'{atoms.info["molecule_file"]}.extxyz', atoms, format='extxyz')
                return 0
            except:
                logging.info(f'get_QM9: failed to extract {filename}')
                return 1
    
    failures = Parallel(n_jobs=n_jobs)(delayed(do_one)(f) for f in os.listdir(destination+"raw")) 
        
    logging.info(f'get_QM9: {sum(failures)} total failures, or {sum(failures)/len(os.listdir(destination+"raw"))*100}% of the dataset')
    if delete_raw:
        shutil.rmtree(destination+'raw')
    return