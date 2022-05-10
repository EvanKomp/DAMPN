"""Download raw dataset and format them."""
import os
import tempfile
import urllib.request
import tarfile

import logging
logger = logging.getLogger(__name__)

def get_Grambow2020_wb97xd3(destination: str):
    """Download and extract the Grambow 2020 dataset.
    
    Grambow, C. A., Pattanaik, L. & Green, W. H. Reactants, products, and transition states of elementary chemical reactions based on quantum chemistry. Sci. Data 7, 1–8 (2020).
    
    Parameters
    ----------
    destination : str
        Directory to extract the dataset to.
    """
    if not destination.endswith('/'):
        destination += '/'
    URL = "https://zenodo.org/record/3715478/files/wb97xd3.tar.gz?download=1"
    
    logging.info('get_Grambow2020_wb97xd3: Downloading archive file...')
    
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    with tempfile.TemporaryDirectory() as tempdir:
        urllib.request.urlretrieve(URL, tempdir+'/archive.tar.gz')
        logging.info(f'get_Grambow2020_wb97xd3: Extracting archive to {destination}')
        file = tarfile.open(tempdir+'/archive.tar.gz')
        file.extractall(destination)
    return
        
        
    