"""Download raw datasets and format them."""
import os
import tempfile
import urllib.request
import tarfile

import logging
logger = logging.getLogger(__name__)

def get_Grambow2020_wb97xd3(destination: str):
    if not destination.endswith('/'):
        destination += '/'
    URL = "https://zenodo.org/record/3715478/files/wb97xd3.tar.gz?download=1"
    
    logging.info('Downloading archive file...')
    
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    with tempfile.TemporaryDirectory() as tempdir:
        urllib.request.urlretrieve(URL, tempdir+'/archive.tar.gz')
        logging.info(f'Extracting archive to {destination}')
        file = tarfile.open(tempdir+'/archive.tar.gz')
        file.extractall(destination)
    return
        
        
    