"""Produces a set of files representing the final structures in a dataset.

No two structures should come from the same reaction.
"""
from typing import Type
import os
import shutil
import numpy
import pandas
import logging
logger = logging.getLogger(__name__)

import dampn.base

class Augmentor:
    """Augments dataset if reactions into dataset of structures.
    
    Selects structures from a dataset of reactions to become a dataset of
    structures. Optionally interpolates structures in the reaction to produce
    non-stable, non-transition state structures in the same reaction space.
    
    Parameters
    ----------
    root : str
        Directory containing the data. This directory must have a special form:
        - It must contain only directories
        - Sub directories must have three files, starting with (one each) "r", "p", or "ts"
    interpolation_window : float, 0.5
        The window to produce interpolations between transtion state and R or P, centered.
        Reactant or product is chosen randomly.
        - 0.0 corresponds to always take the exact center of the interpolation between TS and R/P
        - 1.0 corresponds to allowing any structure along the interpolation between TS and R/P
        Position along the interpolation is chosen uniform-random within the window.
    seed : int, optional
        Seed for random operations in the class.
    """
    def __init__(self, root: str, interpolation_window: float = 0.5, seed=None):
        if not root.endswith('/'):
            root += '/'
        self.root = root
        
        self.rng = numpy.random.default_rng(seed)
        
        reaction_directories = os.listdir(root)
        for directory in reaction_directories:
            if not os.path.isdir(root+directory):
                raise ValueError(
                    f'{root+directory} not a directory, aborting. `root` should contain only directories of reactions')
            else:
                got_r = 0
                got_p = 0
                got_ts = 0
                for file in os.listdir(root+directory):
                    if file.startswith('r'):
                        got_r += 1
                    elif file.startswith('p'):
                        got_p += 1
                    elif file.startswith('ts'):
                        got_ts += 1
                    else:
                        raise ValueError(f'File {root+directory+"/"+file} not compatible. Should start with "r", "p", or "ts", aborting')
                if not numpy.all(numpy.array([got_r, got_p, got_ts]) == 1):
                    raise ValueError(
                        f'Reaction at {root+directory} contains files interpereted as ({got_r}) reactants, ({got_p}) products, and ({got_ts}) transition states. There should be exactly one of each'
                    )
        self.reaction_directories = reaction_directories
        self.reactions_to_use = {'reactant': None, 'product': None, 'tstate': None, 'interp': None}
        if interpolation_window <= 0.0 or interpolation_window > 1.0:
            raise ValueError('Interpolation_window must be between 0 and 1')
        else:
            self.interpolation_window = interpolation_window
            
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Data at {self.root} valid dataset for augmenting.")
        return
    
    @property
    def partitioned(self):
        """bool : whether the reactions have been partitioned"""
        return not any([val is None for val in self.reactions_to_use.values()])

    @property
    def partition_df(self):
        """DataFrame : Mapping of reaction to which structure will be taken."""
        if not self.partitioned:
            raise AttributeError('Must first partition dataset')
        else:
            dfs = []
            for key, value in self.reactions_to_use.items():
                df = pandas.DataFrame({'rxn_dir': value, 'use_as': key})
                dfs.append(df)
            return pandas.concat(dfs, ignore_index=True)

    
    def partition_reactions(
        self,
        frac_reactants: float,
        frac_products: float,
        frac_tstates: float,
        frac_interps: float
    ):
        """Split the dataset of reactions into a dataset of structures of the chosen partition.
        
        Parameters
        ----------
        frac_reactants : float
            Portion of reactions that will contribute a reactant as its structure.
        frac_products : float
            Portion of reactions that will contribute a product as its structure.
        frac_tstates : float
            Portion of reactions that will contribute a transition state as its structure.
        frac_interps : float
            Portion of reactions that will contribute an interpolated structure as its structure.
        """
        if not numpy.isclose(numpy.sum([frac_reactants, frac_products, frac_tstates, frac_interps]), 1.0):
            raise ValueError('Fractions should sum to 1.0')
        
        # get number of each type
        N = len(self.reaction_directories)
        n_reactants = round(frac_reactants*N)
        n_products = round(frac_products*N)
        n_tstates = round(frac_tstates*N)
        
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Partitioning dataset of {N} reacitons into {n_reactants} to be selected as reactants, {n_products} to be selected as products, {n_tstates} to be selected as transition states, and the remaining to be used to create random structures that are between transisiton states and stable geometries.")
        
        reaction_directories = numpy.array(self.reaction_directories)
        # sample from the list without replacement
        reaction_directory_indexes = numpy.arange(N)
        reactant_indexes = numpy.array(self.rng.choice(reaction_directory_indexes, n_reactants, replace=False))
        self.reactions_to_use['reactant'] = reaction_directories[reactant_indexes]
        reaction_directories = numpy.delete(reaction_directories, reactant_indexes)
        
        logging.debug(f"{n_reactants} selected to be reactants: {list(self.reactions_to_use['reactant'])}")
        
        reaction_directory_indexes = numpy.arange(len(reaction_directories))
        product_indexes = self.rng.choice(reaction_directory_indexes, n_products, replace=False)
        self.reactions_to_use['product'] = reaction_directories[product_indexes]
        reaction_directories = numpy.delete(reaction_directories, product_indexes)
        
        logging.debug(f"{n_products} selected to be products: {list(self.reactions_to_use['product'])}")
        
        reaction_directory_indexes = numpy.arange(len(reaction_directories))
        tstate_indexes = self.rng.choice(reaction_directory_indexes, n_tstates, replace=False)
        self.reactions_to_use['tstate'] = reaction_directories[tstate_indexes]
        
        logging.debug(f"{n_tstates} selected to be tstates: {list(self.reactions_to_use['tstate'])}")
        
        interp_directories = numpy.delete(reaction_directories, tstate_indexes)
        self.reactions_to_use['interp'] = interp_directories
        logging.debug(f"{N - n_reactants - n_products - n_tstates} selected to be interps: {list(self.reactions_to_use['interp'])}")
        return
    
    def load_structure_from_rxn(self, rxn_dir: str, which: str):
        """Load a specifict reactant, product, or transitions state from a specific reaction.
        
        Parameters
        ----------
        rxn_dir : str
            reaction directory inside of root
        which : str \in "reactant", "product", "tstate"
            Which structure to load
            
        Returns
        -------
        dampn.base.Structure
        """
        # get the file identifier used in the dataset
        which = {'reactant': 'r', 'product': 'p', 'tstate': 'ts'}[which]
        files_list = os.listdir(self.root+rxn_dir)
        file_ind = numpy.argwhere([file.startswith(which) for file in files_list])[0][0]
        file_name = files_list[file_ind]
        
        structure = dampn.base.Structure.load(self.root+rxn_dir+'/'+file_name)
        return structure
            
    def interp_reaction(self, rxn_dir: str):
        """Interpolate between transition state and (reactant or product, random).
        
        Position of interpolation is chosen randomly within the bounds set by this class.
        
        Parameters
        ----------
        rxn_dir : str
            reaction directory inside of root
        """
        which = self.rng.choice(['reactant', 'product'])
        other = self.load_structure_from_rxn(rxn_dir, which)
        tstate = self.load_structure_from_rxn(rxn_dir, 'tstate')
        
        if not numpy.array_equal(other.elements, tstate.elements):
            raise ValueError(f'Cannot interpolate reaction {rxn_dir}, ts and {which} have different element list.')
        
        # get the difference between these two structure
        delta = tstate.geometry - other.geometry
        
        # now determine how far along this vector to place out new structure
        padding = (1.0 - self.interpolation_window)/2
        magnitude = self.rng.uniform(0.0 + padding, 1.0 - padding)
        
        #create new structure
        new_geometry = other.geometry + magnitude * delta
        struct = dampn.base.Structure(elements=other.elements, geometry=new_geometry)
        return struct
    
    def augment_dataset(
        self,
        destination: str,
        overwrite: bool = False,
        shuffle: bool = True,
        frac_reactants: float = 0.15,
        frac_products: float = 0.15,
        frac_tstates: float = 0.5,
        frac_interps: float = 0.2
    ):
        """Produce directory of files ready for loding/featurization.
        
        Parameters
        ----------
        destination : str
            directory to save data in. Created if nonexistant. If the directory is not empty,
            pass overwrite = True.
        overwrite : bool, False
            Delete existing data.
        shuffle : bool, default True
            Whether to shuffle the dataset before saving, otherwise datapoints will be grouped by
            their type (reactant, product, transition state, interpolated structure)
        frac_reactants : float
            Portion of reactions that will contribute a reactant as its structure.
        frac_products : float
            Portion of reactions that will contribute a product as its structure.
        frac_tstates : float
            Portion of reactions that will contribute a transition state as its structure.
        frac_interps : float
            Portion of reactions that will contribute an interpolated structure as its structure.
        """
        if not destination.endswith('/'):
            destination += '/'
        if not os.path.exists(destination):
            os.makedirs(destination)
            logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Saving augmented data to {destination}.")
        else:
            if len(os.listdir(destination)) > 0:
                if overwrite:
                    logging.info(f"{type(self).__module__+'.'+type(self).__name__}:{destination} not empty, overwriting.")
                    shutil.rmtree(destination)
                    os.makedirs(destination)
                else:
                    raise OSError(f"{destination} not empty, specify `overwrite` to enable removal of current contents.")
            else:
                logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Saving augmented data to {destination}.")
        
        self.partition_reactions(frac_reactants, frac_products, frac_tstates, frac_interps)
        
        # get the metadata, we will be adding an extra column of the index
        metadata = self.partition_df
        if shuffle:
            metadata = metadata.sample(frac=1.0)
        metadata.reset_index(drop=True, inplace=True)
        metadata.index.name = 'data_index'
        
        # loop through each reaction, get the correct structure, and save
        for index, row in metadata.iterrows():
            rxn_dir = row['rxn_dir']
            use_as = row['use_as']
            
            # in this case we just want to grab the current structure
            if any([use_as == which for which in ['reactant', 'product', 'tstate']]):
                structure = self.load_structure_from_rxn(rxn_dir=rxn_dir, which=use_as)
            # in this case we want to interpolate
            elif use_as == 'interp':
                structure = self.interp_reaction(rxn_dir)
            else:
                raise ValueError(f"Encountered data partition type '{use_as}', which is not valid.")
            
            logging.debug(f"Saving structure of type {use_as} from original reaction {rxn_dir} as example {index}")
            
            structure.save(destination+f"{str(index).zfill(6)}.xyz")
        
        metadata.to_csv(destination+"metadata.csv")
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Augmented dataset metadata saved at {destination+'metadata.csv'}")
        return
            
            
        
                