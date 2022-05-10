from typing import Type

import numpy
import pandas
import cclib.io
import ase.io

import dampn.constants

# Typing
Array = Type[numpy.ndarray]

class Structure:
    """A chemical cartesian stucture.
    
    Parameters
    ----------
    elements : ndarray, optional
        Vector of string element symbols
    geometry : ndarray, optional
        Mattrix of cartesian coordinates, shape (N,3)
    info : dict
        additional information to track of the structure
    """
    def __init__(
        self,
        elements: Array = None,
        geometry: Array = None,
        info: dict = {}
    ):
        self._elements = None
        self._geometry = None
        if elements is None:
            pass
        else:
            self.elements = elements
        if geometry is None:
            pass
        else:
            self.geometry = geometry
        
        self.info = {}
        self.info.update(info)
        return
    
    def __repr__(self):
        return self.atom_string
    
    @property
    def elements(self):
        """ndarray : vector of elements in the structure"""
        return self._elements
    
    @elements.setter
    def elements(self, new_elements):
        new_elements = numpy.array(new_elements).reshape(-1,1).astype(str)
        if self.geometry is not None:
            assert len(self.geometry) == len(new_elements),\
                "Must have same number of coordinates and elements."
        self._elements = new_elements
        return
    
    @property
    def geometry(self):
        """ndarray : matrix of shape (N, 3) of cooridantes"""
        return self._geometry
    
    @geometry.setter
    def geometry(self, new_geometry):
        new_geometry = numpy.array(new_geometry).reshape(-1,3).astype(float)
        if self.elements is not None:
            assert len(self.elements) == len(new_geometry),\
                "Must have same number of coordinates and elements."
        self._geometry = new_geometry
        return
    
    @property
    def N(self):
        """int : number of atoms"""
        if self.complete:
            return len(self.elements)
        else:
            return None
    
    @property
    def complete(self):
        """bool : whether the instance has data"""
        return self.geometry is not None and self.elements is not None
    
    @property
    def atom_string(self):
        """str : atom counts in alphabetical order
        
        example methane "C1H4"
        """
        types, counts = numpy.unique(self.elements, return_counts=True)
        return ''.join(numpy.char.add(types, counts.astype(str)))
    
    @classmethod
    def load(cls, filepath: str):
        """Load a structure from file.
        
        Parameters
        ----------
        filepath : str
            Path to load.
        """
        if filepath.endswith('.log'):
            data = cclib.io.ccread(filepath)
            atomic_nums = data.atomnos
            geometry = data.atomcoords[-1]
            elements = numpy.vectorize(dampn.constants.periodic_table.__getitem__)(atomic_nums)
            inst = cls(elements, geometry)
        else:
            atoms = ase.io.read(filepath)
            elements = atoms.get_chemical_symbols()
            geometry = atoms.positions
            inst = cls(elements, geometry, info=atoms.info)
        return inst
    
    def save(self, filepath: str):
        """Save structure to xyz format.
        
        Parameters
        ----------
        filepath : str
            Path to save to.
        """
        atoms = ase.Atoms(symbols=self.elements.flatten(), positions=self.geometry)
        atoms.info.update(self.info)
        
        ase.io.write(filepath, atoms, format='extxyz')
        return 
    
    @property
    def vis(self):
        try:
            import ase
            import ase.visualize
        except:
            raise ModuleNotFoundError('Visualizing a structure requires converting to ase.Atoms. ase is not installed.')
            
        atoms = ase.Atoms(list(self.elements.reshape(-1)), positions=self.geometry)
        return ase.visualize.view(atoms, viewer='x3d')