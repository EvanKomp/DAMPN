from typing import Type

import numpy
import pandas
import cclib.io

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
    """
    def __init__(
        self,
        elements: Array = None,
        geometry: Array = None,
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
        return
    
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
        """Load a structure from xyz file or cc log file.
        
        Parameters
        ----------
        filepath : str
            Path to load.
        """
        
        if filepath.endswith('.xyz'):
            table = pandas.read_table(
                filepath,
                skiprows=2,
                delim_whitespace=True,
                names=['element', 'x', 'y', 'z'])
            inst = cls(table['element'].values, table[['x', 'y', 'z']].values)
        elif filepath.endswith('.log'):
            data = cclib.io.ccread(filepath)
            atomic_nums = data.atomnos
            geometry = data.atomcoords[-1]
            elements = numpy.vectorize(dampn.constants.periodic_table.__getitem__)(atomic_nums)
            inst = cls(elements, geometry)
        return inst
    
    def save(self, filepath: str):
        """Save structure to xyz format.
        
        Parameters
        ----------
        filepath : str
            Path to save to.
        """
        file = open(filepath, 'w')
        file.write(f'{self.N}\n\n')
        string_array = numpy.concatenate(
            [self.elements,
            self.geometry],
            axis=1
        )
        string_array = pandas.DataFrame(data=string_array)
        string_array.to_csv(file, sep='\t', header=False, index=False)
        file.close()
        return string_array