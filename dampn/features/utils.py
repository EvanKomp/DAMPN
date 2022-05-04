import numpy as np
import molmass

from typing import List, Iterable, Union

import dampn.constants

import logging
logger = logging.getLogger(__name__)

class AtomFeaturizer:
    """Converts atomic symbol to a feature vector.
    
    Abstract class,
    
    Example
    -------
    >>>featurizer = AtomFeaturizer()
    >>>featurizer.featurize('H')
    ndarray
    
    Attributes
    ----------
    size
    mapping : dict of position, feature meaning in feature vector 
    """
    def __init__(self, **kwargs):
        self._size = None
        return
    
    @property
    def size(self):
        """int: size of feature vector from this featurizer"""
        return self._size
    
    @size.setter
    def size(self, new_size):
        if type(new_size) == int:
            self._size = new_size
        else:
            raise ValueError('`size` must be int')
    
    @property
    def mapping(self):
        return {}
    
    def featurize(self, atoms: Iterable[str]):
        """Produce feature vector for atom with atomic symbol.
        
        Parameters
        ----------
        atoms : iterable of str
        
        Returns
        -------
        ndarray of shape (len input, AtomFeaturizer.size)
        """
        if type(atoms) == str:
            atoms = [atoms]
        else:
            pass
        try:
            num_examples = len(atoms)
        except:
            raise ValueError(f'Input does not appear to be a tring or iterable of strings.')
        
        feature_vector = []
        for atom in atoms:
            feature_vector.append(self._featurize(atom))
            
        try:
            feature_vector = np.array(feature_vector).reshape(-1,self.size)
            assert len(feature_vector) == num_examples
        except:
            raise ValueError(f'Error when featurizing atoms, unable to coerce featurized output of size {feature_vector.size} to expected shape (number of examples, featurizer size) of {(num_examples, self.size)}')
        return feature_vector
    
    def _featurize(self, atomic_symbol: str):
        """Define in child class.
        
        Featurize a single atom represented by an atomic symbols string.
        """
        raise NotImplemented()
        
        
class AtomEncoder(AtomFeaturizer):
    """Converts atomic symbol to atom encoding vector.
    
    Parameters
    ----------
    atoms : list of str
        the possible atomic symbols. determines the size of the output vector
    
    Example
    -------
    >>>featurizer = AtomEncoder(atoms=['H', 'O', 'C'])
    >>>featurizer.featurize('H')
    ndarray([[1, 0, 0]])
    >>>featurizer.featurize('C')
    ndarray([[0, 0, 1]])
    
    Attributes
    ----------
    size
    mapping : dict of position, feature meaning in feature vector 
    """
    def __init__(self, atoms: List[str] = ['H','C','N','O']):
        atoms = list(atoms)
        for item in atoms:
            assert type(item) == str
        self.size = len(atoms)
        self.atoms = np.array(atoms)
        
    @property
    def mapping(self):
        return {i: f'is_{s}' for i, s in enumerate(self.atoms)}
    
    
    def _featurize(self, atomic_symbol: str):
        if atomic_symbol not in self.atoms:
            raise ValueError(
                f"incoming atom string `{atomic_symbol}` not in the list known by this featurizer {self.atoms}"
            )
        vector = np.zeros((self.size,), dtype=int)
        vector[np.where(self.atoms == atomic_symbol)[0][0]] = 1
        return vector
    
class AtomMassFeaturizer(AtomFeaturizer):
    """Converts atomic symbol to atom mass.
    
    Example
    -------
    >>>featurizer = AtomMassFeaturizer()
    >>>featurizer.featurize('H')
    ndarray([[1.008]])
    
    Attributes
    ----------
    size
    mapping : dict of position, feature meaning in feature vector 
    """
    def __init__(self, **kwargs):
        self.size = 1
        
    @property
    def mapping(self):
        return {0: 'atom_mass'}
    
    def _featurize(self, atomic_symbol: str):
        f = molmass.Formula(atomic_symbol)
        return f.mass
    

##############################################################################
# edge features distance to something

class DistanceFeaturizer:
    """Converts distance symbol to a feature vector.
    
    Abstract class,
    
    Example
    -------
    >>>featurizer = DistanceFeaturizer()
    >>>featurizer.featurize(.789)
    ndarray
    
    Attributes
    ----------
    size
    mapping : dict of position, feature meaning in feature vector 
    """
    def __init__(self, **kwargs):
        self._size = None
        return
    
    @property
    def size(self):
        """int: size of feature vector from this featurizer"""
        return self._size
    
    @size.setter
    def size(self, new_size):
        if type(new_size) == int:
            self._size = new_size
        else:
            raise ValueError('`size` must be int')
    
    @property
    def mapping(self):
        return None
    
    def featurize(self, distances: Iterable[float]):
        """Produce feature vector for an edge with a certain distance.
        
        Parameters
        ----------
        distances : iterable of float
        
        Returns
        -------
        ndarray of shape (len input, DistanceFeaturizer.size)
        """
        if type(distances) == float:
            distances = [distances]
        else:
            pass
        try:
            num_examples = len(distances)
        except:
            raise ValueError(f'Input does not appear to be a tring or iterable of strings.')
        
        feature_vector = []
        for distance in distances:
            feature_vector.append(self._featurize(distance))
            
        try:
            feature_vector = np.array(feature_vector).reshape(-1,self.size)
            assert len(feature_vector) == num_examples
        except:
            raise ValueError(f'Error when featurizing distances, unable to coerce featurized output of size {feature_vector.size} to expected shape (number of examples, featurizer size) of {(num_examples, self.size)}')
        return feature_vector
    
    def _featurize(self, distance: float):
        """Define in child class."""
        raise NotImplemented()
        
class MathFuncDistanceFeaturizer(DistanceFeaturizer):
    """Apply a function to distance.
    
    Parameters
    ----------
    func : callable or str
        If callable, called to featurize float distance
        If string, options are
            - "log" natural logarithm
            - "lin" identity
            - "inv" inverse
    
    Example
    -------
    >>>featurizer =         (func='inv')
    >>>featurizer.featurize(0.5)
    ndarray([[2.0]])
    
    Attributes
    ----------
    size
    mapping : dict of position, feature meaning in feature vector 
    """
    def __init__(self, func: Union[str, callable] = 'lin', **kwargs):
        self.size = 1
        if callable(func):
            self.func = func
        elif func == 'log':
            def log_distance(x):
                return np.log(x)
            self.func = log_distance
        elif func == 'lin':
            def distance(x):
                return x
            self.func = distance
        elif func == 'inv':
            def inverse_distance(x):
                return 1/x
            self.func = inverse_distance
        else:
            raise ValueError(f'{func} not a valid function')
        return
    
    def _featurize(self, distance: float):
        return self.func(distance)
        
    @property
    def mapping(self):
        return {0: self.func.__name__}
    
    
  