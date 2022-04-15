import numpy as np
import scipy.spatial

import dampn.base
import dampn.features.utils

from typing import List, Iterable

import logging
logger = logging.getLogger(__name__)

class StructureFeaturizer:
    """Featurize structures into edge, node features.
    
    Example
    -------
    >>>structure = Structure.load('h2o.xyz')
    >>>featurizer = StructureFeaturizer()
    >>>featurizer.featurize(structure)
    (
        [ndarray([[1,0], [1,0], [0,1]])],
        [ndarray([0, 0, 1, 1, 2, 2], [1, 2, 2, 0, 0, 1])]
        [ndarray([[1.355], [.957], [.957], [1.355], [.957], [.957]])]
    )
    
    Parameters
    ----------
    atom_featurizers : iterable of AtomFeaturizers
        featurizers to use to convert atomic symbol into feature vector
    distance_featurizers : iterable of DistanceFeaturizer
        featurizers to use to convert distance into feature vector
    distance_cutoff : float, optional
        how far to away to no longer consider edges. Will drastically reduce
        neural net overhead for structures large enough for global interaction
        to be minimal
        
    Attributes
    ----------
    mapping : dict of dict, which features for atoms and edges are associated with what index
    atom_featurizers : iterable of AtomFeaturizers
        featurizers to use to convert atomic symbol into feature vector
    distance_featurizers : iterable of DistanceFeaturizer
        featurizers to use to convert distance into feature vector
    distance_cutoff : float, optional
        how far to away to no longer consider edges. Will drastically reduce
        neural net overhead for structures large enough for global interaction
        to be minimal
    """
    def __init__(
        self,
        atom_featurizers: Iterable[dampn.features.utils.AtomFeaturizer] = None,
        distance_featurizers: Iterable[dampn.features.utils.DistanceFeaturizer] = None,
        distance_cutoff: float = None
    ):
        # if featurizers were not specifically built
        if atom_featurizers is None:
            atom_featurizers = [dampn.features.utils.AtomEncoder()]
        if distance_featurizers is None:
            distance_featurizers = [dampn.features.utils.MathFuncDistanceFeaturizer()]
            
        if not hasattr(atom_featurizers, '__len__'):
            atom_featurizers = [atom_featurizers]
        if not hasattr(distance_featurizers, '__len__'):
            distance_featurizers = [distance_featurizers]
            
        for featurizer in atom_featurizers:
            assert isinstance(featurizer, dampn.features.utils.AtomFeaturizer)
        for featurizer in distance_featurizers:
            assert isinstance(featurizer, dampn.features.utils.DistanceFeaturizer)
        
        self.distance_cutoff = distance_cutoff
        self.atom_featurizers = atom_featurizers
        self.distance_featurizers = distance_featurizers
        return
    
    @property
    def mapping(self):
        atom_mapping = {}
        atom_mapping_index = 0
        for featurizer in self.atom_featurizers:
            for feature_name in featurizer.mapping.values():
                atom_mapping[atom_mapping_index] = feature_name
                atom_mapping_index += 1
        distance_mapping = {}
        distance_mapping_index = 0
        for featurizer in self.distance_featurizers:
            for feature_name in featurizer.mapping.values():
                distance_mapping[distance_mapping_index] = feature_name
                distance_mapping_index += 1
        return {'atom_features': atom_mapping, 'distance_features': distance_mapping}
    
    def _featurize_one(self, structure: dampn.base.Structure):
        """Featurize a single structure.
        
        Parameters
        ----------
        structure : Structure
            the structure to featurize
            
        Returns
        -------
        ndarray (number of atoms, size of features) Atom features matrix
        ndarray (2* number of edges, 2) Neighbor list of edges
        ndarray (2* number of edges, size of edge features) Edge feature matrix
        
        """
        # get features for list of atoms using each specified atom featurizer
        atom_features = [
            featurizer.featurize(structure.elements) for featurizer in self.atom_featurizers]
        atom_features = np.concatenate(atom_features, axis=1)
        
        # compute pairwise distance, lower triangle
        distance_matrix = np.tril(
            scipy.spatial.distance_matrix(
                structure.geometry, structure.geometry))
        # consider only greater than 0 indexes
        mask = distance_matrix > 0
        # if we have a cuttoff, cut them off
        if self.distance_cutoff is not None:
            mask *= distance_matrix < self.distance_cutoff
        # this returns a tuple of array - index along 0 and index along 1
        edge_indexes = np.where(mask)
        
        # get features for list of distances using each specified distance featurizer
        distance_features = [
            featurizer.featurize(distance_matrix[edge_indexes]) for featurizer in self.distance_featurizers]
        distance_features = np.concatenate(distance_features, axis=1)
        
        # the ML framework we use has directed edges, and since the neighbor list
        # (edge_indexes above) is arbitrary, we will have dual channel
        neighbor_list = np.array(edge_indexes)
        neighbor_list = np.concatenate(
            [neighbor_list, np.flip(neighbor_list, axis=0)], axis=1).T
        distance_features = np.concatenate([distance_features, distance_features], axis=0)
        
        return neighbor_list, atom_features, distance_features
        
    
    def featurize(self, structures: Iterable[dampn.base.Structure]):
        """Convert an iterable of structures into atom an distance features.
        
        Parameters
        ----------
        structures : Iterable of Structure
            Structures to featurize
            
        Returns
        -------
        list of ndarray : atom feature arrays for each structure
        list of ndarray : adjacency matrics for each structure
        list of ndarrat : edge feature arrays for each structure
        """
        if not hasattr(structures, '__len__'):
            structures = [structures]
        A = []
        F = []
        E = []
        for structure in structures:
            Ai, Fi, Ei = self._featurize_one(structure)
            F.append(Fi)
            A.append(Ai)
            E.append(Ei)
        return A, F, E
            
        
            
        
        
        