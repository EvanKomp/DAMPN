import pytest

import numpy as np

import dampn.base
import dampn.features.structure_featurizer
import dampn.features.utils

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
xyz = os.path.join(dir_path, '..', 'data_for_tests', 'xyz.xyz')

@pytest.fixture
def STRUCTURE():
    structure = dampn.base.Structure.load(xyz)
    return structure

class TestStructureFeaturizer:
    
    def test___init__(self):
        atom_featurizer = dampn.features.utils.AtomEncoder()
        dist_featurizer = dampn.features.utils.MathFuncDistanceFeaturizer()
        subject = dampn.features.structure_featurizer.StructureFeaturizer(
            atom_featurizers = [atom_featurizer, atom_featurizer],
            distance_featurizers = [dist_featurizer, dist_featurizer]
        )
        
        assert len(subject.mapping) == 2, "mapping is not the atom and distance mappings"
        assert len(subject.mapping['atom_features']) == 2*atom_featurizer.size, "atom features mapping does mot match featurizers"
        assert len(subject.mapping['distance_features']) == 2*dist_featurizer.size, "distance features mapping does mot match featurizers"
        return
    
    def test__featurize_one(self, STRUCTURE):
        atom_featurizer = dampn.features.utils.AtomEncoder()
        dist_featurizer = dampn.features.utils.MathFuncDistanceFeaturizer()
        subject = dampn.features.structure_featurizer.StructureFeaturizer(
            atom_featurizers = [atom_featurizer, atom_featurizer],
            distance_featurizers = [dist_featurizer, dist_featurizer]
        )
        
        features = subject._featurize_one(STRUCTURE)
        neighbor_list, atom_features, distance_features = features
        
        assert neighbor_list.shape == (132,2), "neighbor_list incorrect shape"
        assert atom_features.shape == (12,8), "atom features incorrect shape"
        assert distance_features.shape == (132,2), "distance features incorrect shape."
        
        # check if distances are cutoff correctly
        subject.distance_cutoff = 2.0
        features = subject._featurize_one(STRUCTURE)
        neighbor_list, atom_features, distance_features = features
        assert neighbor_list.shape == (24,2), "neighbor_list incorrect shape"
        assert atom_features.shape == (12,8), "atom features incorrect shape"
        assert distance_features.shape == (24,2), "distance features incorrect shape."
        return
        
    def test_featurize(self, STRUCTURE):
        structures = [STRUCTURE, STRUCTURE]
        atom_featurizer = dampn.features.utils.AtomEncoder()
        dist_featurizer = dampn.features.utils.MathFuncDistanceFeaturizer()
        subject = dampn.features.structure_featurizer.StructureFeaturizer(
            atom_featurizers = [atom_featurizer, atom_featurizer],
            distance_featurizers = [dist_featurizer, dist_featurizer],
            distance_cutoff = 2.0
        )
        A, F, E = subject.featurize(structures)
        assert A[0].shape == (24,2), "neighbor_list incorrect shape"
        assert F[0].shape == (12,8), "atom features incorrect shape"
        assert E[0].shape == (24,2), "distance features incorrect shape."
        assert len(A) == len(E) == len(F) == 2, "didn't featurize 2 structres"
        return
        