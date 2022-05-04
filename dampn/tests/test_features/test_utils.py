import pytest
import unittest.mock as mock

import numpy as np

import dampn.features.utils

class TestAtomFeaturizer:
    def test___init__(self):
        subject = dampn.features.utils.AtomFeaturizer()
        assert hasattr(subject, 'size'), "featurizer parent class does not indicate a size attribute"
        assert hasattr(subject, 'mapping'), "featurizer parent class does not indicate a mapping attribute"
        return
    
    def test_featurize(self):
        subject = dampn.features.utils.AtomFeaturizer()
        
        # patch over the not implemented hidden method
        subject._featurize = lambda x: [1, 2]
        subject.size = 2
        
        out = subject.featurize(['H', 'H', 'H'])
        assert out.shape == (3 ,2), "unexpected feature array shape"
        return
        
        
class TestAtomEncoder:
    def test___init__(self):
        subject = dampn.features.utils.AtomEncoder(atoms=['A', 'B'])
        assert subject.size == 2, "incorrect encoding size"
        assert len(subject.mapping) == 2, "incorrect mapping size"
        return
    
    def test__featurize(self):
        subject = dampn.features.utils.AtomEncoder(atoms=['A', 'B'])
        with pytest.raises(ValueError):
            subject._featurize('Z')
        features = subject._featurize('A')
        assert np.array_equal(features, [1, 0]), "enexpected encoding"
        
        
class TestAtomMassFeaturizer:
    def test___init__(self):
        subject = dampn.features.utils.AtomMassFeaturizer()
        assert subject.size == 1, "incorrect encoding size"
        assert len(subject.mapping) == 1, "incorrect mapping size"
        return
    
    def test__featurize(self):
        subject = dampn.features.utils.AtomMassFeaturizer()
        with pytest.raises(BaseException):
            subject._featurize('Z')
        features = subject._featurize('H')
        assert np.allclose(features, [1.007941]), "enexpected encoding"
        

###############################

class TestDistanceFeaturizer:
    def test___init__(self):
        subject = dampn.features.utils.DistanceFeaturizer()
        assert hasattr(subject, 'size'), "featurizer parent class does not indicate a size attribute"
        assert hasattr(subject, 'mapping'), "featurizer parent class does not indicate a mapping attribute"
        return
    
    def test_featurize(self):
        subject = dampn.features.utils.DistanceFeaturizer()
        
        # patch over the not implemented hidden method
        subject._featurize = lambda x: [1, 2]
        subject.size = 2
        
        out = subject.featurize([1, 2, 3])
        assert out.shape == (3 ,2), "unexpected feature array shape"
        return
    
class TestMathFuncDistanceFeaturizer:
    def test___init__(self):
        subject = dampn.features.utils.MathFuncDistanceFeaturizer()
        assert subject.size == 1, "incorrect encoding size"
        assert len(subject.mapping) == 1, "incorrect mapping size"
        return
    
    def test__featurize(self):
        # make a random function
        func = mock.MagicMock(return_value = 5.0)
        subject = dampn.features.utils.MathFuncDistanceFeaturizer(func=func)
        feats = subject._featurize(2.0)
        assert feats == 5.0, "unexpected return"
        assert func.called_with(2.0), "func not called"
        return

    