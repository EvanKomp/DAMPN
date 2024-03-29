import pytest
import tempfile
import os
import numpy

import dampn.base

dir_path = os.path.dirname(os.path.realpath(__file__))
logfile = os.path.join(dir_path, '.', 'data_for_tests', 'qchem.log')
xyz = os.path.join(dir_path, '.', 'data_for_tests', 'xyz.xyz')

class TestStructure:
    
    def test____init__(self):
        elements = ['O', 'H']
        geometry = [[1,2,3], [2,3,4]]
        struct = dampn.base.Structure(elements, geometry)
        assert struct.complete, "Structure did not save elements and geom"
        assert struct.N == 2, 'wrong number of atoms'
        assert struct.geometry.sum() == 15, "incorrect geometry saved"
        assert struct.atom_string == 'H1O1', "wrong elements"
        return
    
    def test_load(self):
        struct = dampn.base.Structure.load(xyz)
        assert struct.atom_string == 'C6H6', "xyz file atoms not loaded properly"
        
        struct = dampn.base.Structure.load(logfile)
        assert struct.atom_string == 'C2H3N3O1', "xyz file atoms not loaded properly"
        
    def test_save(self):
        with tempfile.TemporaryDirectory() as tempdir:
            struct = dampn.base.Structure.load(logfile)
            struct.save(tempdir+'/test.xyz')
            struct_ = struct.load(tempdir+'/test.xyz')
            assert numpy.allclose(struct.geometry, struct_.geometry), "geometry not saved properly"
            assert numpy.array_equal(struct.elements, struct_.elements), "elements not saved properly"
        return