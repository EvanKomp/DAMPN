"""This class partitions and augments a set of reactions into a set of structures from
those reactions."""

import pytest
import tempfile
import os

import dampn.data.augmentor

FILESTRING1 = """2

H       0    0      0
C       0.684925137004899     -0.092756484972356      1.213323839393219
"""

FILESTRING2 = """2

H       0    0      1
C       0.684925137004899     -0.092756484972356      1.213323839393219
"""


class RxnDatasetContext:
    """Creates temporary directory in the correct or incorrect form."""
    def __init__(self, form: str = 'good'):
        self.form = form
        return
    
    def __enter__(self):
        tempdir = tempfile.TemporaryDirectory()
        self.tempdir = tempdir
        root = tempdir.name+'/'
        
        # make reaction directories
        os.mkdir(root+'rxn1')
        os.mkdir(root+'rxn2')
        
        # add reactions
        def write(to, what):
            file = open(to, 'w')
            file.write(what)
            file.close()
            return
        write(root+'rxn1/r.xyz', FILESTRING1)
        write(root+'rxn1/p.xyz', FILESTRING1)
        write(root+'rxn1/ts.xyz', FILESTRING2)
        write(root+'rxn2/r.xyz', FILESTRING1)
        write(root+'rxn2/p.xyz', FILESTRING1)
        write(root+'rxn2/ts.xyz', FILESTRING2)
        
        # consider failure modes
        if self.form == 'good':
            pass
        elif self.form == 'nondir':
            write(root+'notdir.txt', 'hello world')
        elif self.form == 'missing structure':
            os.remove(root+'rxn1/p.xyz')
        elif self.form == 'bad file':
            write(root+'rxn1/X.xyz', FILESTRING1)
        elif self.form == 'extra state':
            write(root+'rxn1/r2.xyz', FILESTRING1)
        else:
            pass
        return root
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.tempdir.cleanup()
        return
    
    
class TestAugmentor:

    def test___init__(self):
        # should pass
        with RxnDatasetContext(form='good') as root:
            augmentor = dampn.data.augmentor.Augmentor(root=root)
            assert not augmentor.partitioned, "Creation of empty partion dataset"
            assert len(augmentor.reaction_directories) == 2, "Not not find reactions"
            assert 'rxn1' in augmentor.reaction_directories, "rxn names not tracked"
        
        # failure modes:
        with RxnDatasetContext(form='nondir') as root:
            with pytest.raises(ValueError):
                augmentor = dampn.data.augmentor.Augmentor(root=root)
        with RxnDatasetContext(form='missing structure') as root:
            with pytest.raises(ValueError):
                augmentor = dampn.data.augmentor.Augmentor(root=root)
        with RxnDatasetContext(form='bad file') as root:
            with pytest.raises(ValueError):
                augmentor = dampn.data.augmentor.Augmentor(root=root)
        with RxnDatasetContext(form='extra state') as root:
            with pytest.raises(ValueError):
                augmentor = dampn.data.augmentor.Augmentor(root=root)       
        return
    
    def test_partition_dataset(self):
        with RxnDatasetContext(form='good') as root:
            augmentor = dampn.data.augmentor.Augmentor(root=root)
            # fraction bad
            with pytest.raises(ValueError):
                augmentor.partition_reactions(frac_reactants=0.5, frac_products=0.6, frac_tstates=0, frac_interps=0)
            
            # should get one reactant and one product
            augmentor.partition_reactions(frac_reactants=0.5, frac_products=0, frac_tstates=0, frac_interps=0.5)
            assert augmentor.partitioned, "did noit flip the partitioned flag"
            assert len(augmentor.reactions_to_use['reactant']) == 1, "did not add one reaction as reactant"
            assert len(augmentor.reactions_to_use['interp']) == 1, "did not add one reaction as interp"
        return
    
    
    def test_interp_reaction(self):
        """Ensure interpolation is within the window"""
        with RxnDatasetContext(form='good') as root:
            augmentor = dampn.data.augmentor.Augmentor(root=root, interpolation_window=.1)
            struc = augmentor.interp_reaction('rxn1')
            assert 0.4 <= struc.geometry[0,2] <= 0.6, "did not interpolate within window"
            
        return
    
    def test_augment_dataset(self):
        with RxnDatasetContext(form='good') as root:
            augmentor = dampn.data.augmentor.Augmentor(root=root)
            
            with tempfile.TemporaryDirectory() as destination:
                augmentor.augment_dataset(destination)
                assert '000000.xyz' in os.listdir(destination), 'Did not save first structure'
                assert 'metadata.csv' in os.listdir(destination), 'Did not save metadata'
                
                with pytest.raises(OSError):
                    augmentor.augment_dataset(destination)
                file = open(destination+'/test.txt', 'w')
                file.write('hello world')
                file.close()
                augmentor.augment_dataset(destination, overwrite=True)
                assert 'test.txt' not in os.listdir(destination), 'Did not overwrite'
        return