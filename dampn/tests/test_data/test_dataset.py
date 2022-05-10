import pytest
from unittest import mock
import tempfile
import os

import numpy as np
import dgl
import itertools

import dampn.data.dataset

@pytest.fixture
def EXAMPLE_GENERATOR():
    """Create data matrixes for an example."""
    shard_sizes = (3,3,3,2)
    node_sizes = (5,3,6)
    
    F_template = np.array(range(np.max(node_sizes)*4)).reshape(-1, 4)
    E_template = np.arange(30).reshape(-1,1)
    feats_all = None
    
    ys = np.arange(np.sum(shard_sizes))
    ids_all = np.array([f'm{i}' for i in range(np.sum(shard_sizes))])
    split_ids_all = None
    
    def gen():
        s = 0
        for shard_size in shard_sizes:
            i = 0
            F = []
            A = []
            E = []
            y = []
            ids = []
            feats = []
            split_ids = []
            while i < shard_size:
                node_size = node_sizes[i]
                edge_size = int(node_size**2 - node_size)
                F.append(F_template[:node_size,:])
                A.append(
                    np.array(list(itertools.permutations(list(range(node_size)), 2))))
                E.append(E_template[:edge_size,:])
                y.append(ys[s])
                ids.append(ids_all[s])
                i += 1
                s += 1

            yield A, F, E, y, ids, feats_all, split_ids_all
                
    full_example = (F_template, E_template, ys, ids_all, feats_all, split_ids_all)
    
    
    return (gen(), full_example, shard_sizes, node_sizes)

class TestDataset:
    """Tough test.
    
    We want the class to create, save and reload dataset, but this test should not restrain a particular format on disk. As long as it can recall the original data
    """
    def test_create_return(self, EXAMPLE_GENERATOR):
        """Here we assert that what is returned as a dataset is the correct format."""
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dataset = dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            assert tuple(dataset.metadata['shard_size']) == shard_sizes, 'incorrect shard sizes returned.'
            assert dataset.data_dir == data_dir, "incorrect data location"
            assert not dataset.has_system_features, "dataset should not have system features"
            assert dataset.num_shards == len(shard_sizes), "unexpected number of shards"
            assert [id_ in dataset.ids for id_ in ids_all], "not all ids saved"
            assert dataset.ids.shape == (len(shard_sizes), max(shard_sizes)), "ids not the same shape as the data should be"
            for i, shard in dataset.metadata.iterrows():
                assert tuple(shard['num_nodes']) == node_sizes[:shard_sizes[i]], f"shard {i} unexpected node sizes"
                assert shard['num_edges'] == [node_size**2 - node_size for node_size in node_sizes[:shard_sizes[i]]], f"shard {i} unexpected node sizes"
                
            assert dataset.size == sum(shard_sizes), "unexpected dataset size"
            print(dataset.flat_ids,ids_all)
            assert all(dataset.flat_ids == ids_all), "ids not in the correct order"
        return
    
    def test_load(self, EXAMPLE_GENERATOR):
        """Here we are concerned with loading from file already"""
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            dataset = dampn.data.dataset.Dataset(data_dir)
            assert tuple(dataset.metadata['shard_size']) == shard_sizes, 'incorrect shard sizes returned.'
            assert dataset.data_dir == data_dir, "incorrect data location"
            assert not dataset.has_system_features, "dataset should not have system features"
            assert dataset.num_shards == len(shard_sizes), "unexpected number of shards"
            assert [id_ in dataset.ids for id_ in ids_all], "not all ids saved"
            assert dataset.ids.shape == (len(shard_sizes), max(shard_sizes)), "ids not the same shape as the data should be"
            for i, shard in dataset.metadata.iterrows():
                assert tuple(shard['num_nodes']) == node_sizes[:shard_sizes[i]], f"shard {i} unexpected node sizes"
                assert shard['num_edges'] == [node_size**2 - node_size for node_size in node_sizes[:shard_sizes[i]]], f"shard {i} unexpected node sizes"
                
            assert dataset.size == sum(shard_sizes), "unexpected dataset size"
            print(dataset.flat_ids,ids_all)
            assert all(dataset.flat_ids == ids_all), "ids not in the correct order"
            
            ### Now what if the dataset is no longer correct?
            os.mkdir(tmpdirname+'fakedata/')
            with pytest.raises(BaseException):
                dampn.data.dataset.Dataset(tmpdirname+'fakedata/')
            return
            
    def test__load_shard_matrices(self, EXAMPLE_GENERATOR):
        """Most other methods require this one to be working perfectly."""
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dataset = dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            
            shard_num = 1
            Ac, Fc, Ec, yc, idsc, featsc, split_ids = dataset._load_shard_matrices(shard_num)
            
            node_sizes = node_sizes[:shard_sizes[shard_num]]
            edge_sizes = tuple([int(node_size**2 - node_size) for node_size in node_sizes])

            assert Ac.shape == (sum(edge_sizes), 2), "incorrect A size"
            assert Fc.shape == (sum(node_sizes), 4), "incorrect F size"
            assert Ec.shape == (sum(edge_sizes), 1), "incorrect E size"
            assert yc.shape == (shard_sizes[shard_num], 1), "incorrect y size"
            assert idsc.shape == (shard_sizes[shard_num], 1), "incorrect id size"
            assert featsc == None, "should be no feats"
            assert split_ids == None, "should be no split ids"
        return
    
    def test__get_example_from_shard(self, EXAMPLE_GENERATOR):
        """Test to ensure that data for a datapoint is extracted correctly. 
        
        Any other method that involves selecting/getting/iterating the data accesses
        the shard data from `_load_shard_matrices` through here. This function should
        get the correct positions within the shard matrix based on metadata.
        """
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dataset = dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            
            # first test that a specific example is retrieved properly
            # second example from second shard
            shard_num = 1
            example_num = 1
            Ai, Fi, Ei, yi, id, featsi, split_idsi = dataset._get_example_from_shard(
                (shard_num, example_num))

            node_size = node_sizes[example_num]
            edge_size = int(node_size**2 - node_size)
            
            assert len(Ai) == len(Ei) == edge_size, "incorrect number of edges"
            assert len(Fi) == node_size, "incorrect number of nodes"
            
            assert yi == np.sum(shard_sizes[:shard_num]) + example_num, "unexpected yi"
            assert id == f'm{int(yi)}', "unexpected id"
            assert split_idsi == None, "Expected split id to be None"
            assert featsi == None, "Expected system feats to be None"
            
            assert np.all(
                Ai == np.array(list(itertools.permutations(list(range(node_size)), 2)))
            ), "unexpected adjacency matrix values"
            assert np.all(
                Ei == E_template[:edge_size]
            ), "unexpected edge features values"
            assert np.all(
                Fi == F_template[:node_size]
            ), "unexpected node feature values"
            
            # now we test if it can handle returing and using the same shard
            shard, output_first = dataset._get_example_from_shard(
                (shard_num, example_num), return_shard=True)
            output_second = dataset._get_example_from_shard(
                (shard_num, example_num), shard=shard)
            
            for i, item_from_first in enumerate(output_first):
                item_from_second = output_second[i]
                assert np.all(item_from_first == item_from_second), "data from dataset call and from returned shard call do not match"
        return
    
    def test___getitem__(self,  EXAMPLE_GENERATOR):
        """dataset should be indexable by index, position, and lastly id if the others fail"""
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dataset = dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            shard_num = 1
            example_num = 1
            
            # first ensure that it calls the method above which is aready tested
            # and assigns the correct data
            with mock.patch(
                "dampn.data.dataset.Dataset._get_example_from_shard",
                return_value=mock.MagicMock()
            ) as mocked_get:
                try:
                    dataset[(shard_num, example_num)]
                except:
                    pass
                mocked_get.assert_called_with((shard_num, example_num))
            
            #now check that we get a graph with the correct data
            graph, yi, id, featsi, split_idsi = dataset[(shard_num, example_num)]
            assert isinstance(graph, dgl.DGLHeteroGraph)
            
            node_size = node_sizes[example_num]
            edge_size = int(node_size**2 - node_size)
            
            Ai = np.array(list(map(lambda tensor: tensor.numpy(), graph.all_edges()))).T
            Ei = graph.edata['e'].numpy()
            Fi = graph.ndata['f'].numpy()
            
            assert len(Ai) == len(Ei) == edge_size, "incorrect number of edges"
            assert len(Fi) == node_size, "incorrect number of nodes"
            
            assert yi == np.sum(shard_sizes[:shard_num]) + example_num, "unexpected yi"
            assert id == f'm{int(yi)}', "unexpected id"
            assert split_idsi == None, "Expected split id to be None"
            assert featsi == None, "Expected system feats to be None"
            
            assert np.all(
                Ai == np.array(list(itertools.permutations(list(range(node_size)), 2)))
            ), "unexpected adjacency matrix values"
            assert np.all(
                Ei == E_template[:edge_size]
            ), "unexpected edge features values"
            assert np.all(
                Fi == F_template[:node_size]
            ), "unexpected node feature values"
            
            # now check we can use other indexes
            graph, yi, id, featsi, split_idsi = dataset[4]
            assert yi == np.sum(shard_sizes[:shard_num]) + example_num, "unexpected yi"
            graph, yi, id, featsi, split_idsi = dataset['m4']
            assert yi == np.sum(shard_sizes[:shard_num]) + example_num, "unexpected yi"
        return
    
    def test_select(self,  EXAMPLE_GENERATOR):
        """just check that the new dataset has the correct data"""
        (gen, full_example, shard_sizes, node_sizes) = EXAMPLE_GENERATOR
        ( F_template, E_template, ys, ids_all, feats_all, split_ids_all) = full_example
        with tempfile.TemporaryDirectory() as tmpdirname:
            data_dir = tmpdirname+'data/'
            dataset = dampn.data.dataset.Dataset.create_dataset(gen, data_dir)
            
            random_ids = np.random.choice(ids_all, 5, replace=False)
            print(random_ids)
            indexes = [int(s[1:]) for s in random_ids]
            
            print(dataset.split_ids)
            
            id_set = dataset.select(tmpdirname+'data_by_id/', ids=random_ids)
            print(id_set.split_ids)
            index_set = dataset.select(
                tmpdirname+'data_by_index/', indexes=indexes)
            
            assert id_set.shard_size == dataset.shard_size == index_set.shard_size,\
                "unexpected shard size"
            
            for id_ in random_ids:
                og_loc = dataset._get_id_location(id_)
                loc_id_set = id_set._get_id_location(id_)
                loc_index_set = index_set._get_id_location(id_)
                
                og_arrays = dataset._get_example_from_shard(og_loc)
                id_arrays = id_set._get_example_from_shard(loc_id_set)
                index_arrays = index_set._get_example_from_shard(loc_index_set)
                
                for i in range(len(og_arrays)):
                    assert np.array_equal(og_arrays[i], id_arrays[i]), "data from id selected dataset does not match"
                    assert np.array_equal(og_arrays[i], index_arrays[i]), "data from index selected dataset does not match"              
        return
                
            