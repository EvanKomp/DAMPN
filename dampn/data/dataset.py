import os
import shutil
import ast

import pandas
import numpy

import dgl
import torch

import logging
logger = logging.getLogger(__name__)

class Dataset:
    """Dataset of graphs saved to disk.
     
    Notes
    -----
    "index" refers to a position in the overall dataset
    "location" refers to the shard number, and example number in the shard
    "id" refers to the identifier assigned to the example
    
    """
    def __init__(self, data_dir):
        if type(data_dir) != str:
            raise ValueError('`data_dir` must be str')
        if not data_dir.endswith('/'):
            data_dir += '/'
        
        self._parse_dir(data_dir)
        self._data_dir = data_dir
        
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Successfully connected to dataset at {data_dir}.")
        return
    
    def __len__(self):
        return self.size
    
    def _parse_dir(self, data_dir):
        """Checks format of directory and loads metadata.
        
        Parameters
        ----------
        data_dir : str
            Path to directory to consider.
        """
        # set has extra feats
        # set ids
        # set metadata
        # check if has system feats
        if not os.path.exists(data_dir):
            raise FileNotFoundError('`data_dir` does not exist.')
        
        if not os.path.exists(data_dir+'metadata.csv'):
            raise FileNotFoundError('No metadata found. Ensure that `data_dir` was created by the create_dataset method.')
        
        # parse metadata
        expected_columns = ['num_nodes', 'num_edges', 'shard_size']
        metadata = pandas.read_csv(data_dir+'metadata.csv', index_col=0)
        assert list(metadata.columns) == expected_columns and metadata.index.name == 'shard_num',\
            "Metadata file does not appear to be the correct format."
        metadata['num_nodes'] = metadata['num_nodes'].apply(ast.literal_eval)
        metadata['num_edges'] = metadata['num_edges'].apply(ast.literal_eval)
        self._metadata = metadata
        
        # other properties
        self._has_system_features = os.path.exists(data_dir+'feats')
        self._has_split_ids = os.path.exists(data_dir+'split_ids')
        
        ids = [numpy.load(data_dir+f'ids/shard-{i}.npy').reshape(-1) for i in range(self.num_shards)]
        shard_sizes = [len(id_) for id_ in ids]
        if not numpy.array_equal(numpy.array(shard_sizes), self.metadata['shard_size'].values):
            raise ValueError('shard sizes in metadata and on file do not match')
        if len(numpy.unique(shard_sizes)) > 2:
            raise ValueError('shards have inconsistant sizes. All should be the same size besides potentially the last.')
        for i, shard_size in enumerate(shard_sizes):
            if i+1 < len(shard_sizes):
                assert shard_size == shard_sizes[0], f"Shard {i} does not have the correct shard size"
            else:
                assert shard_size <= shard_sizes[0], f"Shard {i} does not have the correct shard size"
        
        if not numpy.concatenate(ids).size == numpy.unique(numpy.concatenate(ids)).size:
            raise ValueError('Non unique ids')
        # we need to pad ids potentially
        if len(numpy.unique(shard_sizes)) == 2:
            difference = shard_sizes[0] - shard_sizes[-1]
            logging.debug(f"{type(self).__module__+'.'+type(self).__name__}:Final shard is {difference} examples short of the expected shard size.")
            ids[-1] = numpy.append(ids[-1], [None]*difference)
        self._ids = numpy.array(ids)
        return
    
    @property
    def metadata(self):
        """DataFrame : metadata for the dataset"""
        return self._metadata
    
    @property
    def graph_report(self):
        """DataFrame: report on the graph structure for each example."""
        ids = self.flat_ids
        N_nodes = numpy.concatenate(self.metadata['num_nodes'].values, axis=0)
        N_edges = numpy.concatenate(self.metadata['num_edges'].values, axis=0)
        df = pandas.DataFrame(
            data=numpy.array([ids, N_nodes, N_edges]).T,
            columns = ['id', 'num_nodes', 'num_edges'])
        if self.has_split_ids:
            df['split_ids'] = self.split_ids
        return df
        
    
    @property
    def has_system_features(self):
        """bool : whether the dataset examples have system level features"""
        return self._has_system_features
    
    @property
    def has_split_ids(self):
        """bool : whether the dataset examples have split ids"""
        return self._has_split_ids
    
    @property
    def data_dir(self):
        """str : path to directory containing dataset."""
        return self._data_dir

    @property
    def num_shards(self):
        """int : total number of shards"""
        return len(self.metadata)
    
    @property
    def shard_size(self):
        """int : number of examples in each shard
        
        Notes
        -----
        The last shard may contain fewer examples than this number.
        """
        return self.metadata['shard_size'][0]
    
    @property
    def ids(self):
        """ndarray of shape (num_shards, shard_size)"""
        return self._ids
    
    @property
    def flat_ids(self):
        """ndarray of len self.size"""
        ids_ = self.ids.flatten()
        ids_ = numpy.array([id_ for id_ in ids_ if id_ != None])
        return ids_
    
    @property
    def split_ids(self):
        if self.has_split_ids:
            split_ids = numpy.concatenate([numpy.load(self.data_dir+f'split_ids/shard-{i}.npy').reshape(-1) for i in range(self.num_shards)])
        else:
            split_ids = None
        return split_ids
    
    @property
    def size(self):
        return self.metadata['shard_size'].sum()
    
    def _get_id_location(self, id):
        """Get the location of an example with a specific id.
        
        Parameters
        ----------
        id : int or str
            The identifier for the example requested.
             
        Returns
        -------
        (shard_num, index_in_shard)
        """
        dtype = type(self.ids[0][0])
        try:
            id = dtype(id)
        except:
            raise
        location = numpy.argwhere(self.ids == id)
        if len(location) > 1:
            raise RuntimeError('Critical error, dataset corrupted. Unique id exists in multiple locations.')
        elif len(location) < 1:
            raise ValueError(f"id {id} does not exist in the dataset.")
        return tuple(location[0])
    
    @classmethod
    def create_dataset(cls, shard_generator, data_dir, overwrite: bool = False):
        """Save dataset to file by supplying shards of information.
        
        Parameters
        ----------
        shard_generator : iterable
            Must yield (A, F, E, y, ids, feats, split_ids). See below
        data_dir : str
            Path to directory to save data. Must be empty or not yet created unless overwrite.
        overwrite : bool, default False
            Whether to overwrite any existing data in data_dir
        
        Returns
        -------
        dampn.data.Dataset
        
        Notes
        -----
        S     := number of examples in shard
        Ni    := number of nodes in example i
        Mi    := number of edges in example i
        d     := size of node features
        l     := size of edge features
        T     := target size
        Z (z) := Z has shape z
        
        The generator must yield positionally:
        A := [adjacency matrices Ai for each example i] (S,)
            Ai (Mi, 2) - node index pairings
        F := [node feature matrices Fi for each example i] (S,)
            Fi (Ni, d) - d features for each node in Ni
        E := [edge feature matrices Ei for each example i] (S,)
            Ei (Mi, l) - l features for each edge in Ei
        y := [target matrices yi for each example i] (S,)
            yi (1, T) - target values for example i
        ids := [identifiers each example i] (S,)
        feats := [additional feature matrices for each example i] (S,) or None
            exi (1, G) where G is an the number of features
        split_ids := [indicator of structure for splitting for each i] (S,) or None
            
        """
        # get the number of nodes, edges, save to metadata
        # save shard
        if type(data_dir) != str:
            raise ValueError('`data_dir` must be str')
        if not data_dir.endswith('/'):
            data_dir += '/'
        logging.info(f"{cls.__module__+'.'+cls.__name__}:Creation dataset at {data_dir}...")
        metadata = pandas.DataFrame(columns=['shard_num', 'num_nodes', 'num_edges', 'shard_size'])
        
        if os.path.exists(data_dir):
            if overwrite:
                shutil.rmtree(data_dir)
            else:
                raise RuntimeError(f'{data_dir} already exists and `overwrite` was not specified.')
        os.makedirs(data_dir)
        os.mkdir(data_dir+'A')
        os.mkdir(data_dir+'F')
        os.mkdir(data_dir+'E')
        os.mkdir(data_dir+'y')
        os.mkdir(data_dir+'ids')
        os.mkdir(data_dir+'feats')
        os.mkdir(data_dir+'split_ids')
            
        for i, shard in enumerate(shard_generator):
            A, F, E, y, ids, feats, split_ids = shard
            # check all of the shapes incoming
            shard_size = len(ids)
            if y is None:
                y = numpy.empty((shard_size,1))
                logging.debug(f"{cls.__module__+'.'+cls.__name__}:Shard {i} appears to have no targets.")
            assert all([len(item) == shard_size for item in [A, F, E, y]]), "shard size inconsistant"
            if feats is not None:
                assert len(feats) == shard_size, "shard size inconsistant"
            else:
                logging.debug(f"{cls.__module__+'.'+cls.__name__}:Shard {i} appears to have no system level features.")
            if split_ids is not None:
                assert len(split_ids) == shard_size, "shard size inconsistant"
            else:
                logging.debug(f"{cls.__module__+'.'+cls.__name__}:Shard {i} appears to have no split_ids.")
        
            num_nodes = [len(f) for f in F]
            num_edges = [len(a) for a in A]
            assert [len(e) for e in E] == num_edges, f"adjacency matrix sizes {num_edges} and edge features sizes {[len(e) for e in E]} incompatable"
            metadata = metadata.append({'shard_num': i,'num_nodes': num_nodes, 'num_edges': num_edges, 'shard_size': shard_size}, ignore_index=True)
            cls._save_shard(data_dir, i, A, F, E, y, ids, feats=feats, split_ids=split_ids)
            logging.info(f"{cls.__module__+'.'+cls.__name__}:Shard {i} of size {shard_size} saved.")
        # remove feats and shard ids if we dont not save any
        if len(os.listdir(data_dir+'feats')) == 0:
            shutil.rmtree(data_dir+'feats')
            logging.debug(f"{cls.__module__+'.'+cls.__name__}:No system level features passed for this dataset.")
        if len(os.listdir(data_dir+'split_ids')) == 0:
            shutil.rmtree(data_dir+'split_ids')
            logging.debug(f"{cls.__module__+'.'+cls.__name__}:No split_ids passed for this dataset.")
            
        metadata.set_index('shard_num', drop=True, inplace=True)
        metadata.to_csv(data_dir+'metadata.csv')
        logging.info(f"{cls.__module__+'.'+cls.__name__}:Dataset creation at {data_dir} successful")
        return cls(data_dir)
    
    @staticmethod
    def _save_shard(data_dir, shard_num, A, F, E, y, ids, feats=None, split_ids=None):
        """Save shard to file.
        
        Not meant to be interacted with directly, used by the `create_dataset` method.
        
        Parameters
        ----------
        data_dir : str
            Directory to save shard
        shard_num : int
            id for this shard
        A : list of adjacency matrices Ai for each example i
        F : list of node feature matrices Fi for each example i
        E : list of edge feature matrices Ei for each example i
        y : list of target matrices yi for each example i
        ids : list of identifiers each example i
        feats : list of additional feature matrices for each example i
        split_ids : indicators for splitting, optional
        
        See `create_dataset`.
        """
        A = numpy.vstack(A).astype(int)
        numpy.save(data_dir+f'A/shard-{shard_num}.npy', A)
        F = numpy.vstack(F)
        numpy.save(data_dir+f'F/shard-{shard_num}.npy', F)
        E = numpy.vstack(E)
        numpy.save(data_dir+f'E/shard-{shard_num}.npy', E)
        y = numpy.vstack(y)
        numpy.save(data_dir+f'y/shard-{shard_num}.npy', y)
        ids = numpy.vstack(ids)
        numpy.save(data_dir+f'ids/shard-{shard_num}.npy', ids)
        if feats is not None:
            feats = numpy.vstack(feats)
            numpy.save(data_dir+f'feats/shard-{shard_num}.npy', feats)
        if split_ids is not None:
            split_ids = numpy.vstack(split_ids)
            numpy.save(data_dir+f'split_ids/shard-{shard_num}.npy', split_ids)
            
        return
    
    def _load_shard_matrices(self, shard_num):
        """Load the data from one shard into memory.
        
        Not meant to be interacted with directly, returns convoluded matrices.
        
        Parameters
        ----------
        shard_num : int
            Which shard to load.
            
        Returns
        -------
        Ac : ndarray of concatenated adjacency matrices in shard
        Fc : ndarray of concatenated node feature matrices in shard
        Ec : ndarray of concatenated edge feature matrices in shard
        yc : ndarray of target value matrices in shard
        idsc : ndarray of example identifiers in shard
        (optional) featsc : ndarray of system level features in shard
        """
        Ac = numpy.load(self.data_dir+f'A/shard-{shard_num}.npy')
        Fc = numpy.load(self.data_dir+f'F/shard-{shard_num}.npy')
        Ec = numpy.load(self.data_dir+f'E/shard-{shard_num}.npy')
        yc = numpy.load(self.data_dir+f'y/shard-{shard_num}.npy')
        idsc = numpy.load(self.data_dir+f'ids/shard-{shard_num}.npy')
        if self.has_system_features:
            featsc = numpy.load(self.data_dir+f'feats/shard-{shard_num}.npy')
        else:
            featsc = None
        if self.has_split_ids:
            split_ids = numpy.load(self.data_dir+f'split_ids/shard-{shard_num}.npy')
        else:
            split_ids = None
        return Ac, Fc, Ec, yc, idsc, featsc, split_ids
    
    def _get_example_from_shard(self, location, shard=None, return_shard=False):
        """Load the data from one location into memory,.
        
        Parameters
        ----------
        location : tuple of int
            Shard number and example number in shard to load
        shard : tuple of array, optional
            Use passed shar instead of loading it. In this case location[0] eg shard_num is ignored
        return_shard : bool, default False
            If true, return the entire shard as the first return, and the extraced matrices
            as second return tuple. Useful in combination with `shard` to extract multiple examples
            from the same shard.
        Returns
        -------
        Ai : ndarray adjacency matricex
        Fi : ndarray node feature matrices
        Ei : ndarray edge feature matrices
        yi : target values
        idsi : example identifier
        featsi : ndarray of system level features
        split_idsi : ndarray of slit identifiers
        """
        if shard is None:
            shard = self._load_shard_matrices(location[0])
        else:
            pass
        
        # now we must determine the position in the matrices of the data
        # this will require: number of edges, sum number of edges for al, previous structures
        # and the same for nodes
        N_list = self.metadata.loc[location[0], 'num_nodes']
        N_previous = int(numpy.sum(N_list[:location[1]]))
        Ni = N_list[location[1]]
        
        M_list = self.metadata.loc[location[0], 'num_edges']
        M_previous = int(numpy.sum(M_list[:location[1]]))
        Mi = M_list[location[1]]
        
        # now get the matrices
        Ac, Fc, Ec, yc, idsc, featsc, split_idsc = shard
        Ai = Ac[M_previous:M_previous+Mi, :]
        Ei = Ec[M_previous:M_previous+Mi, :]
        Fi = Fc[N_previous:N_previous+Ni, :]
        yi = yc[location[1]]
        id = idsc[location[1]]
        if split_idsc is None:
            split_idsi = None
        else:
            split_idsi = split_idsc[location[1]]
        if featsc is None:
            featsi = None
        else:
            featsi = featsc[location[1]]
        
        if return_shard:
            return shard, (Ai, Fi, Ei, yi, id, featsi, split_idsi)
        else:
            return Ai, Fi, Ei, yi, id, featsi, split_idsi
    
    def select(self, data_dir, ids=None, indexes=None, shard_size=None, **kwargs):
        """Select specific examples in the data.
        
        Creates a new dataset of selected ids.
        
        Parameters
        ----------
        data_dir : str
            directory to save selected data
        ids : ndarray, optional
            ids to take
        indexes : ndarray of int, optional
            ids to take
        shard_size : int, optional
            shard size for selected dataset. default to shard size of current dataset.
        Returns
        -------
        dampn.data.Dataset
        """
        if shard_size is None:
            shard_size = self.shard_size
        
        assert bool(ids is not None) != bool(indexes is not None), "One of `indexes` or `ids` must be passed, exclusive."
        
        if ids is None:
            locations = []
            for index in indexes:
                shard_num = index // self.shard_size
                position = index % self.shard_size
                location = (shard_num, position)
                locations.append(location)
        else:
            ids = numpy.array(ids).reshape(-1)
            # get the location of each index specified
            # this is shape (num ids, 2). first column is shard number, second is position in shard
            locations = numpy.array([self._get_id_location(id) for id in ids])
        
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Selecting {len(locations)} examples for the creation of a new dataset at {data_dir} using shard sizes of {shard_size}.")
        
        # loop through each desired datapoint, try to save time
        # by only loading new shard if needed
        def shard_generator():
            A, F, E, y, ids, feats, split_ids = [], [], [], [], [], [], []
            last_shard_num = None 
            shard=None
            for i, location in enumerate(locations):
                shard_num, position = location
                if last_shard_num == shard_num:
                    shard, (Ai, Fi, Ei, yi, id, featsi, split_ids_i) = self._get_example_from_shard(
                        location, shard=shard, return_shard=True)
                else:
                    shard, (Ai, Fi, Ei, yi, id, featsi, split_ids_i) = self._get_example_from_shard(
                        location, shard=None, return_shard=True)
                    last_shard_num = location[0]
                A.append(Ai)
                F.append(Fi)
                E.append(Ei)
                y.append(yi)
                ids.append(id)
                feats.append(featsi)
                split_ids.append(split_ids_i)

                if len(A) == shard_size or i+1 == len(locations):
                    # patch over feats if all are None
                    if numpy.isnan(numpy.array(feats).astype(float)).all():
                        feats = None
                    if (numpy.array(split_ids)==None).all():
                        split_ids = None
                    yield A, F, E, y, ids, feats, split_ids
                    A, F, E, y, ids, feats, split_ids = [], [], [], [], [], [], []
                    continue
            
        return Dataset.create_dataset(shard_generator(), data_dir, **kwargs)
    
    def __getitem__(self, index):
        """Return the data at a specific index, in graph form.
        
        Parameters
        ----------
        index : int
            position in dataset
        """
        if type(index) == int:
            shard_num = index // self.shard_size
            position = index % self.shard_size
            location = (shard_num, position)
        elif type(index) == tuple:
            location = index
        else:
            location = self._get_id_location(index)
        Ai, Fi, Ei, yi, id, featsi, split_idsi = self._get_example_from_shard(location)
        
        graph = dgl.graph(tuple(Ai.T), num_nodes=len(Fi))
        graph.ndata['f'] = torch.from_numpy(Fi)
        graph.edata['e'] = torch.from_numpy(Ei)
        return graph, yi, id, featsi, split_idsi
    
    def batch_generator(self, batch_size: int):
        """Batch DGL graphs into an iterator.
        
        Parameters
        ----------
        batch_size : int
            Number of examples per batch in iterator
        
        Returns
        -------
        generator of dgl.Graph
        """
        def generator():
            # startup variables
            dgl_graphs = []
            ys = []
            
            # loop shard wise
            for shard_num, shard_size in enumerate(self.metadata['shard_size']):
                # when we first start the shard we want to load a new one, set to None
                # the shard will be kept in memory as we loop through examples after
                # this
                shard = None
                
                # loop through iexamples in shard
                for example_num in range(shard_size):
                    shard, (Ai, Fi, Ei, yi, id, featsi) = self._get_example_from_shard(
                        (shard_num, example_num), shard=shard, return_shard=True)
                    dgl_graph =  dgl.graph(tuple(Ai.T), num_nodes=len(Fi))
                    dgl_graph.ndata['f'] = torch.from_numpy(Fi)
                    dgl_graph.edata['e'] = torch.from_numpy(Ei)
                    dgl_graphs.append(dgl_graph)
                    ys.append(yi)
                    
                    # if we have a batch yield it and reset counter
                    if len(ys) == batch_size:
                        yield dgl_graphs, ys
                        dgl_graphs = []
                        ys = []
        return generator()