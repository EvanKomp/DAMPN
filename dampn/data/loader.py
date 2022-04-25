import os

import dampn.base
import dampn.data.dataset
import dampn.features.structure_featurizer

import pandas
import numpy

import logging
logger = logging.getLogger(__name__)


class StructureLoader:
    """Load a directory of structures into a dataset.
    
    Abstract parent class.
    
    Parameters
    ----------
    source_dir : str
        path to directory containing structure files
    data_dir : str
        path to new directory to save formatted machine ready data
    featurizer : dampn.features.structure_featurizer.StructureFeaturizer, optional
        featurizer to apply to each structure. Creates new featurizer if not passed.
    **kwargs : passed to featurizer construction if featurizer not given
    """
    def __init__(
        self,
        source_dir: str,
        data_dir: str,
        featurizer: dampn.features.structure_featurizer.StructureFeaturizer = None,
        **kwargs
    ):
        if featurizer is None:
            featurizer = dampn.features.structure_featurizer.StructureFeaturizer(**kwargs)
        self.featurizer = featurizer
        self.check_source(source_dir)
        self.data_dir = data_dir
        self.partition = {}
        return
    
    def check_source(self, source_dir: str):
        """Checks is source data directtory is valid for this loader.
    
        Parameters
        ----------
        source_dir : str
            Path to directory to check.
        """
        raise NotImplemented()
        
    def _partition_data(self, shard_size: int):
        """Organize paths in source dir into shards.
        
        Parameters
        ----------
        shard_size : number of examples in each shard
        """
        raise NotImplemented()
    
    def _load_shard(self, shard_num: int):
        """Load a shard from the partition from file.
        
        Parameters
        ----------
        shard_num : int, which shard to load
        
        Returns
        -------
        tuple of list, first list is structures, second is targets, third is ids
        """
        raise NotImplemented()
        
    def featurize_shard(self, shard):
        """Featurize structures and produce shard matrices.
        
        Parameters
        ----------
        shard : tuple of list, first list is structures, second is targets, third is ids
        
        Returns
        -------
        tuple of list : 
            A_shard := [adjacency matrices Ai for each example i] (S,)
                Ai (Mi, 2) - node index pairings
            F_shard := [node feature matrices Fi for each example i] (S,)
                Fi (Ni, d) - d features for each node in Ni
            E_shard := [edge feature matrices Ei for each example i] (S,)
                Ei (Mi, l) - l features for each edge in Ei
            y_shard := [target matrices yi for each example i] (S,)
                yi (1, T) - target values for example i
            ids_shard := [identifiers each example i] (S,)
        S is shard size
        """
        structures, y_shard, ids_shard = shard
        A_shard, F_shard, E_shard = self.featurizer.featurize(structures)
        return A_shard, F_shard, E_shard, y_shard, ids_shard, None
    
    def load_dataset(self, shard_size: int = 128, overwrite: bool = False):
        """Create dataset from source files.
        
        Parameters
        ----------
        shard_size : number of examples in each shard
        overwrite : bool
            Whether or not to delete any data in destination path.
        
        Returns
        -------
        Dataset
        """
        self._partition_data(shard_size)
        
        def generator():
            for shard_num in self.partition.keys():
                shard_raw = self._load_shard(shard_num)
                shard_featurized = self.featurize_shard(shard_raw)
                yield shard_featurized
        
        return dampn.data.dataset.Dataset.create_dataset(generator(), self.data_dir, overwrite=overwrite)
        
        
class BinaryTSClassificationLoader(StructureLoader):
    """Load a directory of structures into a binary TS or not dataset.
    
    Parameters
    ----------
    source_dir : str
        path to directory containing structure files
    data_dir : str
        path to new directory to save formatted machine ready data
    featurizer : dampn.features.structure_featurizer.StructureFeaturizer, optional
        featurizer to apply to each structure. Creates new featurizer if not passed.
    **kwargs : passed to featurizer construction if featurizer not given
    """
    def check_source(self, source_dir: str):
        """Checks is source data directtory is valid for this loader.
    
        Parameters
        ----------
        source_dir : str
            Path to directory to check.
        """
        if not source_dir.endswith('/'):
            source_dir += '/'
            
        files_list = os.listdir(source_dir)
        logging.debug(f"{type(self).__module__+'.'+type(self).__name__}:Data source directory {source_dir} contains the following files: {files_list}")
        
        # We expect a bunch of structures and a csv file with info
        structure_list = []
        source_info = None
        for file in files_list:
            if file.endswith('xyz'):
                structure_list.append(file)
            elif file == "source_info.csv":
                source_info = pandas.read_csv(source_dir+"source_info.csv", index_col=0)
                if "use_as" not in source_info.columns:
                    raise ValueError("`use_as` not a column in source_info.csv for this dataset. It is required to determine if the target is TS or not.")

                self._source_info = source_info
            else:
                logging.debug(f"{type(self).__module__+'.'+type(self).__name__}:Unexpected file {file}, ignoring.")
                
        if source_info is None:
            raise ValueError('Expected file "source_info.csv"')
        if not len(structure_list) == len(source_info):
            raise ValueError('Number of structures does not equal data in source_info.csv')
        
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Data source directory {source_dir} valid for loading.")
        self.source_dir = source_dir
        self._structure_list = structure_list
        return
    
    def _partition_data(self, shard_size: int):
        """Organize paths in source dir into shards.
        
        Parameters
        ----------
        shard_size : number of examples in each shard
        """
        chunks = [
            self._structure_list[i:i + shard_size] for i in range(
                0, len(self._structure_list), shard_size
            )]
        for i, chunk in enumerate(chunks):
            self.partition[i] = chunk
        logging.debug(f"{type(self).__module__+'.'+type(self).__name__}:Files partitioned into chunks of size {shard_size}. Resulting partition: {self.partition}")
        return
    
    def _load_shard(self, shard_num: int):
        """Load a shard from the partition from file.
        
        Parameters
        ----------
        shard_num : int, which shard to load
        
        Returns
        -------
        tuple of list, first list is structures, second is targets, third is ids
        """
        structure_files = self.partition[shard_num]
        structures = []
        ids = []
        targets = []
        
        for structure_file in structure_files:
            structure = dampn.base.Structure.load(self.source_dir+structure_file)
            id_ = int(structure_file.split('.')[0])
            target = self._source_info.loc[id_, 'use_as']
            target = numpy.array(target == 'tstate').astype(int).reshape(1,1)
            structures.append(structure)
            ids.append(id_)
            targets.append(target)
        return structures, targets, ids
                                 
        
    