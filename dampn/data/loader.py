import os

import dampn.base
import dampn.data.dataset
import dampn.features.structure_featurizer

import pandas
import numpy

import logging
logger = logging.getLogger(__name__)


class BaseStructureLoader:
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
    stoichiometry_split : bool, default True
        whether to pass the stoichiometry of the structure as a split id for dataset
        splitting
    **kwargs : passed to featurizer construction if featurizer not given
    """
    def __init__(
        self,
        source_dir: str,
        data_dir: str,
        featurizer: dampn.features.structure_featurizer.StructureFeaturizer = None,
        stoichiometry_split: bool = True,
        **kwargs
    ):
        if featurizer is None:
            featurizer = dampn.features.structure_featurizer.StructureFeaturizer(**kwargs)
        self.featurizer = featurizer
        self._source_info = None
        self.check_source(source_dir)
        self.data_dir = data_dir
        self.partition = {}
        self.stoichiometry_split = stoichiometry_split
        return
    
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

                self._source_info = source_info
            else:
                logging.debug(f"{type(self).__module__+'.'+type(self).__name__}:Unexpected file {file}, ignoring.")
        
        if self.__class__ is BaseStructureLoader: 
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
        if self.stoichiometry_split:
            split_ids = [struc.atom_string for struc in structures]
            return A_shard, F_shard, E_shard, y_shard, ids_shard, None, split_ids
        else:
            return A_shard, F_shard, E_shard, y_shard, ids_shard, None, None
    
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

    
class StructurePropertyLoader(BaseStructureLoader):
    """Load a directory of structures into a dataset.
    
    Parameters
    ----------
    source_dir : str
        path to directory containing structure files
    data_dir : str
        path to new directory to save formatted machine ready data
    property : str
        name of field store in atomic strucutre file to consider the target
    featurizer : dampn.features.structure_featurizer.StructureFeaturizer, optional
        featurizer to apply to each structure. Creates new featurizer if not passed.
    stoichiometry_split : bool, default True
        whether to pass the stoichiometry of the structure as a split id for dataset
        splitting
    **kwargs : passed to featurizer construction if featurizer not given
    """
    def __init__(
        self,
        source_dir: str,
        data_dir: str,
        property: str,
        featurizer: dampn.features.structure_featurizer.StructureFeaturizer = None,
        stoichiometry_split: bool = True,
        **kwargs
    ):
        super().__init__(
            source_dir=source_dir,
            data_dir=data_dir,
            featurizer=featurizer,
            stoichiometry_split=stoichiometry_split,
            **kwargs)
        self.property = property
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
            try:
                target = numpy.array(structure.info[self.property]).reshape(1,-1)
            except KeyError:
                raise ValueError(
                    f"File `{structure_file} does not appear to have the desired property `{self.property}`. Properties in this structure are {structure.info}"
                )
            structures.append(structure)
            ids.append(id_)
            targets.append(target)
        return structures, targets, ids
        
class BinaryTSClassificationLoader(BaseStructureLoader):
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
        super().check_source(source_dir)
                
        if self._source_info is None:
            raise ValueError('Expected file "source_info.csv"')
        if not len(self._structure_list) == len(self._source_info):
            raise ValueError('Number of structures does not equal data in source_info.csv')
        
        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Data source directory {source_dir} valid for loading.")
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
                                 
        
    