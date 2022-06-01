"""Split a dataset.


Example
-------
>>>dataset = Dataset('./dataset')
>>>dataset.size
10
>>>splitter = RandomSplitter(split_frac=0.2)
>>>train, test = splitter.split(dataset, data_dir='./split_data')
>>>train.data_dir
"./split_data/train"
>>>test.size
2
"""
import numpy
import dampn.data.dataset

from typing import Union, Iterable

import logging
logger = logging.getLogger(__name__)

class DataSplitter:
    """Abstract parent.
    
    Method `select_indexes` must be defined by child.
    
    Attributes
    ----------
    split_func : function used to split the dataset based on input
        currently train test or k fold split
    """
    def __init__(self, split_frac: float = None, k: int = None):
        """Splits dataset based on specified fractions or folds.
        
        Parameters
        ----------
        split_frac : float, optional
            Specifies testing fraction
        k : int, optional
            If passed, k fold split the training set.
        """
        assert bool(split_frac) != bool(k), "one of k or split_frac must be passed"
        self.split_frac = None
        self.k = None
        
        if split_frac:
            assert (split_frac < 1.0) and (split_frac > 0.0), "`split_frac` must be between 0 and 1"
            self.split_frac = split_frac
        else :
            assert type(k) == int, "k must be int"
            self.k = k
        return
    
    @property
    def split_func(self):
        if self.k:
            logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Splitting into {self.k} folds.")
            return self._k_fold_split
        else:
            logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Splitting {self.split_frac*100}% into test set.")
            return self._train_test_split
    
    def select_N(self,
        dataset: dampn.data.dataset.Dataset,
        N: int,
        already_selected: Iterable = None
    ):
        """Get N examples from the remaining available.
        
        Parameters
        ----------
        dataset : dampn.data.Dataset
            Dataset to split
        N : int
            Number to select
        already_selected : iterable
            mask of indexes already selected
            
        Returns
        -------
        ndarray : indexes selected
        """
        if already_selected is None:
            already_selected = numpy.zeros(len(dataset)).astype(bool)

        if (N + sum(already_selected)) > len(dataset):
            logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Asked for {N} examples to be selected, but there are not enough samples left. Returning all remaining {int(len(dataset) - sum(already_selected))} samples.")
            return ~already_selected
        else:
            logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Selecting {N} examples to be selected from the remaining {int(len(dataset) - sum(already_selected))} samples.")
            return self._select_N(dataset, N, already_selected)
            
        return
        
        
    def _select_N(self,
        dataset: dampn.data.dataset.Dataset,
        N: int,
        already_selected: Iterable = None
    ):
        """Defines how to select N samples from a set.
        
        Must be deinfed by child
        
        Parameters
        ----------
        dataset : dampn.data.Dataset
            Dataset to split
        N : int
            Number to select
        already_selected : iterable
            mask of indexes already selected
            
        Returns
        -------
        ndarray : mask of indexes selected
        """
        raise NotImplemented()
        
    def _train_test_split(self, dataset: dampn.data.dataset.Dataset, data_dir: str, overwrite: bool = False):
        """Split the dataset into two sets.
        
        Parameters
        ----------
        dataset : Dataset to split
        data_dir : str
            new location to place data
        overwrite : bool
            whether to overwrite existing splits
        
        Returns
        -------
        (dataset, dataset) training and testing dataset
        """
        N_test = round(self.split_frac * len(dataset))
        
        test_mask = self.select_N(dataset, N_test)
        indexes = numpy.array(range(len(dataset)))
        test_indexes = indexes[test_mask]
        train_indexes = indexes[~test_mask]
        
        if not data_dir.endswith('/'):
            data_dir = data_dir+'/'
        
        train = dataset.select(indexes=train_indexes, data_dir=data_dir+'train', overwrite=overwrite)
        test = dataset.select(indexes=test_indexes, data_dir=data_dir+'test', overwrite=overwrite)
        
        # check resulting splits
        split_unique_ids_size = numpy.unique(
            numpy.concatenate(
                [train.flat_ids, test.flat_ids]
            )
        ).size
        assert split_unique_ids_size == train.size + test.size, "ids not split uniquely"
        assert split_unique_ids_size == dataset.size, "not all ids selected"
        return train, test
    
    def _k_fold_split(self, dataset: dampn.data.dataset.Dataset, data_dir: str, overwrite: bool = False):
        """Split the dataset into k folds.
        
        Parameters
        ----------
        dataset : Dataset to split
        data_dir : str
            new location to place data
        overwrite : bool
            whether to overwrite existing splits
        
        Returns
        -------
        (k datasets,) k dataset folds
        """
        already_selected = numpy.zeros(dataset.size).astype(bool)
        N_fold = round(len(dataset)/self.k)
        selected_indexes_list = []
        
        for i in range(self.k):
            selected_mask = self.select_N(dataset, N_fold, already_selected)
            already_selected += selected_mask
            assert sum(already_selected > 1) == 0, "selected an index that was already selected"
            
            indexes = numpy.array(range(len(dataset)))
            selected_indexes = indexes[selected_mask]
            selected_indexes_list.append(selected_indexes)

        logging.info(f"{type(self).__module__+'.'+type(self).__name__}:Selected indexes {selected_indexes_list}")
        assert numpy.unique(numpy.concatenate(selected_indexes_list)).size == dataset.size, "resulting partition of folds is not the same size as the original dataset"
        
        folds = []
        for i, indexes in enumerate(selected_indexes_list):
            folds.append(
                dataset.select(indexes=indexes, data_dir=data_dir+f'fold_{i}', overwrite=overwrite))
        
        split_unique_ids_size = numpy.unique(
            numpy.concatenate(
                [d.flat_ids for d in folds]
            )
        ).size
        assert split_unique_ids_size == sum([d.size for d in folds]), "ids not split uniquely"
        assert split_unique_ids_size == dataset.size, "not all ids selected"
        return folds
    
    def split(self, dataset: dampn.data.dataset.Dataset, data_dir: str, overwrite: bool = False):
        """Split the dataset.
        
        Parameters
        ----------
        dataset : Dataset to split
        data_dir : str
            new location to place data
        overwrite : bool
            whether to overwrite existing splits
        
        Returns
        -------
        (k datasets,) where k is the number of splits
        """
        return self.split_func(dataset, data_dir, overwrite=overwrite)
    
    
class RandomSplitter(DataSplitter):
    """Split dataset by random selection.
    
    Attributes
    ----------
    split_func : function used to split the dataset based on input
        currently train test or k fold split
    """
    def _select_N(self,
        dataset: dampn.data.dataset.Dataset,
        N: int,
        already_selected: Iterable = []
    ):
        """Selects N indexes from  a dataset randomly without replacement.
        
        Parameters
        ----------
        dataset : dampn.data.Dataset
            Dataset to split
        N : int
            Number to select
        already_selected : iterable
            indexes already selected
            
        Returns
        -------
        ndarray : mask of indexes selected
        """
        dataset_indexes = numpy.array(list(range(len(dataset))))
        available_indexes = dataset_indexes[~already_selected]
        chosen_indexes = numpy.random.choice(available_indexes, N, replace=False)
        selected = numpy.isin(dataset_indexes, chosen_indexes)
        return selected
        
    
        