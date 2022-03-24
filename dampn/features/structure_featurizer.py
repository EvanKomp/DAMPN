import dampn.base

from typing import List

import logging
logger = logging.getLogger(__name__)

class StructureFeaturizer:
    
    def __init__(
        self,
        atoms: List[str] = ['H','C','N','O'],
        atom_features: List[str] = ['identity']
        distance_features: str = 'inverse'
    
    ):