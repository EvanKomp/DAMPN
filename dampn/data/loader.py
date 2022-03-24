import dampn.base
import dampn.data.dataset

import logging
logger = logging.getLogger(__name__)


class StructureLoader:
    """"""
    def __init__(self, source_dir: str):
        
        self.source_dir = source_dir
        return