from typing import List

import numpy as np
from torch.utils.data import Dataset

from partition.utils import IndexedSubset


class CentralizedPartition:
    def __init__(self):
        pass

    def __call__(self, dataset) -> List[Dataset]:
        total_num = len(dataset)
        idxs = list(range(total_num))
        dataset_ref = dataset
        return [
            IndexedSubset(
                dataset_ref,
                indices=idxs,
            )
        ]
