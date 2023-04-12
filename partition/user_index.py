

from typing import List

import numpy as np
from torch.utils.data import Dataset

from partition.utils import IndexedSubset


class UserPartition:
    def __init__(
            self, user_idxs
    ):
        self.user_idx = user_idxs

    def __call__(self, dataset) -> List[Dataset]:
        dataset_ref = dataset
        return [
            IndexedSubset(
                dataset_ref,
                indices=v,
            )
            for _, v in self.user_idx.items()
        ]
