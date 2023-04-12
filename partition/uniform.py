from typing import List

import numpy as np
from torch.utils.data import Dataset

from partition.utils import IndexedSubset


class UniformPartition:
    def __init__(
            self,
            num_clients: int,
            num_class: int = 10,
    ):
        self.num_clients = num_clients
        self.num_class = num_class

    def __call__(self, dataset) -> List[Dataset]:
        total_num = len(dataset)
        idxs = np.random.permutation(total_num)
        partitioned_idxs = np.array_split(idxs, self.num_clients)
        net_dataidx_map = {i: partitioned_idxs[i] for i in range(self.num_clients)}
        dataset_ref = dataset
        return [
            IndexedSubset(
                dataset_ref,
                indices=net_dataidx_map[i],
            )
            for i in range(self.num_clients)
        ]
