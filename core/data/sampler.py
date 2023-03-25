''' data sampler '''
import math
import random
import torch
from torch.utils.data.sampler import Sampler, BatchSampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """
    # pylint: disable=super-init-not-called
    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        ''' set epoch '''
        self.epoch = epoch


class MutilScaleBatchSampler(BatchSampler):
    ''' MutilScaleBatchSampler '''
    def __init__(self, sampler: Sampler[int], scales: list, batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.scales = scales

    def _update_scale(self):
        return self.scales[random.randrange(0, len(self.scales))]

    def __iter__(self):
        batch, scale = [], self._update_scale()
        for idx in self.sampler:
            batch.append((idx, scale))
            if len(batch) == self.batch_size:
                yield batch
                scale = self._update_scale()
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


class ContinuousScaleBatchSampler(BatchSampler):
    ''' ContinuousScaleBatchSampler '''
    def __init__(self, sampler: Sampler[int], min_scale: float, max_scale: float, batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def _update_scale(self):
        range = self.max_scale - self.min_scale
        return (random.random() * range + self.min_scale, random.random() * range + self.min_scale)

    def __iter__(self):
        batch, scale = [], self._update_scale()
        for idx in self.sampler:
            batch.append((idx, scale))
            if len(batch) == self.batch_size:
                yield batch
                scale = self._update_scale()
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
