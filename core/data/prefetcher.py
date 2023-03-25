''' prefetch dataloader '''
import queue as Queue
import threading
import torch
from torch.utils.data import DataLoader


class PrefetchGenerator(threading.Thread):
    """A general prefetch generator.

    Ref:
    https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.
    """

    def __init__(self, generator, num_prefetch_queue):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Ref:
    https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    TODO:
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.
    """

    def __init__(self, num_prefetch_queue, **kwargs):
        self.num_prefetch_queue = num_prefetch_queue
        super().__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CPUPrefetcher():
    """ CPU prefetcher.  """
    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        ''' get next data '''
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        ''' reset '''
        self.loader = iter(self.ori_loader)


class CUDAPrefetcher():
    """ CUDA prefetcher. """
    def __init__(self, loader, opt):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.preload()

    def preload(self):
        ''' pre load '''
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)
        return None

    def next(self):
        ''' get next data '''
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        ''' reset dataloader '''
        self.loader = iter(self.ori_loader)
        self.preload()
