import logging


from tqdm import tqdm
from pathlib import Path

from typing import Union, List, Any
import webdataset as wds
from webdataset.shardlists import expand_urls
import random
import time

from torch.utils.data import DataLoader, IterableDataset

from datapipeline.wds_io import SimpleAnnoJsonLoader, write_as_sharded_wds
from datapipeline.wds_dataloaders import multiworker_head_by_dataloader


def run(pipeline: Union[List[Any], wds.DataPipeline], limit=-1):
    if not isinstance(pipeline, wds.DataPipeline):
        pipeline = wds.DataPipeline(*pipeline)
    pipeline_keys = set()
    num_samples = 0
    for i, e in enumerate(tqdm(pipeline)):
        num_samples += 1
        pipeline_keys.update(e.keys())
        if i == limit:
            logging.info(f"Reached limit {limit}, stoping iterator")
            break
    logging.info(
        f"Pipeline done! Total number of items is {num_samples}. Processed keys is {pipeline_keys}"
    )


def run_and_shardsave(pipeline, out_printf_frmt, max_elements_per_shard=50, limit=-1):
    Path(out_printf_frmt).parent.mkdir(parents=True, exist_ok=True)
    with wds.ShardWriter(out_printf_frmt, maxcount=max_elements_per_shard) as sink:
        for i, e in enumerate(tqdm(pipeline)):
            assert "__key__" in e, f"Bad dataset format {e.keys()}"
            sink.write(e)
            if i == limit:
                logging.info(f"Reached limit {limit}, stoping iterator")
                break
    logging.info(f"total number of items: {i}")


class InfiniteShardList(IterableDataset):
    """An Infenete iterable dataset yielding a list of URLs."""

    def __init__(self, urls, limit=float("-inf"), seed=None):
        """Initialize the SimpleShardList.

        Args:
            urls (str or List[str]): A list of URLs as a Python list or brace notation string.
            lomit (int or float): Iterating untill limit
            seed (int or bool or None): Random seed for shuffling; if None, no shuffling is done,
                if True, a random seed is generated.
        """
        super().__init__()
        if isinstance(urls, str):
            urls = expand_urls(urls)
        else:
            urls = list(urls)
        self.urls = urls
        self.limit = limit
        assert isinstance(self.urls[0], str)
        if seed is True:
            logging.warning("Dataloader num_worker > 1  cannot work with seed=True")
            seed = time.time()
        self.seed = seed
        self.epoch = -1

    def __len__(self):
        """Return the number of URLs in the list.

        Returns:
            int: The number of URLs.
        """
        return self.limit

    def __iter__(self):
        """Return an iterator over the shards.

        Yields:
            dict: A dictionary containing the URL of each shard.
        """
        i = 0
        while True:
            self.epoch += 1
            urls = self.urls.copy()
            if self.seed is not None:
                seed = self.seed + self.epoch
                logging.info(f"Seed for the {self.epoch} epoch is {seed}")
                random.Random(seed).shuffle(urls)
            for j, url in enumerate(urls):
                # TODO __epoch__ doesn't work
                yield {"url": url, "__epoch__": self.epoch}
                i += 1
                if i == self.limit:
                    logging.debug(
                        f"Reaching limit at {self.epoch} epoch, {j} of {len(self.urls)} shards are processed"
                    )
                    return
