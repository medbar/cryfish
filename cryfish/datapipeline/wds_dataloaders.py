import torch
import time
import webdataset as wds

from webdataset.filters import pipelinefilter
from webdataset import WebLoader, DataPipeline

from typing import Union, List
from tqdm import tqdm

from collections import defaultdict


def build_dataloder(datapipeline: Union[List, DataPipeline], **kwargs):
    # build slightly extented pytorch Dataloader
    # for p in datapipeline:
    #    print(type(p), isinstance(p, torch.utils.data.IterableDataset))
    if not isinstance(datapipeline, DataPipeline):
        datapipeline = DataPipeline(*datapipeline)
    return WebLoader(datapipeline, **kwargs)


def _multiworker_head_by_dataloader(data, **kwargs):
    """apply parrallel computing for all stages before that"""
    dl = WebLoader(data, **kwargs)
    for e in dl:
        yield e


multiworker_head_by_dataloader = pipelinefilter(_multiworker_head_by_dataloader)


def test_dataloader(dl):
    keys2total_len = defaultdict(int)
    start = time.time()
    num_paddings = 0
    total_seq_len = 0
    batch_size = 0
    for i, b in enumerate(tqdm(dl)):
        batch_size += len(b["__key__"])
        for k, v in b.items():
            keys2total_len[k] += getattr(v, "__len__", lambda: 0)()
            if k.endswith("_attention_mask.pth"):
                assert v.dtype == torch.bool, v.dtype
                total_seq_len += v.sum() / (16 if k.startswith("audio_feats") else 1)
                num_paddings += (~v).sum() / (16 if k.startswith("audio_feats") else 1)
    et = time.time() - start
    print(
        f"The number of batches is {i}. {keys2total_len=}.\nElapsed time is {et}s, \n{keys2total_len['__key__']/et}it/s\n{i/et}batch/s"
    )
    print(
        f"{total_seq_len=}, {num_paddings=}, {num_paddings/total_seq_len}, avg batch size is {batch_size/i}"
    )
