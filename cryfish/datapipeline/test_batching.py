from wds_batching import (
    _bucketing_batching,
    _unbatching_padded,
    _batching_constant_batch_size,
)
import torch
import random
import webdataset as wds
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def random_element(seed):
    random.seed(seed)
    af = random.randint(1, 400)
    pt = [random.randint(1, 100), random.randint(1, 100)]
    tt = random.randint(1, 100)
    return {
        "__key__": f"{seed}",
        "audio_feats.pth": torch.ones((af, 4)) * af,
        "lprompt_tokens_ids.pth": torch.ones((pt[0],)) * pt[0],
        "rprompt_tokens_ids.pth": torch.ones((pt[1],)) * pt[1],
        "text_tokens_ids.pth": torch.ones((tt,)) * tt,
    }


def inspect_batches(batches, elements=None):
    batches = list(batches)
    num_elements = 0
    num_paddings = 0
    batch_size = 0
    for b in batches:
        batch_size += len(b["__key__"])
        for k, v in b.items():
            if k.endswith("_attention_mask.pth"):
                assert v.dtype == torch.bool, v.dtype
                num_elements += v.sum()
                num_paddings += (~v).sum()
    print(
        f"{num_elements=}, {num_paddings=}, {num_paddings/num_elements}, {batch_size/len(batches)=}"
    )
    print(f"{[b['__key__'] for b in batches[-4:]]=}")

    if elements is None:
        return
    es2 = [e for e in _unbatching_padded(batches)]
    assert len(elements) == len(es2), f"{len(elements)} != {len(es2)}"

    for e1, e2 in zip(
        sorted(elements, key=lambda x: x["__key__"]),
        sorted(es2, key=lambda x: x["__key__"]),
    ):
        for k, v in e1.items():
            v2 = e2[k]
            if isinstance(v, str):
                assert v == v2, f"{k} {v} {v2}"
            else:
                assert torch.allclose(v, v2), f"{k} {v} {v2}"


def test_batching():
    elements = [*map(random_element, range(1000))]
    batches = [
        b
        for b in _bucketing_batching(
            elements,
            batch_min_tokens=3200,
            num_buckets=20,
            collate_fn_kwargs={"pad_token_id": -100},
        )
    ]
    # batches = [b for b in _batching_constant_batch_size(elements, batch_size=12, collate_fn_kwargs={"pad_token_id": -100})]
    inspect_batches(batches, elements)


if __name__ == "__main__":
    test_batching()
