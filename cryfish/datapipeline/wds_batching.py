import torch
import webdataset as wds
import logging
import random
import os
import time
import json

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from webdataset.filters import pipelinefilter
from pathlib import Path
from typing import Union, List


def collate_with_pad_v1(samples, pad_token_id=-100):
    batch = _collate_metainfo(samples)
    keys = set(k for s in samples for k in s.keys())
    if "raw_wav_list.pth" in keys:
        data = _collate_wav_only(samples)
        if "audio_id2sample_id.pth" in batch:
            assert (
                data["audio_id2sample_id.pth"] == batch["audio_id2sample_id.pth"]
            ), f"audio_id2sample_id.pth is not the same {data['audio_id2sample_id.pth']} != {batch['audio_id2sample_id.pth']}"
        batch.update(data)
    if "spec_list.pth" in keys:
        data = _collate_spec_only(samples)
        if "audio_id2sample_id.pth" in batch:
            assert (
                data["audio_id2sample_id.pth"] == batch["audio_id2sample_id.pth"]
            ), f"audio_id2sample_id.pth is not the same {data['audio_id2sample_id.pth']} != {batch['audio_id2sample_id.pth']}"
        batch.update(data)
    if "audio_feats.pth" in keys:
        data = _collate_audiofeats_only(samples)
        if "audio_id2sample_id.pth" in batch:
            assert (
                data["audio_id2sample_id.pth"] == batch["audio_id2sample_id.pth"]
            ), f"audio_id2sample_id.pth is not the same {data['audio_id2sample_id.pth']} != {batch['audio_id2sample_id.pth']}"
        batch.update(data)
    if "prompt_list_tokens_ids.pth" in keys:
        batch.update(_collate_prompt_list_tokens_ids_only(samples))
    if "text_tokens_ids.pth" in keys:
        batch.update(_collate_text_tokens_ids_only(samples, pad_token_id=pad_token_id))
    return batch


def get_unpaded_values(v, sample_id, attention_mask, audio_id2sample_id=None):
    if audio_id2sample_id is None:
        if attention_mask is None:
            return v[sample_id]
        return v[sample_id][attention_mask[sample_id].bool()]
    else:
        if attention_mask is None:
            return [
                v[aid] for aid, sid in enumerate(audio_id2sample_id) if sid == sample_id
            ]
        return [
            v[aid][attention_mask[aid].bool()]
            for aid, sid in enumerate(audio_id2sample_id)
            if sid == sample_id
        ]


def _unbatching_padded(data):
    for batch in data:
        batch = {
            k: v.split("\n===\n") if isinstance(v, str) and k != "__key__" else v
            for k, v in batch.items()
        }
        for sample_id in range(len(batch["__key__"])):
            # for each pth tensor attention mask must be exists
            element = {}
            for k, v in batch.items():
                # v - is a [B, ...] tensor
                if k == "spec.pth" or k == "spec_attention_mask.pth":
                    element["spec_list.pth"] = get_unpaded_values(
                        v,
                        sample_id,
                        batch.get("spec_attention_mask.pth", None),
                        batch["audio_id2sample_id.pth"],
                    )
                elif k == "raw_wav.pth" or k == "raw_wav_attention_mask.pth":
                    element["raw_wav_list.pth"] = get_unpaded_values(
                        v,
                        sample_id,
                        batch.get("raw_wav_attention_mask.pth", None),
                        batch["audio_id2sample_id.pth"],
                    )
                elif k == "audio_feats.pth" or k == "audio_feats_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v,
                        sample_id,
                        batch.get("audio_feats_attention_mask.pth", None),
                        batch["audio_id2sample_id.pth"],
                    )
                elif k == "prompt_list_tokens_ids.pth":
                    element[k] = v[sample_id]
                elif k == "text_tokens_ids.pth" or k == "text_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, sample_id, batch.get("text_attention_mask.pth", None)
                    )
                elif k == "predicted.pth" or k == "predicted_attention_mask.pth":
                    element[k] = get_unpaded_values(
                        v, sample_id, batch.get("predicted_attention_mask.pth", None)
                    )
                elif k == "audio_id2sample_id.pth":
                    pass
                else:
                    assert not isinstance(v, torch.Tensor), f"Cannot unpad {k}, {v}"
                    element[k] = v[sample_id]
            yield element


unbatching_padded = pipelinefilter(_unbatching_padded)


def _batching_constant_batch_size(
    data,
    batch_size=10,
    partial=True,
    collate_fn=collate_with_pad_v1,
    collate_fn_kwargs={},
):
    """Create batches of the given size.
    :param data: iterator
    :param partial: return partial batches
    :returns: iterator
    """
    assert batch_size > 0, f"Wrong batch size {batch_size}"
    batch = []
    for sample in data:
        batch.append(sample)
        if len(batch) == batch_size:
            batch = collate_fn(batch, **collate_fn_kwargs)
            yield batch
            batch = []
    if len(batch) == 0:
        return
    elif partial:
        batch = collate_fn(batch, **collate_fn_kwargs)
        yield batch


batching_constant_batch_size = pipelinefilter(_batching_constant_batch_size)


def _bucketing_batching(
    data,
    num_buckets=20,
    batch_min_tokens=3000,
    batch_max_tokens=10000,
    audio_feats_reduction=1 / 4,
    drop_last=False,
    collate_fn=collate_with_pad_v1,
    collate_fn_kwargs={"pad_token_id": -100},
    batch_must_be_with_audio=False,
):
    """
    Bucketing batching with dynamic batch size
    batch size is determined by the number of tokens in the batch
    batch_min_tokens is the minimum number of tokens in the batch
    batch_max_tokens is the maximum number of tokens in the batch
    audio_feats_reduction is the reduction factor for audio sample rate
    drop_last is whether to drop the last batch if it is not full
    collate_fn is the function to collate the batch
    collate_fn_kwargs is the kwargs for the collate_fn
    """
    buckets = [[None, []] for _ in range(num_buckets)]
    num_e = 0
    num_batches = 0
    avg_num_buckets = 0
    avg_mindist = 0
    for e in data:
        num_e += 1
        e_lens = _get_e_lens(e, audio_feats_reduction)
        if sum(e_lens) > batch_max_tokens:
            logging.warning(
                f"Sample {e['__key__']} has {sum(e_lens)} tokens, which is greater than batch_max_tokens {batch_max_tokens}. Skip it."
            )
            continue
        mindist = float("inf")
        mindist_i = 0
        for i, (b_lens, b) in enumerate(buckets):
            if len(b):
                avg_num_buckets += 1
            dist = _num_added_tokens(b_lens, len(b), e_lens)
            if dist < mindist:
                mindist = dist
                mindist_i = i
        avg_mindist += mindist
        i = mindist_i
        buckets[i][1].append(e)
        if buckets[i][0] is None:
            buckets[i][0] = e_lens
        else:
            buckets[i][0] = [max(l1, l2) for l1, l2 in zip(e_lens, buckets[i][0])]
        batch_num_tokens = sum(buckets[i][0]) * len(buckets[i][1])
        if batch_num_tokens >= batch_min_tokens:
            if batch_must_be_with_audio and buckets[i][0][1] == 0:
                logging.warning(f"{batch_must_be_with_audio=}, {buckets[i]=}. Skip it")
                buckets[i] = [None, []]
                continue
            if batch_num_tokens < batch_max_tokens:
                batch = collate_fn(buckets[i][1], **collate_fn_kwargs)
                yield batch
                buckets[i] = [None, []]
                num_batches += 1
            else:
                logging.debug(
                    f"Bucket {i} has {batch_num_tokens} tokens (bucket len is {len(buckets[i][1])}), which is greater than batch_max_tokens {batch_max_tokens}. Split it."
                )
                batch = collate_fn(buckets[i][1][:-1], **collate_fn_kwargs)
                yield batch
                buckets[i] = [e_lens, [buckets[i][1][-1]]]
                num_batches += 1
    buckets = [b for b in buckets if b[0] is not None]
    if not drop_last:
        logging.debug("Iteration completed. Flush buckets.")
        for i, b in enumerate(buckets):
            if b[0] is None:
                continue
            if batch_must_be_with_audio and b[0][1] == 0:
                logging.warning(f"{batch_must_be_with_audio=}, {buckets[i]=}. Skip it")
                continue
            if sum(b[0]) * len(b[1]) > batch_max_tokens:
                logging.warning(f"{batch_max_tokens=}, {buckets[i]=}. Skip it")
                continue
            batch = collate_fn(b[1], **collate_fn_kwargs)
            yield batch
            num_batches += 1
    else:
        logging.debug(f"Dropped {len(buckets)} buckets")
    logging.debug(
        f"Bucketing stats: number of elements processed: {num_e}, bathes collated {num_batches}, "
        f"Average buckets using {avg_num_buckets/num_e}, Average min dist {avg_mindist/num_e}"
    )


bucketing_batching = pipelinefilter(_bucketing_batching)


def _num_added_tokens(batch, bz, add_len):
    if batch is None:
        # 0.1 is a magic number for better bucketing
        # 1000 if element has no audio
        magic = 0.1 if add_len[1] else 1000
        return sum(le**2 * magic for le in add_len)
    return sum(
        (b - l) ** 2 if b >= l else (b - l) ** 2 * bz for b, l in zip(batch, add_len)
    )


def _get_e_lens(e, audio_feats_reduction):
    if "audio_feats.pth" in e:
        assert (
            len(e["audio_feats.pth"].shape) == 3
        ), f"audio_feats.pth must be 3D tensor {e['audio_feats.pth'].shape=}"
        return [
            sum(len(p) for p in e["prompt_list_tokens_ids.pth"]),
            e["audio_feats.pth"].shape[0]
            * e["audio_feats.pth"].shape[1]
            * audio_feats_reduction,
            len(e["text_tokens_ids.pth"]),
        ]
    else:
        return [
            sum(len(p) for p in e["prompt_list_tokens_ids.pth"]),
            0,
            len(e["text_tokens_ids.pth"]),
        ]


def _collate_spec_only(samples):
    """
    input: spec.pth, [spec_attention_mask.pth]
    output: spec.pth, spec_attention_mask.pth
    """
    audio_id2sample_id = []
    specs = []
    attention_mask = []
    for i, s in enumerate(samples):
        if "spec_list.pth" in s:
            for j, a in enumerate(s["spec_list.pth"]):
                if "spec_attention_mask_list.pth" in s:
                    att = s["spec_attention_mask_list.pth"][j]
                else:
                    att = a.new_ones((a.shape[0],), dtype=bool)
                specs.append(a)
                audio_id2sample_id.append(i)
                attention_mask.append(att)
    if len(specs) > 0:
        spec = pad_sequence(specs, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(
            attention_mask,
            padding_value=False,
            batch_first=True,
        )

        return {
            "spec.pth": spec,
            "spec_attention_mask.pth": attention_mask.bool(),
            "audio_id2sample_id.pth": audio_id2sample_id,
        }
    else:
        return {}


def _collate_wav_only(samples):
    """
    input: raw_wav_list.pth, [raw_wav_attention_mask_list.pth]
    output: raw_wav.pth, raw_wav_attention_mask.pth
    """

    audio_id2sample_id = []
    wavs = []
    attention_mask = []
    for i, s in enumerate(samples):
        if "raw_wav_list.pth" in s:
            for j, a in enumerate(s["raw_wav_list.pth"]):
                if "raw_wav_attention_mask_list.pth" in s:
                    att = s["raw_wav_attention_mask_list.pth"][j]
                else:
                    att = a.new_ones((a.shape[0],), dtype=bool)
                wavs.append(a)
                audio_id2sample_id.append(i)
                attention_mask.append(att)
    if len(wavs) > 0:
        wavs = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(
            attention_mask,
            padding_value=False,
            batch_first=True,
        )

        return {
            "raw_wav.pth": wavs,
            "raw_wav_attention_mask.pth": attention_mask.bool(),
            "audio_id2sample_id.pth": audio_id2sample_id,
        }
    else:
        return {}


def _collate_audiofeats_only(samples):
    """
    input: audio_feats.pth, audio_feats_attention_mask.pth
    output: audio_feats.pth, audio_feats_attention_mask.pth
    """
    audio_feats = [s["audio_feats.pth"] for s in samples if "audio_feats.pth" in s]
    assert all(len(a.shape) == 3 for a in audio_feats), f"bad audio_feats {audio_feats}"

    audio_id2sample_id = []
    audio_feats = []
    attention_mask = []
    for i, s in enumerate(samples):
        if "audio_feats.pth" in s:
            for j, a in enumerate(s["audio_feats.pth"]):
                if "audio_feats_attention_mask.pth" in s:
                    att = s["audio_feats_attention_mask.pth"][j]
                else:
                    att = a.new_ones((a.shape[0],), dtype=bool)
                audio_feats.append(a)
                audio_id2sample_id.append(i)
                attention_mask.append(att)
    if len(audio_feats) > 0:
        feats = pad_sequence(audio_feats, batch_first=True, padding_value=0.0)
        attention_mask = pad_sequence(
            attention_mask,
            padding_value=False,
            batch_first=True,
        )

        return {
            "audio_feats.pth": feats,
            "audio_feats_attention_mask.pth": attention_mask.bool(),
            "audio_id2sample_id.pth": audio_id2sample_id,
        }
    else:
        return {}


def _collate_prompt_list_tokens_ids_only(samples):
    """
    prompt_list_tokens_ids.pth is List[List[LongTensor]]
    this tensor is not padded because the new zip_embeddings function
    """
    return {
        "prompt_list_tokens_ids.pth": [s["prompt_list_tokens_ids.pth"] for s in samples]
    }


def _collate_text_tokens_ids_only(samples, pad_token_id=-100):
    """
    text_tokens_ids.pth is List[LongTensor]
    text_attention_mask.pth is List[BoolTensor]
    """
    text = [s["text_tokens_ids.pth"] for s in samples]
    assert len(text[0].shape) == 1, f"bad text_tokens_ids.pth {text[0].shape=}"
    assert (
        "text_attention_mask.pth" not in samples[0]
    ), f"text_attention_mask.pth already in {samples[0].keys()}"
    padded = pad_sequence(text, batch_first=True, padding_value=pad_token_id)
    length = torch.as_tensor([t.shape[0] for t in text])
    attention_mask = torch.arange(padded.size(1)).unsqueeze(0) < length.unsqueeze(1)
    return {
        "text_tokens_ids.pth": padded,
        "text_attention_mask.pth": attention_mask.bool(),
    }


def _collate_metainfo(samples):
    batch = {"__key__": [s["__key__"] for s in samples]}
    if "wav_paths.txt" in samples[0]:
        batch["wav_paths.txt"] = "\n===\n".join(
            [
                (
                    s["wav_paths.txt"]
                    if isinstance(s["wav_paths.txt"], str)
                    else "\n".join(s["wav_paths.txt"])
                )
                for s in samples
            ]
        )
    if "prompt.txt" in samples[0]:
        batch["prompt.txt"] = "\n===\n".join([s["prompt.txt"] for s in samples])
    if "task.txt" in samples[0]:
        batch["task.txt"] = "\n===\n".join([s["task.txt"] for s in samples])
    if "text.txt" in samples[0]:
        batch["text.txt"] = "\n===\n".join([s["text.txt"] for s in samples])
    if "db_name.txt" in samples[0]:
        batch["db_name.txt"] = "\n===\n".join([s["db_name.txt"] for s in samples])
    # if "q.txt" in samples[0]:
    #    batch["q.txt"] = "\n===\n".join([str(s."q.txt") for s in samples])
    return batch
