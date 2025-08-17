import torch
import time
import numpy as np
import soundfile as sf
import json
import webdataset as wds
import logging
import random
import hashlib
import librosa
import os
from glob import glob
from tqdm import tqdm
from pathlib import Path
from itertools import islice
from typing import List, Optional

from torch.utils.data import DataLoader, IterableDataset
from transformers import WhisperFeatureExtractor
from webdataset.filters import pipelinefilter
from webdataset.utils import pytorch_worker_info


def make_tar_list(tar_patterns: List[str], seed: Optional[int] = None) -> List[str]:
    """for example tar_patterns is ['exp/train_1/egs/dump*.tar.gz', 'exp/train_2/egs/dump*.tar.gz']"""
    tars = []
    for t in tar_patterns:
        tars.extend(glob(t))
    logging.info(f"Found {len(tars)} tars in {tar_patterns}")
    if seed is not None:
        random.Random(seed).shuffle(tars)
    return tars


def get_uniq_key(wav_paths, task, Q=None):
    return hashlib.md5(f"{wav_paths} {task} {Q}".encode("utf-8")).hexdigest()


class SimpleAnnoJsonLoader(IterableDataset):
    def __init__(
        self,
        ann_path,
        whisper_path,
        seed=None,
        whisper_padding="max_length",
        max_wav_len=None,
    ):
        super().__init__()
        with open(ann_path, "r") as f:
            self.annotation = json.load(f)["annotation"]
        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)
        # False == disable paddings
        # max_length == whisper default behavior
        self.whisper_padding = whisper_padding
        self.seed = seed
        self.max_wav_len = max_wav_len

    def __len__(self):
        return len(self.annotation)

    def __iter__(self):
        anns = self.annotation
        rank, world_size, worker, num_workers = pytorch_worker_info()
        assert world_size == 1, "Do not use this class for DDP"
        if num_workers > 1:
            full_len = len(anns)
            anns = list(islice(anns, worker, None, num_workers))
            logging.info(
                f"Subset for {worker} worker contains {len(anns)}/{full_len} annotations"
            )
            logging.debug(f"First anno is {anns[0]}")
        if len(anns) == 0:
            logging.warning(
                f"Zero len annotations list! {worker=}, {num_workers=}, {len(anns)=}, {len(self.annotation)}"
            )
            return

        if self.seed is not None:
            random.Random(self.seed).shuffle(anns)
        next_flag = False
        for ann in anns:
            paths = ann.get("paths", None)
            if paths is None:
                paths = [ann.get("path", None)]
            if len(paths) == 0 or paths[0] is None:
                paths = []

            if "__key__" not in ann:
                ann["__key__"] = get_uniq_key(
                    paths, ann.get("task", None), ann.get("Q", None)
                )
            assert (
                "." not in ann["__key__"]
            ), f"the key must be without dots. {ann['__key__']}"
            spec_list = []
            spec_attention_mask_list = []
            raw_wav_list = []
            for path in paths:
                if not os.path.exists(path):
                    logging.error(f"File {path} does not exist")
                    next_flag = True
                    break
                audio, sr = sf.read(path)
                assert sr == 16000, f"Bad {sr=} for audio {path=}, {ann=}"
                if len(audio.shape) == 2:  # stereo to mono
                    logging.warning("Found stereo audio. Converting it into mono")
                    audio = audio[:, 0]
                if audio.shape[0] < sr / 100:
                    next_flag = True
                    break
                # assert (
                #     audio.shape[0] > sr / 100
                # ), f"{ann=} has too short audio {audio.shape=}"
                if len(audio) < sr:  # pad audio to at least 1s
                    sil = np.zeros(sr - len(audio), dtype=float)
                    audio = np.concatenate((audio, sil), axis=0)
                if self.max_wav_len is not None and audio.shape[0] > self.max_wav_len:
                    logging.warning(
                        f"Found audio length greater than {self.max_wav_len} samples {audio.shape=}, {ann=}"
                    )
                    audio = audio[: self.max_wav_len]  # truncate audio to at most 30s
                feats = self.wav_processor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="pt",
                    padding=self.whisper_padding,
                    return_attention_mask=False,
                    truncation=False,
                )
                spectrogram = feats["input_features"].squeeze(0).T  # T x feats
                attention_mask = torch.ones((audio.shape[0] // 160,), dtype=bool)
                spec_list.append(spectrogram)
                spec_attention_mask_list.append(attention_mask)
                raw_wav_list.append(torch.from_numpy(audio))
            if next_flag:
                next_flag = False
                continue
            text = ann["text"]
            task = ann.get("task", None)
            assert task is not None, f"Annotation {ann} doesn't have task name!"
            Q = ann.get("Q", "")
            yield {
                "__key__": ann["__key__"],
                "spec_list.pth": spec_list,
                "spec_attention_mask_list.pth": spec_attention_mask_list,
                "raw_wav_list.pth": raw_wav_list,
                "text.txt": text,
                "task.txt": task,
                "q.txt": Q,
                "wav_paths.txt": "\n".join(paths),
            }


def _write_as_sharded_wds(
    dataset, out_printf_frmt, max_elements_per_shard=50, keys_subset=None
):
    """
    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    Path(out_printf_frmt).parent.mkdir(parents=True, exist_ok=True)
    if keys_subset is not None:
        keys_subset = set(keys_subset)
        keys_subset.add("__key__")

    with wds.ShardWriter(out_printf_frmt, maxcount=max_elements_per_shard) as sink:
        for e in dataset:
            assert "__key__" in e, f"Bad dataset format {e.keys()}"
            if keys_subset is not None:
                assert all(k in e for k in keys_subset), f"Bad dataset element {e.keys()}. Expected keys: {keys_subset}"
                sink.write({k: v for k, v in e.items() if k in keys_subset})
            else:
                sink.write(e)
            yield e


write_as_sharded_wds = pipelinefilter(_write_as_sharded_wds)


def _tee_as_json_anno(data, anno_fname):
    anno = []
    for s in data:
        a = {
            **{k: v for k, v in s.items() if k.startswith("__") and k.endswith("__")},
            "paths": s["wav_paths.txt"].split("\n"),
            "task": s.get("task.txt", None),
            "text": s.get("text.txt", None),
            "prompt": s.get("prompt.txt", None),
        }
        if "q.txt" in s:
            a["Q"] = s["q.txt"]
        if "predicted.txt" in s:
            a["predicted"] = s["predicted.txt"]
        if "db_name.txt" in s:
            a["__db_name__"] = s["db_name.txt"]
        anno.append(a)
        yield s
    logging.info(f"Write {len(anno)} annotations to {anno_fname}")
    with open(anno_fname, "w") as f:
        json.dump({"annotation": anno}, f, ensure_ascii=False, indent=2)


tee_as_json_anno = pipelinefilter(_tee_as_json_anno)


def _tee_as_kaldi_dir(data, out_dir):
    raise NotImplementedError("Not implemented")


tee_as_kaldi_dir = pipelinefilter(_tee_as_kaldi_dir)


def _tee_predicted(data, fname):
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "w") as f:
        for s in data:
            text = f"{s['__key__']}\nref_txt={s.get('text.txt', 'None')}\nhyp_txt={s['predicted.txt']}"
            logging.info(f"{s['__key__']}\n{s['predicted.txt']}")
            f.write(text)
            f.write("\n===\n")
            yield s


tee_predicted = pipelinefilter(_tee_predicted)


class WavAnnoJsonLoader(IterableDataset):
    def __init__(
        self,
        ann_path,
        seed=None,
    ):
        super().__init__()
        if isinstance(ann_path, str) or isinstance(ann_path, Path):
            ann_path = [ann_path]
        self.annotation = []
        for p in ann_path:
            with open(p, "r") as f:
                ann = json.load(f)["annotation"]
            logging.info(f"Loaded {len(ann)} annotations from {p}.")
            self.annotation.extend(ann)
        if seed == "time":
            seed = time.time()
            logging.info(f"Seed is {seed}")
        self.seed = seed

    def __len__(self):
        return len(self.annotation)

    def __iter__(self):
        anns = self.annotation
        rank, world_size, worker, num_workers = pytorch_worker_info()
        # assert world_size == 1 or self.seed is not None, "Do not use this class for DDP or seed must be not None"
        if num_workers > 1:
            full_len = len(anns)
            anns = list(islice(anns, worker, None, num_workers))
            logging.info(
                f"Subset for {worker} worker contains {len(anns)}/{full_len} annotations"
            )
            logging.debug(f"First anno is {anns[0]}")
        if len(anns) == 0:
            logging.warning(
                f"Zero len annotations list! {worker=}, {num_workers=}, {len(anns)=}, {len(self.annotation)}"
            )
            return

        if self.seed is not None:
            seed = self.seed + (rank * 1000000)
            if world_size > 1:
                logging.info(f"Seed for {rank=} is {seed}")
            random.Random(seed).shuffle(anns)
        for ann in anns:
            if "task" not in ann:
                if "text" not in ann or not isinstance(ann["text"], str):
                    logging.warning(f"Skip bad anno {ann=}")
                    continue
                if "Q" not in ann or not isinstance(ann["Q"], str):
                    logging.warning(f"Skip bad anno {ann=}")
                    continue
                if "<SpeechHere>" in ann["text"]:
                    ann["task"] = "custom"
                else:
                    ann["task"] = "QA"

            paths = ann.get("paths", None)
            if paths is None:
                paths = [ann.get("path", None)]
            if len(paths) == 0 or paths[0] is None:
                paths = []

            if "__key__" not in ann:
                ann["__key__"] = get_uniq_key(
                    paths, ann.get("task", None), ann.get("Q", None)
                )
            if not isinstance(ann["__key__"], str):
                ann["__key__"] = str(ann["__key__"])
            assert (
                "." not in ann["__key__"]
            ), f"the key must be without dots. {ann['__key__']}"
            raw_wav_list = []
            if not all(os.path.exists(p) for p in paths):
                logging.error(f"{ann['__key__']} wavs not found {paths}")
                continue
            for path in paths:
                audio, sr = sf.read(path)
                if sr != 16000:
                    logging.warning(
                        f"Resample {ann['__key__']} from {sr} to {16000} Hz"
                    )
                    audio = librosa.resample(
                        audio, orig_sr=sr, target_sr=16000, res_type="fft"
                    )
                    sr = 16000
                assert sr == 16000, f"Bad {sr=} for audio {path=}, {ann=}"
                if len(audio.shape) == 2:  # stereo to mono
                    logging.warning("Found stereo audio. Converting it into mono")
                    audio = audio[:, 0]
                if audio.shape[0] < sr / 100:
                    logging.warning(f"{ann=} has too short audio {audio.shape=}")
                    break
                if len(audio) < sr:  # pad audio to at least 1s
                    sil = np.zeros(sr - len(audio), dtype=float)
                    audio = np.concatenate((audio, sil), axis=0)

                raw_wav_list.append(torch.from_numpy(audio))
            if len(raw_wav_list) != len(paths):
                logging.warning(
                    f"Found {len(raw_wav_list)}/{len(paths)} wavs for {ann=}"
                )
                continue
            text = ann["text"]
            task = ann.get("task", None)
            assert task is not None, f"Annotation {ann} doesn't have task name!"
            Q = ann.get("Q", "")
            db_name = ann.get("__db_name__", "")
            yield {
                "__key__": ann["__key__"],
                "raw_wav_list.pth": raw_wav_list,
                "text.txt": str(text),
                "task.txt": str(task),
                "q.txt": str(Q),
                "wav_paths.txt": "\n".join(paths),
                "db_name.txt": db_name,
            }
