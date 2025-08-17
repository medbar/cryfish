import torch
import webdataset as wds
import logging
import kaldiio
from tqdm import tqdm
from pathlib import Path
from webdataset.filters import pipelinefilter

from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def _wav_as_feats(data):
    """
    audio_feats.pth is a wavs tensor [B, T, 1]
    audio_feats_attention_mask.pth is a attention mask tensor [B, T]
    """
    for sample in data:
        raw_wav_list = sample.pop("raw_wav_list.pth")
        if raw_wav_list is None or sum(len(w) for w in raw_wav_list) == 0:
            yield sample
            continue
        wavs = (
            pad_sequence(raw_wav_list, padding_value=0, batch_first=True)
            .to(torch.bfloat16)
            .unsqueeze(2)
        )
        wav_lengths = torch.as_tensor([len(w) for w in raw_wav_list])
        attention_mask = torch.arange(wavs.size(1)).unsqueeze(
            0
        ) < wav_lengths.unsqueeze(1)

        yield {
            **sample,
            "audio_feats.pth": wavs,
            "audio_feats_attention_mask.pth": attention_mask,
        }


wav_as_feats = pipelinefilter(_wav_as_feats)


def _replace_beats2eat(data, feats):
    """
    audio_feats.pth is a wavs tensor [B, T, 1]
    """

    dict_eat_emb = kaldiio.load_scp(feats)

    for sample in data:
        wav_paths = sample.pop("wav_paths.txt")
        audio_feats = sample.pop("audio_feats.pth")

        n_frames = audio_feats.shape[1]

        eat_emb_cur = dict_eat_emb[wav_paths]
        n_frames_eat_emb = len(eat_emb_cur)

        eat_emb = torch.as_tensor([eat_emb_cur[:n_frames]])
        audio_feats = audio_feats[:, :, :1472]

        diff = n_frames - eat_emb.shape[1]
        if diff > 0:
            if diff > 5:
                logging.warning(
                    f"Padding too much feats embs {n_frames=}, {eat_emb.shape[1]=}, wav_paths={wav_paths}, audio_feats={audio_feats}"
                )
                audio_feats = audio_feats[:, : eat_emb.shape[1]]
                n_frames = audio_feats.shape[1]
            diff = n_frames - eat_emb.shape[1]
            eat_emb = torch.nn.functional.pad(eat_emb, (0, 0, 0, diff), value=0.0)
        assert (
            audio_feats.shape[1] == eat_emb.shape[1]
        ), f"audio_feats.shape[1]={audio_feats.shape[1]} == eat_emb.shape[1]={eat_emb.shape[1]}, diff={diff}"

        embedding = torch.cat([audio_feats, eat_emb], dim=-1)
        embedding = torch.cat(
            [
                F.normalize(embedding[:, :1280], p=2, dim=1),
                F.normalize(embedding[:, 1280:1472], p=2, dim=1),
                F.normalize(embedding[:, 1472:2240], p=2, dim=1),
            ],
            dim=1,
        )

        yield {
            **sample,
            "audio_feats.pth": embedding,
        }


replace_beats2eat = pipelinefilter(_replace_beats2eat)


def _replace_beats2eat_norm_ecapa_multi_ch(data, feats):
    """
    audio_feats.pth is a wavs tensor [B, T, 1]
    audio_feats_attention_mask.pth is a attention mask tensor [B, T]
    """

    dict_eat_emb = kaldiio.load_scp(feats)

    for sample in data:
        wav_paths = sample.pop("wav_paths.txt")
        audio_feats = sample.pop("audio_feats.pth")
        audio_feats_attention_mask = sample.pop("audio_feats_attention_mask.pth")
        n_frames_audio_feats = audio_feats.shape[1]

        wav_paths_list = wav_paths.split("\n")
        count_wav = len(wav_paths_list)

        max_len = 0
        for i in range(0, count_wav):
            eat_emb_cur = dict_eat_emb[wav_paths_list[i]]
            n_frames_eat_emb = len(eat_emb_cur)
            if n_frames_eat_emb > max_len:
                max_len = n_frames_eat_emb

        if n_frames_audio_feats > max_len:
            audio_feats = audio_feats[:, :max_len, :]
        else:
            max_len = n_frames_audio_feats

        for i in range(0, count_wav):
            audio_feats_cur_ch = audio_feats[i]

            eat_emb_cur = dict_eat_emb[wav_paths_list[i]]
            n_frames_eat_emb = len(eat_emb_cur)

            eat_emb = torch.as_tensor(eat_emb_cur[:max_len])
            diff = max_len - eat_emb.shape[0]
            if diff > 0:
                if diff > 5:
                    logging.warning(
                        f"Padding too much feats embs {max_len=}, {eat_emb.shape[0]=}, wav_paths={wav_paths}."
                    )
                eat_emb = torch.nn.functional.pad(eat_emb, (0, 0, 0, diff), value=0.0)

            assert (
                max_len == eat_emb.shape[0]
            ), f"max_len={max_len} == eat_emb.shape[0]={eat_emb.shape[0]}, diff={diff}"

            audio_feats[i] = torch.cat(
                [
                    audio_feats_cur_ch[:, :1280],
                    F.normalize(audio_feats_cur_ch[:, 1280:1472], p=2, dim=1),
                    eat_emb,
                ],
                dim=-1,
            )
        audio_feats_attention_mask = audio_feats_attention_mask[
            :, : audio_feats.shape[1]
        ]

        yield {
            **sample,
            "audio_feats.pth": audio_feats,
            "audio_feats_attention_mask": audio_feats_attention_mask,
        }


replace_beats2eat_norm_ecapa_multi_ch = pipelinefilter(
    _replace_beats2eat_norm_ecapa_multi_ch
)


@torch.inference_mode()
def _feature_extractor_from_wavs(
    data,
    model,
    device="cpu",
    move_results_to_cpu=True,
    feats_dtype=torch.bfloat16,
):
    """
    data is iterator of samples

    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    model = model.to(device).eval()
    for e in tqdm(data):
        logging.debug(f"Start processing {e['__key__']}")
        assert "raw_wav_list.pth" in e, f"No raw_wav_list.pth in {e}"
        batch = {
            "raw_wav.pth": pad_sequence(
                e["raw_wav_list.pth"],
                padding_value=0,
                batch_first=True,
            ).to(device)
        }
        raw_wav_length = torch.as_tensor([len(w) for w in e["raw_wav_list.pth"]])
        raw_wav_attention_mask = torch.arange(batch["raw_wav.pth"].size(1)).unsqueeze(
            0
        ) < raw_wav_length.unsqueeze(1)
        batch["raw_wav_attention_mask.pth"] = raw_wav_attention_mask.to(device)

        if "spec_list.pth" in e:
            batch["spec.pth"] = pad_sequence(
                e["spec_list.pth"],
                padding_value=0,
                batch_first=True,
            ).to(device)

        if "spec_attention_mask_list.pth" in e:
            batch["spec_attention_mask.pth"] = pad_sequence(
                e["spec_attention_mask_list.pth"],
                padding_value=False,
                batch_first=True,
            ).to(device)
        # shape is B, T, C
        audio_feats_dict = model(batch)
        logging.debug(f"Extracted {audio_feats_dict['audio_feats.pth'].shape=}")
        audio_feats_dict["audio_feats.pth"] = audio_feats_dict["audio_feats.pth"].to(
            feats_dtype
        )
        if move_results_to_cpu:
            audio_feats_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in audio_feats_dict.items()
            }
        yield {**e, **audio_feats_dict}


feature_extractor_from_wavs = pipelinefilter(_feature_extractor_from_wavs)


@torch.inference_mode()
def _feature_extractor(
    data,
    model,
    device="cpu",
    move_results_to_cpu=True,
    batch_mode=True,
    feats_dtype=torch.bfloat16,
):
    """
    out_printf_frmt is like "dir/shard-000-%06d.tar"
    """
    model = model.to(device).eval()
    for batch in tqdm(data):
        orig_batch = batch
        assert (
            batch_mode
            and len(batch["raw_wav.pth"].shape) == 2
            or not batch_mode
            and len(batch["raw_wav.pth"].shape) == 1
        ), f"Batch shape {batch['raw_wav.pth'].shape=} is not correct shape for {batch_mode=}"
        logging.debug(f"Start processing {batch['__key__']}")
        if not batch_mode:
            batch = {
                k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        # shape is B, T, C
        audio_feats_dict = model(batch)
        logging.debug(f"Extracted {audio_feats_dict['audio_feats.pth'].shape=}")
        audio_feats_dict["audio_feats.pth"] = audio_feats_dict["audio_feats.pth"].to(
            feats_dtype
        )
        if move_results_to_cpu:
            audio_feats_dict = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in audio_feats_dict.items()
            }
        if not batch_mode:
            audio_feats_dict = {
                k: v.squeeze(0) if isinstance(v, torch.Tensor) else v
                for k, v in audio_feats_dict.items()
            }
        yield {**orig_batch, **audio_feats_dict}


feature_extractor = pipelinefilter(_feature_extractor)
