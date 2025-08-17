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


def _insert_prompt_autotemplate(
    data,
    task2promt_json,
    tokenizer,
    choice_strategy="first",
    rng=None,
    seed=42,
    use_system_prompt=True,
    add_generation_prompt=True
):
    """
    input ["task.txt", Optional["q.txt"]]
    output [..., "prompt.txt"]
    "<Speech><SpeechHere></Speech> Recognize the english speech and give me the transcription only in english."
    prompt template is taken from tokenizer.apply_chat_template
    """
    assert (
        use_system_prompt
    ), "use_system_prompt must be True, we don't want 'You are Qwen, created by Alibaba Cloud' system prompt =)))"
    assert choice_strategy in [
        "first",
        "random",
    ], f"Unknown choice strategy {choice_strategy}"
    if isinstance(task2promt_json, dict):
        task2prompt_format = task2promt_json
    else:
        assert os.path.exists(task2promt_json), f"{task2promt_json=}"
        with open(task2promt_json) as f:
            # task name to list of prompt formast strings
            task2prompt_format = {
                k: v if isinstance(v, list) else [v] for k, v in json.load(f).items()
            }

    if choice_strategy == "random":
        assert rng is None or seed is None, "seed or rng must be None"
        if rng is None:
            if seed is None:
                seed = int((os.getpid() + time.time()) * 1e9)
                logging.debug(f"insert prompt random seed is {seed}")
            rng = random.Random(seed)

    for sample in data:
        task = sample.get("task.txt", None)
        if task not in task2prompt_format and task.lower != 'custom':
            task='QA'
        if task in task2prompt_format:
            if choice_strategy == "first":
                prompt_format = task2prompt_format[task][0]
            elif choice_strategy == "random":
                prompt_format = rng.choice(task2prompt_format[task])
        else:
            prompt_format = "{}"
        if task == 'QA':
            assert 'q.txt' in sample, f"{sample=}"
        q = sample.get("q.txt", "ERROR")
        prompt = prompt_format.format(q)
        if (
            "raw_wav_list.pth" in sample
            and len(sample["raw_wav_list.pth"]) > 0
            or "audio_feats.pth" in sample
            and len(sample["audio_feats.pth"]) > 0
        ):
            assert (
                "<SpeechHere>" in prompt
            ), f"Prompt format {prompt_format} does not contain <SpeechHere>. {task=}, {q=}, {sample=}"
            num_wavs = len(
                sample.get("audio_feats.pth", sample.get("raw_wav_list.pth"))
            )
            num_p = prompt.count("<SpeechHere>")
            if num_p < num_wavs:
                assert task == "QA", "{sample=}"
                for _ in range(num_p, num_wavs):
                    prompt = prompt_format.format(prompt)
            assert num_wavs == prompt.count(
                "<SpeechHere>"
            ), "{num_wavs=}, {num_p=}, {prompt=}, {sample=}"
        chat = [
            {"role": "user", "content": prompt},
        ]
        if use_system_prompt:
            system_prompts = task2prompt_format.get("SYSTEM_PROMPT", None)
            if system_prompts is not None:
                if isinstance(system_prompts, str):
                    sys_prompt = system_prompts
                else:
                    sys_prompt = (
                        rng.choice(system_prompts)
                        if choice_strategy == "random"
                        else system_prompts[0]
                    )
                chat.insert(0, {"role": "system", "content": sys_prompt})
            elif isinstance(system_prompts, str):
                chat.insert(0, {"role": "system", "content": system_prompts})
            else:
                raise ValueError(f"No system prompt 'SYSTEM_PROMPT' found in {task2promt_json}")
        #chat.append({"role": "assistant", "content": ""})
        prompt = tokenizer.apply_chat_template(chat,
                                               tokenize=False,
                                               add_generation_prompt=add_generation_prompt)
        yield {**sample, "prompt.txt": prompt}


insert_prompt_autotemplate = pipelinefilter(_insert_prompt_autotemplate)


def _tokenize_samples(
    data, tokenizer, insert_bos=False, insert_eos=True, bos_sym=None, end_sym=None
):
    """
    converting [bos, prompt, text, eos] into sequence of token indices

    input ["prompt.txt", Optional["text.txt"]
    output [..., "prompt_list_tokens_ids.pth", "text_tokens_ids.pth"]
    """
    if insert_bos:
        logging.warning(
            "insert_bos=True is deprecated. Avoid using it. Use insert_prompt_autotemplate instead."
        )
    if bos_sym is None:
        bos_sym = tokenizer.bos_token
    if end_sym is None:
        end_sym = tokenizer.eos_token
    for sample in data:
        sample = sample.copy()
        assert (
            "prompt.txt" in sample
        ), f"Construct prompt before running tokenizer. {sample.keys()=}"
        prompt = sample["prompt.txt"]
        if insert_bos:
            if not prompt.startswith(bos_sym):
                prompt = f"{bos_sym}{prompt}"
            else:
                logging.warning(f"Prompt {prompt} already starts with {bos_sym}")
        splitted_by_speech = prompt.split("<SpeechHere>")
        sample["prompt_list_tokens_ids.pth"] = [
            tokenizer(
                p,
                return_tensors="pt",
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids.squeeze(0)
            for p in splitted_by_speech
        ]
        # attention mask is always 1
        text = sample.get("text.txt", "")
        if not isinstance(text, str):
            logging.warning(f"{type(text)=}, converting it to str")
            text = str(text)
        if insert_eos:
            if not text.endswith(end_sym):
                text = f"{text}{end_sym}"
            else:
                logging.warning(f"Text {text} already ends with {end_sym}")
                pass
        sample["text_tokens_ids.pth"] = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            return_attention_mask=False,
            padding=False,
        ).input_ids.squeeze(0)
        yield sample


tokenize_samples = pipelinefilter(_tokenize_samples)


def _tokenize_batches(data, tokenizer, bos_sym=None, end_sym=None):
    """
    converting [bos, prompt, text, eos] into sequence of token indices

    input ["prompt.pkl", Optional["text.pkl"] # list of texts
    output [..., "prompt_tokens_ids.pth", "text_tokens_ids.pth"]
    """
    # TODO
    raise NotImplementedError()
