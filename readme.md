
<div align="center">
    <h1>
    <table style="border: none;">
  <tr>
    <td style="border: none;">CryFish</td>
    <td style="border: none;"><img src="media/logo.png" alt="logo" height="75"></td>
  </tr>
</table>
    </h1>
    <p>
    Official PyTorch code and supplementary materials for the CryFish Audio LLM.
    </p>
    <a href="https://arxiv.org/abs/2508.12666"><img src="https://img.shields.io/badge/arXiv-2508.12666-b31b1b" alt="version"></a>
    <a href="https://www.isca-archive.org/interspeech_2025/mitrofanov25_interspeech.pdf"><img src="https://img.shields.io/badge/ISCA-2025-blue" alt="version"></a>
    <a href="https://huggingface.co/theio/CryFish"><img src="https://img.shields.io/badge/Cryfish-🤗-ffcc66" alt="version"></a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="mit"></a>
</div>


<div align="center">
<img src="media/rose_metrics.png" alt="metrics" style="height: 400px;">
</div>

## Running training and inference

### Inference
We provide a basic example for inference with a predefined path for audio in `cryfish/inference.py`.


You need to use paths to the pre-downloaded `wavlm-large`, `Qwen2.5-7B-Instruct`, and `cryfish.safetensors` with [model weights](https://huggingface.co/theio/CryFish).


```bash
python -m cryfish.inference "path/to/wavlm-large" "path/to/Qwen2.5-7B-Instruct" --ckpt "/path/to/cryfish.safetensors"

```
The safetensors file contains the LoRA for the LLM, connector weights, and fine-tuned weights for WavLM. We didn't create a separate LoRA adapter because LoRA includes an `lm_head` module, which breaks export, and just the adapter would end up much larger than the unified checkpoint altogether.

### Training
A basic example of training can be found in `cryfish/train.py`.
```bash
python -m cryfish.train \
  "/path/to/experiments/run1" \
  "/path/to/wavlm-large" \
  "/path/to/Qwen2.5-7B-Instruct" \
  "/path/to/prev_checkpoint.safetensors" \
  "/path/to/train_annotations.json" \
  "/path/to/val_annotations.json"
```

`data_preparation/data_train_example.json` + `data_preparation/readme.md` — there's a small example of what the data should look like and which keys are expected.


There is also a small script for merging checkpoints. Our code does not save weights that are not being trained, so if you fine-tune the model without unfreezing WavLM while loading them from starting checkpoint, you will need to merge the WavLM weights into the checkpoint or safetensors afterward; otherwise, the final checkpoint will not contain the fine-tuned WavLM.
```bash
python -m cryfish.pl_module.utils_ckpt ckpt_1.ckpt ckpt_2.ckpt out.safetensors
```

## FAQ

### Why CryFish?

One of the earliest checkpoints, when asked about audio events, answered "I can hear fishes crying," so that stuck as the model's working name.
