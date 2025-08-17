# CryFish 

Repository for CryFish model, training script and a bit of additional info on data generation pipeline.




## Running training and inference

### Inference
We're providing a basic example for inference with predefined path for audio in `cryfish/inference.py`


You need to pass path for predownloaded `wavlm-large`, `Qwen2.5-7b-instruct` and `cryfish.safetensors` with model weights. \\


```bash
python -m cryfish.inference "path/to/wavlm-large" "path/to/Qwen2.5-7B-Instruct" --ckpt "/path/to/cryfish.safetensors"

```
Safetensors file contains LoRA for LLM, connector weights, and finetuned weights for wavlm. We're not providing separate lora adapter due to it containing `lm_head`, which makes exported adapter be way bigger than checkpoint. 

### Training
Basic example of running training can be found in `cryfish/train.py`\\
```bash
python -m cryfish.train \
  "/path/to/experiments/run1" \
  "/path/to/wavlm-large" \
  "/path/to/Qwen2.5-7B-Instruct" \
  "/path/to/prev_checkpoint.safetensors" \
  "/path/to/train_annotations.json" \
  "/path/to/val_annotations.json"
```

`train_annotations.json` - there's small example of how the data should look like and what keys are expected. 


There is also a small script for merging ckpts, our code does not save weights which are not being trained, so if you want to finetune model without unfreezing wavlm, then you'll have to merge wavlm weights into ckpt or safetensors. 
```bash
python -m cryfish.pl_module.utils_ckpt ckpt_1.ckpt ckpt_2.ckpt out.safetensors
```
