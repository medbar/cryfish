import os
import sys
import random
from pathlib import Path

import torch
import webdataset as wds

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, ModelSummary, OnExceptionCheckpoint

# Ensure repo root is importable
THIS_DIR = Path(__file__).resolve().parent
PROMPTS_PATH = THIS_DIR / "prompts" / "train_prompt2.json"


# Local modules mapped from the training YAML
from cryfish.pl_module.init_torch import init as init_torch
from cryfish.pl_module.cryfish_module import CryFishDecoder
from cryfish.models.fixed_stride_connector import FixedSRConnector
from cryfish.models.connector_with_frontend import ConnectorWithFrontend
from cryfish.datapipeline.wds_io import WavAnnoJsonLoader
from cryfish.datapipeline.wds_tokenizing import insert_prompt_autotemplate, tokenize_samples
from cryfish.datapipeline.wds_audio_feats import wav_as_feats
from cryfish.datapipeline.wds_batching import bucketing_batching, batching_constant_batch_size
from cryfish.datapipeline.wds_dataloaders import build_dataloder




def main(
        exp_dir, 
        wavlm_path, 
        llm_path, 
        ckpt_load_path, 
        train_paths: list, 
        val_paths: list,
        total_steps=None,
        seed=42,
        ):

    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # seep/precision
    init_torch(seed=42, precision="medium")

    # Connector Frontend (WavLM)
    frontend_model = AutoModel.from_pretrained(
        wavlm_path
    )

    # Connector Backend 
    connector_backend = FixedSRConnector(
        input_dim=1024,
        emb_dim=1024,
        ff_dim=2048,
        out_dim=3584,
        dropout=0.1,
        num_layers_before=1,
        num_layers_after=2,
        nhead=16,
        num_constant_queries=5,
        sample_rate_reduction=20,
    )

    # Connector Wrapper 
    connector_model = ConnectorWithFrontend(
        frontend_model=frontend_model,
        connector_model=connector_backend,
        frontend_chunk_size=80000,  # 5 sec
        freeze_frontend=True,
    )

    # LLM + Tokenizer
    # set dtype="auto" to load in bf16! should not affect lora training
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path) 
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        # Ensure pad token is defined for batching and loss masking
        tokenizer.pad_token = tokenizer.eos_token

    # PEFT LoRA Config
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.2,
        target_modules=[
            "down_proj",
            "gate_proj",
            "up_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "lm_head",
        ],
    )

    # CryFish LightningModule
    model = CryFishDecoder(
        llm_model=llm_model,
        connector_model=connector_model,
        llm_tokenizer=tokenizer,
        peft_config_or_path=peft_cfg,
        debug_text_dir=str(Path(exp_dir) / "examples"),
        save_text_every_n_step=1,
        DEBUG=False,
    )

    # Optional (inplace) checkpoint load
    ckpt_path = ckpt_load_path
    if os.path.exists(ckpt_path):
        try:
            model.inplace_load_from_checkpoint(ckpt_path)
        except Exception as ex:
            print(f"Warning: failed to inplace load from {ckpt_path}: {ex}")

    # ############ Optimizer and wiring into Lightning ############
    trainable_params = model.get_trainable_parameters(connector=True, llm=True)
    optimizer = torch.optim.Adam(trainable_params, lr=1e-5, weight_decay=1e-6)  # recommend to swap for AdamW
    model.set_optimizers_config(optimizer)

    # ############ Data Pipelines (WebDataset-like) ############
    # Train
    train_ann_paths = train_paths
    train_loader = WavAnnoJsonLoader(ann_path=train_ann_paths, seed="time")
    train_pipeline = [
        train_loader,
        wds.shuffle(bufsize=60, initial=20, rng=random.Random(22)),
        insert_prompt_autotemplate(
            task2promt_json=str(PROMPTS_PATH),
            tokenizer=tokenizer,
            choice_strategy="random",
            use_system_prompt=True,
            seed=142,
        ),
        tokenize_samples(tokenizer=tokenizer, insert_bos=False, insert_eos=True),
        wav_as_feats(),
        bucketing_batching(
            num_buckets=25,
            batch_min_tokens=2600,
            batch_max_tokens=3000,
            audio_feats_reduction=0.000157,
            drop_last=True,
            collate_fn_kwargs={"pad_token_id": tokenizer.pad_token_id},
        ),
        wds.shuffle(bufsize=3, initial=3, rng=random.Random(43)),
    ]
    train_dl = build_dataloder(
        train_pipeline,
        num_workers=13,
        pin_memory=True,
        batch_size=None,
        sampler=None,
    )

    # Val
    val_ann_paths = val_paths
    val_loader = WavAnnoJsonLoader(ann_path=val_ann_paths)
    val_pipeline = [
        val_loader,
        insert_prompt_autotemplate(
            task2promt_json=str(PROMPTS_PATH),
            tokenizer=tokenizer,
            choice_strategy="first",
            use_system_prompt=True,
        ),
        tokenize_samples(tokenizer=tokenizer, insert_bos=False, insert_eos=True),
        wav_as_feats(),
        batching_constant_batch_size(
            batch_size=1,
            collate_fn_kwargs={"pad_token_id": tokenizer.pad_token_id},
        ),
    ]
    val_dl = build_dataloder(
        val_pipeline,
        num_workers=8,
        pin_memory=True,
        batch_size=None,
        sampler=None,
    )

    # ############ Lightning Callbacks and Trainer ############
    checkpoint_cb = ModelCheckpoint(
        dirpath=exp_dir,
        every_n_train_steps=500,
        save_last=True,
        verbose=True,
    )
    on_exception_ckpt = OnExceptionCheckpoint(dirpath=exp_dir)
    progress_cb = TQDMProgressBar(refresh_rate=10)
    summary_cb = ModelSummary()

    # grad_clip_val=1 works
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=exp_dir,
        max_epochs=2,
        precision="bf16-mixed",
        deterministic=False,
        use_distributed_sampler=False,
        accumulate_grad_batches=16,
        callbacks=[progress_cb, summary_cb, checkpoint_cb, on_exception_ckpt],
    )

    # Optional zero-epoch validation
    try:
        trainer.validate(model=model, dataloaders=val_dl)
    except Exception as ex:
        print(f"Warning: validation before training failed: {ex}")

    # Train
    # Resume from last if present
    resume_ckpt = None
    last_ckpt = Path(exp_dir) / "last.ckpt"
    if last_ckpt.exists():
        resume_ckpt = "last"
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main(
        exp_dir=sys.argv[1],
        wavlm_path=sys.argv[2],
        llm_path=sys.argv[3],
        ckpt_load_path=sys.argv[4],
        train_paths=sys.argv[5],
        val_paths=sys.argv[6],
    )


