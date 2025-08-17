import os
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType

from cryfish.pl_module.init_torch import init as init_torch
from cryfish.pl_module.cryfish_module import CryFishDecoder
from cryfish.models.fixed_stride_connector import FixedSRConnector
from cryfish.models.connector_with_frontend import ConnectorWithFrontend


def _build_connector(frontend_model_name_or_path: str) -> ConnectorWithFrontend:
    """Create ConnectorWithFrontend exactly as in training script."""
    frontend_model = AutoModel.from_pretrained(frontend_model_name_or_path)

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

    connector = ConnectorWithFrontend(
        frontend_model=frontend_model,
        connector_model=connector_backend,
        frontend_chunk_size=80000,  # 5 sec
        freeze_frontend=True,
    )
    return connector


def _build_llm_and_tokenizer(llm_model_name_or_path: str):
    """Load LLM and tokenizer as in training (no explicit dtype override)."""
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm_model, tokenizer


def _build_lora_config() -> LoraConfig:
    """Return the LoRA config matching training hyperparameters/targets."""
    return LoraConfig(
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


def load_cryfish_for_inference(
    wavlm_path: str,
    llm_path: str,
    ckpt_path: str | None = None,
    device: str | None = None,
    seed: int = 42,
):
    """
    Build and optionally load a trained CryFishDecoder (no data pipeline).

    Args:
        wavlm_path: Path to WavLM (or compatible) frontend for audio feature extraction.
        llm_path: Path to HF CausalLM checkpoint used during training.
        ckpt_path: Optional path to a Lightning .ckpt with trained trainable weights.
        device: Optional torch device string; defaults to "cuda" if available else "cpu".
        seed: Set matmul precision/seed same as training init.

    Returns:
        model (CryFishDecoder), tokenizer (AutoTokenizer)
    """
    init_torch(seed=seed, precision="medium")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    connector_model = _build_connector(wavlm_path)
    llm_model, tokenizer = _build_llm_and_tokenizer(llm_path)
    peft_cfg = _build_lora_config()

    model = CryFishDecoder(
        llm_model=llm_model,
        connector_model=connector_model,
        llm_tokenizer=tokenizer,
        peft_config_or_path=peft_cfg,
        debug_text_dir=None,
        save_text_every_n_step=1,
        DEBUG=False,
    )

    if ckpt_path is not None and os.path.exists(ckpt_path):
        model.inplace_load_from_checkpoint(ckpt_path)

    model = model.to(device).eval()
    return model, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load CryFish model for inference")
    parser.add_argument("wavlm_path", type=str, help="Path to WavLM (frontend) model")
    parser.add_argument("llm_path", type=str, help="Path to Qwen 2.5 7b Instruct")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to Lightning checkpoint with trained weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (defaults to cuda if available)",
    )
    args = parser.parse_args()

    model, tokenizer = load_cryfish_for_inference(
        wavlm_path=args.wavlm_path,
        llm_path=args.llm_path,
        ckpt_path=args.ckpt,
        device=args.device,
    )
    print(f"Model loaded to {next(model.parameters()).device}")


