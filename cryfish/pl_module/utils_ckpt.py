import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def load_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint '{path}' does not contain 'state_dict'")
    return ckpt["state_dict"]


def main():
    parser = argparse.ArgumentParser(description="Merge two Lightning checkpoints' state_dicts and save as .safetensors")
    parser.add_argument("ckpt1", type=str, help="Path to first checkpoint (.ckpt/.pt). Its keys take priority on conflicts.")
    parser.add_argument("ckpt2", type=str, help="Path to second checkpoint (.ckpt/.pt). Only new keys are taken when missing in first.")
    parser.add_argument("out", type=str, help="Output .safetensors path")
    args = parser.parse_args()

    sd1 = load_state_dict(args.ckpt1)
    sd2 = load_state_dict(args.ckpt2)

    new_keys = [k for k in sd2.keys() if k not in sd1]

    merged = dict(sd2)
    merged.update(sd1)  # first checkpoint has priority on overlapping keys

    # Ensure CPU tensors for safetensors
    merged_cpu = {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in merged.items()}

    out_path = Path(args.out)
    if out_path.suffix != ".safetensors":
        out_path = out_path.with_suffix(".safetensors")

    save_file(merged_cpu, str(out_path))

    print(f"Keys in the first checkpoint: {len(sd1.keys())}")
    print(f"New keys taken from the second checkpoint: {len(new_keys)}")
    print(f"Saved merged state_dict to: {out_path}")


if __name__ == "__main__":
    main()


