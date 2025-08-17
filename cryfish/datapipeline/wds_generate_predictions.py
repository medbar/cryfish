import torch
import webdataset as wds
import logging
from tqdm import tqdm
from pathlib import Path
from webdataset.filters import pipelinefilter

# from traceback_with_variables import activate_by_import


def to(b, device):
    if isinstance(b, torch.Tensor):
        return b.to(device)
    if isinstance(b, dict):
        return {k: to(v, device) for k, v in b.items()}
    if isinstance(b, list):
        return [to(v, device) for v in b]
    if isinstance(b, tuple):
        return tuple(to(v, device) for v in b)
    return b


@torch.inference_mode()
def _generate_predictions(
    data, model, device="cpu", move_results_to_cpu=True, progress_bar=False
):
    model = model.to(device).eval()
    if progress_bar:
        data = tqdm(data)
    for i, batch in enumerate(data):
        batch_device = to(batch, device)
        predicted_batch = model.predict_step(batch_device)
        if move_results_to_cpu:
            predicted_batch = to(predicted_batch, "cpu")
        yield {**batch, **predicted_batch}


generate_predictions = pipelinefilter(_generate_predictions)
