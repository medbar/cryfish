import logging
import json
import contextlib
import random

import torch
import torch.nn as nn


class ConnectorWithFrontend(nn.Module):
    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    def __init__(
        self,
        frontend_model,
        connector_model,
        frontend_chunk_size=5 * 16000,
        freeze_frontend=True,
    ):
        super().__init__()
        self.frontend_model = frontend_model
        self.connector_model = connector_model
        self.frontend_chunk_size = frontend_chunk_size
        if freeze_frontend:
            for param in self.frontend_model.parameters():
                param.requires_grad = False

    def forward(self, batch, return_dict=False):
        # audio_feats.pth is B, T, F
        audio_feats = batch["audio_feats.pth"].squeeze(2).to(self.dtype)
        B, T = audio_feats.shape
        # audio_feats_attention_mask.pth is B, T
        audio_feats_attention_mask = batch["audio_feats_attention_mask.pth"]
        last_chunk_size = T % self.frontend_chunk_size
        if last_chunk_size > 0:
            pad = self.frontend_chunk_size - last_chunk_size
            audio_feats = torch.nn.functional.pad(audio_feats, (0, pad), value=0)
            audio_feats_attention_mask = torch.nn.functional.pad(
                audio_feats_attention_mask, (0, pad), value=False
            )
        # split audio_feats into chunks
        chunks = audio_feats.unfold(
            1, self.frontend_chunk_size, self.frontend_chunk_size
        )
        # B x num_chunks x chunk_size
        num_chunks = chunks.shape[1]
        chunks_attention_mask = audio_feats_attention_mask.unfold(
            1, self.frontend_chunk_size, self.frontend_chunk_size
        )
        assert (
            chunks_attention_mask.shape[1] == num_chunks
        ), f"{chunks_attention_mask.shape[1]} != {num_chunks}"

        chunks = chunks.reshape(B * num_chunks, self.frontend_chunk_size)
        chunks_attention_mask = chunks_attention_mask.reshape(
            B * num_chunks, self.frontend_chunk_size
        )

        frontend_output = self.frontend_model(
            chunks, attention_mask=chunks_attention_mask
        ).last_hidden_state
        frontend_output = frontend_output.view(B, -1, frontend_output.shape[-1])
        frontend_output_attention_mask = (
            torch.nn.functional.interpolate(
                audio_feats_attention_mask.unsqueeze(1).half(),
                size=frontend_output.shape[1],
            ).squeeze(1)
            > 0
        )
        connector_output = self.connector_model(
            {
                "audio_feats.pth": frontend_output,
                "audio_feats_attention_mask.pth": frontend_output_attention_mask,
            },
            return_dict=return_dict,
        )
        return connector_output
