import logging
import json
import contextlib
import random

import torch
import torch.nn as nn

# pip install torchtune
# from torchtune.module import RotaryPositionalEmbeddings


class FixedSRConnector(nn.Module):
    @property
    def device(self):
        return list(self.parameters())[0].device

    @property
    def dtype(self):
        return list(self.parameters())[0].dtype

    def __init__(
        self,
        input_dim=2048,
        emb_dim=1024,
        ff_dim=2048,
        out_dim=4096,
        dropout=0.1,
        num_layers_before=1,
        num_layers_after=2,
        nhead=16,
        num_constant_queries=5,
        sample_rate_reduction=20,  # 20mc whisper embs -> 400mc llama input
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.num_layers_before = num_layers_before
        self.num_layers_after = num_layers_after
        self.num_constant_queries = num_constant_queries
        self.sample_rate_reduction = sample_rate_reduction
        self.emb_proj = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.SiLU())
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=nn.functional.silu,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers_before
        )
        self.sr_reduction = nn.AvgPool1d(
            sample_rate_reduction, sample_rate_reduction, ceil_mode=True
        )
        self.queries = nn.Embedding(num_constant_queries, emb_dim)
        self.decoder = nn.TransformerEncoder(
            transformer_layer, num_layers=num_layers_after
        )
        self.head = nn.Linear(emb_dim, out_dim)

    def forward(self, batch, return_dict=False):
        # audio_feats is [B, T, F]
        embs = batch["audio_feats.pth"].to(self.dtype)
        B, T, F = embs.shape
        # attention mask is [B, T]
        attention_mask = batch["audio_feats_attention_mask.pth"]
        embs = self.emb_proj(embs)
        # True for paddings, false for real ones
        # [B, T+num_constant_queries, E]
        embs = torch.cat([self.queries.weight.repeat(B, 1, 1), embs], dim=1)
        attention_mask = torch.nn.functional.pad(
            attention_mask, (self.num_constant_queries, 0), value=True
        )
        embs = self.encoder(embs, src_key_padding_mask=~attention_mask)
        assert embs.shape[1] == self.num_constant_queries + T, f"{embs.shape=}, {T=}"
        # [B, ceil(T/sample_rate_reduction), E]
        queries_embs = embs[:, : self.num_constant_queries]
        queries_attention_mask = attention_mask[:, : self.num_constant_queries]
        embs = embs[:, self.num_constant_queries :]
        attention_mask = attention_mask[:, self.num_constant_queries :]
        embs = self.sr_reduction(embs.transpose(1, 2)).transpose(1, 2)
        attention_mask = (
            self.sr_reduction(attention_mask.unsqueeze(1).float()).squeeze(1) > 0
        )
        # adding queries
        embs = torch.cat([queries_embs, embs], dim=1)
        attention_mask = torch.cat([queries_attention_mask, attention_mask], dim=1)
        assert (
            attention_mask.shape[1] == embs.shape[1]
        ), f"{embs.shape=}, {attention_mask.shape=}"
        # [B, num_constant_queries + ceil(T/sample_rate_reduction), E]
        embs = self.decoder(embs, src_key_padding_mask=~attention_mask)
        embs = self.head(embs)
        if return_dict:
            return {
                "audiollm_embeds.pth": embs,
                "audiollm_embeds_attention_mask.pth": attention_mask,
            }
        else:
            return embs, attention_mask
