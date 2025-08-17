import logging
import json
import contextlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from pathlib import Path
from transformers import LlamaTokenizer, StoppingCriteriaList, GenerationConfig

from peft import LoraConfig, TaskType, get_peft_model, PeftModel


from torchmetrics import Metric
from torchmetrics.aggregation import MeanMetric

from pytorch_lightning import LightningModule

#from models.utils import StoppingCriteriaSub
from cryfish.pl_module.concat_mm_embs import zip_embeddings


class CryFishDecoder(LightningModule):
    # enable loading from ckpt without LLM weights
    strict_loading = False

    @property
    def device(self):
        return list(self.parameters())[0].device

    def __init__(
        self,
        llm_model,
        connector_model,
        optimizers_config=None,
        llm_tokenizer=None,
        peft_config_or_path=None,
        debug_text_dir=None,
        llm_token_embs=None,
        save_text_every_n_step=1,
        DEBUG=False,
    ):
        super().__init__()
        self.DEBUG = DEBUG
        self.connector_model = connector_model
        if llm_token_embs is None:
            llm_token_embs = deepcopy(llm_model.get_input_embeddings())
            llm_model.set_input_embeddings(None)
            for p in llm_token_embs.parameters():
                p.requires_grad = False
        self.llm_token_embs = llm_token_embs

        if peft_config_or_path is not None:
            logging.info(
                "Peft config is found. Freeze llm parameters and initializing peft"
            )
            for name, param in llm_model.named_parameters():
                param.requires_grad = False
            if isinstance(peft_config_or_path, str):
                logging.info(f"Loading peft from {peft_config_or_path}")
                self.llm_model = PeftModel.from_pretrained(
                    llm_model, peft_config_or_path
                )
            else:
                self.llm_model = get_peft_model(llm_model, peft_config_or_path)
        else:
            logging.info("Peft config is None. Finetune all model parameters")
            self.llm_model = llm_model

        self.llm_model.print_trainable_parameters()
        self.llm_tokenizer = llm_tokenizer
        self.debug_text_dir = debug_text_dir
        self.save_text_every_n_step = save_text_every_n_step
        self.validation_step_outputs = []
        self.generate_cfg = None
        self.optimizers_config = optimizers_config
        self.max_len_of_generated_text = None
        self.samples_processed = 0
        self.train_loss = []

    def set_optimizers_config(self, optimizers_config):
        self.optimizers_config = optimizers_config

    def get_trainable_parameters(self, connector=True, llm=True, embs=False):
        parameters = []
        if connector:
            parameters.extend(
                [p for p in self.connector_model.parameters() if p.requires_grad]
            )
        if llm:
            parameters.extend(
                [p for p in self.llm_model.parameters() if p.requires_grad]
            )
        if embs:
            parameters.extend(
                [p for p in self.llm_token_embs.parameters() if p.requires_grad]
            )
        return parameters

    def configure_optimizers(self):
        return self.optimizers_config

    def _encode_auditory_feature(self, **kwargs):
        return self.connector_model(**kwargs)

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        raise DeprecationWarning("This method is deprecated")

    def concatenate_text_and_speech(
        self,
        audiollm_embeds,
        prompt_list_tokens_ids,
        audio_id2sample_id,
        audiollm_attention_mask=None,
        text_tokens_ids=None,
        text_attention_mask=None,
    ):
        """
        audiollm_embeds is [B+C, T, D] tensor,
        prompt_list_tokens_ids is List, len=B, every element is [left_prompt_tokens_ids, *delta_tokens_ids, right_prompt_tokens_ids]
        audio_id2sample_id is [B] list.
        audiollm_attention_mask is [B+C, T] bool tensor
        text_tokens_ids is [B, L3] long tensor,
        text_attention_mask is [B, L3] bool tensor
        """
        if audiollm_embeds is not None:
            prompt_list_embs = [
                [self.llm_token_embs(p) for p in element]
                for element in prompt_list_tokens_ids
            ]
            BC, T, D = audiollm_embeds.shape
            if audiollm_attention_mask is None:
                audiollm_attention_mask = audiollm_embeds.ones(
                    (BC, T), dtype=torch.bool
                )

            logging.debug(f"Starts zip embeddings {audiollm_embeds.shape=}")
            input_embeds, input_attention_mask = zip_embeddings(
                prompt_list_embs,
                audiollm_embeds,
                audio_id2sample_id,
                audiollm_attention_mask,
            )
        else:
            assert all(
                len(e) == 1 for e in prompt_list_tokens_ids
            ), f"{prompt_list_tokens_ids=}"
            prompt_tokens_ids = [e[0] for e in prompt_list_tokens_ids]
            input_ids = prompt_tokens_ids[0].new_zeros(
                (len(prompt_tokens_ids), max(e.shape[0] for e in prompt_tokens_ids))
            )
            input_attention_mask = input_ids.new_zeros(input_ids.shape, dtype=bool)
            for i, e in enumerate(prompt_tokens_ids):
                input_ids[i, -len(e) :] = e
                input_attention_mask[i, -len(e) :] = True
            input_embeds = self.llm_token_embs(input_ids)
            logging.debug(
                f"Batch w\o audio contains {input_attention_mask.sum()=} tokens"
            )

        logging.debug(f"Embeds text token ids")
        if text_tokens_ids is not None and text_attention_mask is None:
            logging.debug("text_attention_mask is None")
            text_attention_mask = text_tokens_ids.new_ones(
                (text_tokens_ids.shape[0], text_tokens_ids.shape[1]), dtype=torch.bool
            )

        if text_tokens_ids is not None:
            text_embs = self.llm_token_embs(text_tokens_ids)
            input_embeds = torch.cat([input_embeds, text_embs], dim=1)
            input_attention_mask = torch.cat(
                [input_attention_mask, text_attention_mask], dim=1
            )
        else:
            text_embs = None
        position_ids = (torch.cumsum(input_attention_mask, dim=1) - 1).clamp(
            min=0
        )  # skip all paddings
        return input_embeds, input_attention_mask, position_ids  # , labels

    def forward(self, batch):
        """
        batch = {
        "audio_feats.pth": [BxTxFeats] audio feats,
        "audio_feats_attention_mask.pth": [BxT] attention mask. 1 not mask, 0 - masking this element
        "audio_id2sample_id.pth": [B] list of sample ids
        "prompt_list_tokens_ids.pth" : List, len=B, every element is [left_prompt_tokens_ids, *delta_tokens_ids, right_prompt_tokens_ids]
        "text_tokens_ids.pth": [B x L]
        "text_attention_mask.pth": [B x L]
        }
        """
        if "audio_feats.pth" in batch:
            logging.debug(f"Processing {batch['audio_feats.pth'].shape=} audio feats")
            audiollm_embeds, audiollm_attention_mask = self.connector_model(batch)
        else:
            audiollm_embeds = None
            audiollm_attention_mask = None
        #            audio_feats=batch["audio_feats.pth"],
        #            feats_attention_mask=batch.get("feats_attention_mask.pth", None),
        #        )
        labels = batch.get("text_tokens_ids.pth", None)
        logging.debug(f"Concat {len(batch['prompt_list_tokens_ids.pth'])=} prompts")
        inputs_embeds, attention_mask, position_ids = self.concatenate_text_and_speech(
            audiollm_embeds=audiollm_embeds,
            prompt_list_tokens_ids=batch["prompt_list_tokens_ids.pth"],
            audio_id2sample_id=batch.get("audio_id2sample_id.pth", None),
            text_tokens_ids=labels,
            audiollm_attention_mask=audiollm_attention_mask,
            text_attention_mask=batch.get("text_attention_mask.pth", None),
        )
        logging.debug(f"Forward {labels.shape=}, {inputs_embeds.shape=}")
        if labels is not None:
            # added last promt token into labels
            # labels = torch.cat([-100, labels], dim=1)
            labels = torch.nn.functional.pad(labels, (1, 0), value=-100)
            # Default torch CE implementation ignores -100, not 0
            # assert (
            #    self.llm_token_embs.padding_idx == 0
            # ), "Carefully. Remove it if you are sure that everything is OK"
            labels = labels.masked_fill(labels == self.llm_tokenizer.pad_token_id, -100)
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            labels=labels,  # carefull, labels are not shifted yet
            num_logits_to_keep=labels.shape[1] if labels is not None else 0,
        )
        logging.debug(f"Forward ends, {outputs.logits.shape=}, {outputs.loss=}")
        # DEBUG
        if self.DEBUG:
            torch.save(
                {
                    "batch": batch,
                    "labels": labels,
                    "audiollm_embeds": audiollm_embeds,
                    "audiollm_attention_mask": audiollm_attention_mask,
                    "outputs": outputs,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                f'tmp/{batch["__key__"][0]}.pth',
            )
        return outputs

    def training_step(self, batch, batch_idx):
        logging.debug(
            f"Process {batch_idx}, {len(batch['prompt_list_tokens_ids.pth'])=}"
        )
        outputs = self.forward(batch)
        loss = outputs.loss
        self.log(
            "loss_train",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "num_samples",
            batch["text_tokens_ids.pth"].shape[0],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )

        self.samples_processed += batch["text_tokens_ids.pth"].shape[0]
        self.log(
            "samples_processed",
            self.samples_processed,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=1,
            reduce_fx=sum,
        )

        self.train_loss.append(loss.detach())
        if (batch_idx + 1) % 500 == 0:
            train_loss = sum(self.train_loss) / len(self.train_loss)
            self.log(
                "500avg_loss",
                train_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            self.train_loss.clear()

        if "audio_feats.pth" in batch:
            self.log(
                "num_audio_feats",
                batch["audio_feats.pth"].shape[0] * batch["audio_feats.pth"].shape[1],
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                # batch_size=batch["audio_feats.pth"].shape[0],
            )
        logging.debug(f"loss_train for {batch_idx} is {loss.item()}")
        return {
            "loss": loss,
        }

    @torch.no_grad()
    @torch.inference_mode()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # logging.debug(f"valid started for {batch_idx}")
        outputs = self.forward(batch)
        btz = len(batch["prompt_list_tokens_ids.pth"])
        self.log(
            "loss_valid",
            outputs.loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=btz,
        )
        if (
            self.debug_text_dir is not None
            and batch_idx % self.save_text_every_n_step == 0
        ):
            # remove last symbol, because  rprompt applied as a first item in labels in the forward
            # removes the last because the rprompt[-1] is used as the first element in labels in the forward
            preds = outputs.logits.argmax(dim=-1)[:, :-1]
            self.validation_step_outputs.append(
                (
                    batch_idx,
                    batch["text_tokens_ids.pth"],
                    preds.cpu(),
                    outputs.loss.cpu().item(),
                )
            )

    def on_validation_epoch_end(self):
        if self.debug_text_dir is not None:
            Path(self.debug_text_dir).mkdir(exist_ok=True)
            fname = str(time.strftime("%Y-%m-%d-%H:%M:%S.txt", time.gmtime()))
            with open(f"{self.debug_text_dir}/{fname}", "w") as f:
                for batch_idx, ref, hyp, loss in self.validation_step_outputs:
                    f.write(f"{batch_idx=} {loss=}\n")
                    for ref, hyp in zip(ref, hyp):
                        if self.llm_tokenizer is not None:
                            ref_txt = self.llm_tokenizer.decode(ref)
                            hyp_txt = self.llm_tokenizer.decode(hyp)
                            f.write(f"{ref_txt=}\n{hyp_txt=}\n")
                        ref_int = " ".join(f"{r.item()}" for r in ref)
                        hyp_int = " ".join(f"{h.item()}" for h in hyp)
                        f.write(f"{ref_int=}\n{hyp_int=}\n\n")
        self.validation_step_outputs.clear()  # free memory

    def prepare_for_generating(
        self, generate_cfg=None, llm_tokenizer=None, eos_token=None, max_length=None
    ):
        """
        eos_token is a eos token or a list of EOS tokens
        If you know token ids you can pass it directly to the generate_cfg['eos_token_id']
        """
        self.max_len_of_generated_text = max_length
        if llm_tokenizer is not None:
            self.llm_tokenizer = llm_tokenizer
        if eos_token is not None:
            eos_token_id = self.llm_tokenizer.convert_tokens_to_ids(eos_token)
            logging.info(f"Stopping {eos_token=}, ids is {eos_token_id=}")
            assert isinstance(generate_cfg, dict) and not generate_cfg.get(
                "eos_token_id", None
            ), f"cannot reassign eos_token_id in {generate_cfg=}"
            generate_cfg["eos_token_id"] = eos_token_id
        if generate_cfg is None:
            self.generate_cfg = None
        elif isinstance(generate_cfg, GenerationConfig):
            self.generate_cfg = generate_cfg
        else:
            self.generate_cfg = GenerationConfig(**generate_cfg)

        self.llm_model.set_input_embeddings(self.llm_token_embs)
        assert (
            self.llm_tokenizer is not None
        ), "Tokenizer must be specified in __init_ or in prepare_for_generating"

    @torch.no_grad()
    @torch.inference_mode()
    def predict_step(self, batch, batch_idx=None, dataloader_idx=0):
        assert self.llm_tokenizer is not None, "Cannot generate text without tokenizer!"
        if "audio_feats.pth" in batch:
            audiollm_embeds, audiollm_attention_mask = self.connector_model(batch)
            logging.debug(f"{audiollm_embeds.shape=}, {audiollm_attention_mask.shape=}")
        else:
            audiollm_embeds = None
            audiollm_attention_mask = None

        # text_tokens_ids = batch['prompt_tokens_ids'][0].new_full((B, 1), self.llm_tokenizer.bos_token_id)
        inputs_embeds, attention_mask, position_ids = self.concatenate_text_and_speech(
            prompt_list_tokens_ids=batch["prompt_list_tokens_ids.pth"],
            audiollm_embeds=audiollm_embeds,
            audio_id2sample_id=batch.get("audio_id2sample_id.pth", None),
            audiollm_attention_mask=audiollm_attention_mask,
        )
        logging.debug(
            "".join(
                f"{k}.shape={v.shape}"
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            )
        )
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            generation_config=self.generate_cfg,
            max_length=self.max_len_of_generated_text,
            # stopping_criteria=self.stopping_criteria,
        )
        logging.debug(f"For batch {batch['__key__']}, {outputs.shape=}, {outputs}")
        text = self.llm_tokenizer.batch_decode(outputs, add_special_tokens=False)
        logging.debug(f"hyp: {text}")
        predicted = outputs.unsqueeze(0) if len(outputs.shape) == 1 else outputs
        meta = {
            k: v for k, v in batch.items() if k.startswith("__") or k.endswith(".txt")
        }
        return {
            **meta,
            "predicted.txt": text,
            "predicted.pth": predicted,
        }

    def on_save_checkpoint(self, checkpoint):
        state = checkpoint["state_dict"]
        param_grad_dic = {k: v.requires_grad for (k, v) in self.named_parameters()}
        for name in list(state.keys()):
            # saving only trainable parameters
            if not param_grad_dic[name]:
                state.pop(name)

    def inplace_load_from_checkpoint(self, ckpt_path):
        data = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missed, unexpected = self.load_state_dict(data, strict=self.strict_loading)
        logging.debug(f"Loaded model from {ckpt_path}\n{missed=}\n{unexpected=}")
        return self


