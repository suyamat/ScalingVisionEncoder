# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

"""
ONE-PEACE Pretrain
"""
from typing import Optional
from dataclasses import dataclass

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model
from fairseq.distributed import fsdp_wrap
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from ..unify_model_config import UnifyModelConfig
from .one_peace_base import ModelWrapper, OnePeaceBaseModel, init_one_peace_params
from ..components import Linear, trunc_normal_

logger = logging.getLogger(__name__)


@dataclass
class OnePeacePretrainConfig(UnifyModelConfig):
    reset_logit_scale: bool = False
    logit_scale_init: float = 1 / 0.07
    stage2_pretrain: bool = False


@register_model("one_peace_pretrain", dataclass=OnePeacePretrainConfig)
class OnePeacePretrainModel(OnePeaceBaseModel):

    def __init__(self, cfg: OnePeacePretrainConfig, src_dict):
        super().__init__(cfg, src_dict)

        enc_embed_dim = cfg.encoder.embed_dim
        dec_embed_dim = cfg.decoder.embed_dim
        self.encoder_wrapper = ModelWrapper(cfg.encoder, src_dict)
        self.decoder_wrapper = ModelWrapper(cfg.decoder)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(cfg.logit_scale_init))

        if cfg.encoder.use_text_moe:
            self.text_proj = Linear(enc_embed_dim, enc_embed_dim)
        if cfg.encoder.use_image_moe:
            self.image_proj = Linear(enc_embed_dim, enc_embed_dim)
        if cfg.encoder.use_audio_moe:
            self.audio_proj = Linear(enc_embed_dim, enc_embed_dim)

        self.text_mask_token = None
        if cfg.encoder.use_text_moe and cfg.decoder.use_text_moe:
            self.decoder_text_embed = Linear(enc_embed_dim, dec_embed_dim)
            self.text_mask_token = nn.Parameter(torch.zeros(1, dec_embed_dim))
            self.text_mask_head = Linear(dec_embed_dim, enc_embed_dim)
            trunc_normal_(self.text_mask_token)

        self.image_mask_token = None
        if cfg.encoder.use_image_moe and cfg.decoder.use_image_moe:
            self.decoder_image_embed = Linear(enc_embed_dim, dec_embed_dim)
            self.image_mask_token = nn.Parameter(torch.zeros(1, dec_embed_dim))
            self.image_mask_head = Linear(dec_embed_dim, enc_embed_dim)
            trunc_normal_(self.image_mask_token)

        self.audio_mask_token = None
        if cfg.encoder.use_audio_moe and cfg.decoder.use_audio_moe:
            self.decoder_audio_embed = Linear(enc_embed_dim, dec_embed_dim)
            self.audio_mask_token = nn.Parameter(torch.zeros(1, dec_embed_dim))
            self.audio_mask_head = Linear(dec_embed_dim, enc_embed_dim)
            trunc_normal_(self.audio_mask_token)

        self.apply(init_one_peace_params)

        for i, layer in enumerate(self.encoder_wrapper.fusion_model.layers):
            if (i + 1) % cfg.encoder.fsdp_checkpoint_wrap_layer_preserve_frequency != 0:
                continue
            if (i + 1) % cfg.encoder.fsdp_checkpoint_wrap_layer_skip_frequency == 0:
                continue
            if cfg.encoder.checkpoint_activations:
                self.encoder_wrapper.fusion_model.layers[i] = fsdp_wrap(checkpoint_wrapper(layer))
            else:
                self.encoder_wrapper.fusion_model.layers[i] = fsdp_wrap(layer)

        for i, layer in enumerate(self.decoder_wrapper.fusion_model.layers):
            if (i + 1) % cfg.decoder.fsdp_checkpoint_wrap_layer_preserve_frequency != 0:
                continue
            if (i + 1) % cfg.decoder.fsdp_checkpoint_wrap_layer_skip_frequency == 0:
                continue
            if cfg.decoder.checkpoint_activations:
                self.decoder_wrapper.fusion_model.layers[i] = fsdp_wrap(checkpoint_wrapper(layer))
            else:
                self.decoder_wrapper.fusion_model.layers[i] = fsdp_wrap(layer)

        if cfg.stage2_pretrain:
            self.text_proj.requires_grad_(False)
            self.encoder_wrapper.requires_grad_(False)
            self.encoder_wrapper.audio_adapter.requires_grad_(True)
            self.encoder_wrapper.fusion_model.audio_layer_norm.requires_grad_(True)
            for layer in self.encoder_wrapper.fusion_model.layers:
                layer.audio_ffn.requires_grad_(True)

    def forward(
        self,
        src_tokens: Optional[torch.Tensor] = None,
        text_preserve_ids: Optional[torch.Tensor] = None,
        src_images: Optional[torch.Tensor] = None,
        image_preserve_ids: Optional[torch.Tensor] = None,
        src_audios: Optional[torch.Tensor] = None,
        audio_padding_masks: Optional[torch.Tensor] = None,
        audio_preserve_ids: Optional[torch.Tensor] = None,
        encoder_type: str = None,
        return_logit_scale: bool = False
    ):
        if return_logit_scale:
            with torch.no_grad():
                self.logit_scale.clamp_(0, math.log(100))
            logit_scale_exp = self.logit_scale.exp()
            return logit_scale_exp
        else:
            enc_image_features = self.encoder_wrapper(
                src_tokens=src_tokens, text_preserve_ids=text_preserve_ids,
                src_images=src_images, image_preserve_ids=image_preserve_ids,
                src_audios=src_audios, audio_padding_masks=audio_padding_masks, audio_preserve_ids=audio_preserve_ids,
                encoder_type=encoder_type
            )
            
            return enc_image_features


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        if self.cfg.reset_logit_scale:
            del state_dict['logit_scale']
        if self.cfg.stage2_pretrain:
            del state_dict['image_mask_token']
            del state_dict['image_mask_head.weight']
            del state_dict['image_mask_head.bias']
            for param_name in list(state_dict.keys()):
                if 'image_' in param_name:
                    del state_dict[param_name]

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict:
                logger.info('{} not exists, re-initialized'.format(prefix + param_name))
                state_dict[prefix + param_name] = self.state_dict()[param_name]