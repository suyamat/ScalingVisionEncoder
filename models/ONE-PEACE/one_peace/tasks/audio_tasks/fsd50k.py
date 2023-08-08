# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import torch

from fairseq.tasks import register_task

from ..base_task import BaseTask, BaseTaskConfig
from ...data.audio_data.fsd50k import FSD50KDataset
from ...metrics import MAP

logger = logging.getLogger(__name__)


@dataclass
class FSD50KConfig(BaseTaskConfig):
    pass


@register_task("fsd50k", dataclass=FSD50KConfig)
class FSD50KTask(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = MAP()

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        self.datasets[split] = FSD50KDataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_duration=self.cfg.max_duration,
            feature_encoder_spec=self.cfg.feature_encoder_spec,
            num_classes=self.cfg.num_classes
        )

    @torch.no_grad()
    def eval_step(self, model, sample):
        logits = model(**sample['net_input'])
        ids = torch.tensor(sample['id']).to(logits.device)
        self.metric.compute(ids, logits, sample['target'])
