# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import logging
import torch

from fairseq.tasks import register_task

from ..base_task import BaseTask, BaseTaskConfig
from ...data.vl_data.nlvr2_dataset import Nlvr2Dataset
from ...metrics import Accuracy

logger = logging.getLogger(__name__)


@dataclass
class Nlvr2Config(BaseTaskConfig):
    pass


@register_task("nlvr2", dataclass=Nlvr2Config)
class Nlvr2Task(BaseTask):
    def __init__(self, cfg, dictionary):
        super().__init__(cfg, dictionary)
        self.metric = Accuracy()

    def load_dataset(self, split, epoch=1, **kwargs):
        dataset = super().load_dataset(split, epoch, **kwargs)

        self.datasets[split] = Nlvr2Dataset(
            split,
            dataset,
            self.bpe,
            self.dict,
            max_src_length=self.cfg.max_src_length,
            patch_image_size=self.cfg.patch_image_size
        )

    @torch.no_grad()
    def eval_step(self, model, sample):
        logits = model(**sample['net_input'])
        ids = torch.tensor(sample['id']).to(logits.device)
        self.metric.compute(ids, logits, sample['target'])
