# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a network across multiple GPUs.
"""

import contextlib
import logging
import os
import sys
import time
from itertools import chain
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

from fairseq import checkpoint_utils, models, optim, utils
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler

from .utils.layer_decay import LayerDecayValueAssigner, get_parameter_groups
from .utils.ema_module import EMAModule
from .optim import FP16Optimizer, MemoryEfficientFP16Optimizer, AMPOptimizer

logger = logging.getLogger(__name__)


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, cfg, task, model, criterion):
        self.cfg = cfg
        self.task = task

        # catalog shared parameters
        shared_params = _catalog_shared_params(model)
        self.cuda = torch.cuda.is_available() and not cfg.common.cpu

        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.is_fsdp:
            import fairscale

            if self.cfg.distributed_training.zero_sharding != "none":
                raise ValueError(
                    "FullyShardedDataParallel is not compatible with --zero-sharding "
                    "option (it's already built in)"
                )
            if (
                max(self.cfg.optimization.update_freq) > 1
                and fairscale.__version__ < "0.4.0"
            ):
                raise RuntimeError(
                    "Please update to fairscale 0.4.0 or newer when combining "
                    "--update-freq with FullyShardedDataParallel"
                )
            if self.use_sharded_state:
                assert (
                    fairscale.__version__ >= "0.3.9"
                ), "--use-sharded-state requires newer fairscale. pip install -U fairscale"
        else:
            if self.cfg.distributed_training.cpu_offload:
                raise ValueError("--cpu-offload requires --ddp-backend=fully_sharded")

        # copy model and criterion to current device/dtype
        self._criterion = criterion
        self._model = model
        if not self.is_fsdp:
            if cfg.common.bf16:
                self._criterion = self._criterion.to(dtype=torch.bfloat16)
                self._model = self._model.to(dtype=torch.bfloat16)
                # if hasattr(self._model, 'logit_scale'):
                #     self._model.logit_scale.data = self._model.logit_scale.data.float()
            if cfg.common.fp16:
                assert not cfg.common.amp, "Cannot use fp16 and AMP together"
                self._criterion = self._criterion.half()
                self._model = self._model.half()
                # if hasattr(self._model, 'logit_scale'):
                #     self._model.logit_scale.data = self._model.logit_scale.data.float()
            elif cfg.common.amp:
                self._amp_retries = 0
        if (
            not cfg.distributed_training.pipeline_model_parallel
            # the DistributedFairseqModel wrapper will handle moving to device,
            # so only handle cases which don't use the wrapper
            and not self.use_distributed_wrapper
        ):
            self._criterion = self._criterion.to(device=self.device)
            self._model = self._model.to(device=self.device)

        # check that shared parameters are preserved after device transfer
        for shared_param in shared_params:
            ref = _get_module_by_path(self._model, shared_param[0])
            for path in shared_param[1:]:
                logger.info(
                    "detected shared parameter: {} <- {}".format(shared_param[0], path)
                )
                _set_module_by_path(self._model, path, ref)

        self._dummy_batch = None  # indicates we don't have a dummy batch at first
        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._warn_once = set()
        self._wrapped_criterion = None
        self._wrapped_model = None
        self._ema = None

        if self.cuda and self.data_parallel_world_size > 1:
            self._grad_norm_buf = torch.cuda.DoubleTensor(self.data_parallel_world_size)
        else:
            self._grad_norm_buf = None

        # get detailed cuda environment
        if self.cuda:
            self.cuda_env = utils.CudaEnvironment()
            if self.data_parallel_world_size > 1:
                self.cuda_env_arr = distributed_utils.all_gather_list(
                    self.cuda_env, group=distributed_utils.get_global_group()
                )
            else:
                self.cuda_env_arr = [self.cuda_env]
            if self.data_parallel_rank == 0:
                utils.CudaEnvironment.pretty_print_cuda_env_list(self.cuda_env_arr)
        else:
            self.cuda_env = None
            self.cuda_env_arr = None

        metrics.log_start_time("wall", priority=790, round=0)

        self._start_time = time.time()
        self._previous_training_time = 0
        self._cumulative_training_time = None

    def reinitialize(self):
        """Reinitialize the Trainer, typically after model params change."""
        self._lr_scheduler = None
        self._optimizer = None
        self._wrapped_criterion = None
        self._wrapped_model = None

    @property
    def data_parallel_world_size(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 1
        return distributed_utils.get_data_parallel_world_size()

    @property
    def data_parallel_process_group(self):
        return distributed_utils.get_data_parallel_group()

    @property
    def data_parallel_rank(self):
        if self.cfg.distributed_training.distributed_world_size == 1:
            return 0
        return distributed_utils.get_data_parallel_rank()

    @property
    def is_data_parallel_master(self):
        # NOTE: this returns true for all model parallel replicas with data
        # parallel rank 0
        return self.data_parallel_rank == 0

    @property
    def use_distributed_wrapper(self) -> bool:
        return (self.data_parallel_world_size > 1) or (
            self.is_fsdp and self.cfg.distributed_training.cpu_offload
        )

    @property
    def should_save_checkpoint_on_current_rank(self) -> bool:
        """Indicates whether to save checkpoints on the current DDP rank."""
        if self.is_fsdp and self.cfg.distributed_training.use_sharded_state:
            return True
        else:
            return self.is_data_parallel_master

    @property
    def always_call_state_dict_during_save_checkpoint(self) -> bool:
        if self.is_fsdp and not self.cfg.distributed_training.use_sharded_state:
            # FSDP calls communication collective when consolidating checkpoints
            return True
        else:
            return False

    @property
    def checkpoint_suffix(self) -> str:
        """Suffix to add to the checkpoint file name."""
        if self.is_fsdp and self.cfg.distributed_training.use_sharded_state:
            return self.cfg.checkpoint.checkpoint_suffix + "-shard{0}".format(
                self.data_parallel_rank
            )
        else:
            return self.cfg.checkpoint.checkpoint_suffix or ""

    @property
    def criterion(self):
        if self._wrapped_criterion is None:
            if utils.has_parameters(self._criterion) and self.use_distributed_wrapper:
                self._wrapped_criterion = models.DistributedFairseqModel(
                    self.cfg.distributed_training,
                    self._criterion,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_criterion = self._criterion
        return self._wrapped_criterion

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.use_distributed_wrapper:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.cfg.distributed_training,
                    self._model,
                    process_group=self.data_parallel_process_group,
                    device=self.device,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def ema(self):
        if self._ema is None:
            self._build_ema()
        return self._ema

    def _build_ema(self):
        if self.cfg.ema.store_ema:
            self._ema = EMAModule(self._model, self.cfg.ema, self.device)
            logger.info("Exponential Moving Average Shadow Model is initialized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        if self.cfg.common.no_bias_decay or self.cfg.common.layer_decay != 1.0:
            num_layers = self.model.cfg.encoder.layers
            assigner = LayerDecayValueAssigner(
                list(self.cfg.common.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))
            ) if self.cfg.common.layer_decay < 1.0 else None
            skip_weight_decay_list = self.model.no_weight_decay()
            params = get_parameter_groups(
                self.model,
                self.cfg.optimizer.weight_decay,
                skip_weight_decay_list,
                assigner.get_layer_id if assigner is not None else None,
                assigner.get_scale if assigner is not None else None
            )
        else:
            params = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(self.model.parameters(), self.criterion.parameters()),
                )
            )

        if self.is_fsdp and (self.cfg.common.fp16 or self.cfg.common.bf16):
            self._optimizer = MemoryEfficientFP16Optimizer.build_optimizer(
                self.cfg, params
            )
        elif self.cfg.common.fp16 or self.cfg.common.bf16 or self.cfg.common.amp:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                logger.info(
                    "NOTE: your device does NOT support faster training with --fp16 or --amp, "
                    "please switch to FP32 which is likely to be faster"
                )
            if (
                self.cfg.common.memory_efficient_fp16
                or self.cfg.common.memory_efficient_bf16
            ):
                self._optimizer = MemoryEfficientFP16Optimizer.build_optimizer(
                    self.cfg, params
                )
            elif self.cfg.common.amp:
                self._optimizer = AMPOptimizer.build_optimizer(self.cfg, params)
            else:
                self._optimizer = FP16Optimizer.build_optimizer(self.cfg, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                logger.info(
                    "NOTE: your device may support faster training with --fp16 or --amp"
                )
            self._optimizer = optim.build_optimizer(self.cfg.optimizer, params)

        if self.is_fsdp:
            assert self._optimizer.supports_flat_params, (
                "--ddp-backend=fully_sharded is only compatible with pointwise "
                "optimizers (e.g., Adam, AdamW, Adadelta, Adamax, SGD, etc.). "
                "However, the sharding will result in slightly different results when "
                "using non-pointwise optimizers (e.g., Adagrad, Adafactor, LAMB)"
            )

        if self.cfg.distributed_training.zero_sharding == "os":
            if (
                self.cfg.common.fp16
                and not self.cfg.common.memory_efficient_fp16
                and not self.cfg.common.memory_efficient_bf16
            ) and not self.cfg.common.fp16_no_flatten_grads:
                raise ValueError(
                    "ZeRO is incompatible with fp16 and flattened grads. "
                    "Please use --fp16-no-flatten-grads"
                )
            else:
                optim.shard_(self._optimizer, self.data_parallel_process_group)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(
            self.cfg.lr_scheduler,
            self.optimizer,
        )
        self._lr_scheduler.step_update(0)

    @property
    def is_fsdp(self):
        return self.cfg.distributed_training.ddp_backend == "fully_sharded"

    @property
    def use_sharded_state(self):
        return self.cfg.distributed_training.use_sharded_state

    def consolidate_optimizer(self):
        """For OSS, we need to consolidate the state dict."""
        if self.cfg.checkpoint.no_save_optimizer_state:
            return
        self._gathered_optim_state = None
        if hasattr(self.optimizer.optimizer, "consolidate_state_dict"):
            self.optimizer.optimizer.consolidate_state_dict()
        elif self.is_fsdp and not self.use_sharded_state:
            st = self.model.gather_full_optim_state_dict(
                self.optimizer
            )  # only returns on rank 0
            self._gathered_optim_state = st

    def state_dict(self):
        state_dict = {
            "args": None,  # legacy
            "cfg": (
                OmegaConf.to_container(self.cfg, resolve=True, enum_to_str=True)
                if OmegaConf.is_config(self.cfg)
                else self.cfg
            ),
            "model": self.model.state_dict(),
            "criterion": (
                self.criterion.state_dict()
                if utils.has_parameters(self.criterion)
                else None
            ),
            "optimizer_history": (self._optim_history or [])
            + [
                {
                    "criterion_name": self.get_criterion().__class__.__name__,
                    "optimizer_name": self.optimizer.__class__.__name__,
                    "lr_scheduler_state": self.lr_scheduler.state_dict(),
                    "num_updates": self.get_num_updates(),
                }
            ],
            "task_state": self.task.state_dict() if self.task is not None else {},
            "extra_state": {
                "metrics": metrics.state_dict(),
                "previous_training_time": self.cumulative_training_time(),
            },
        }
        if self.cfg.ema.store_ema:
            # Save EMA model state as extra state
            state_dict["extra_state"]["ema"] = self.ema.get_model().state_dict()
            if self.cfg.ema.ema_fp32:
                # Save EMA params in fp32
                state_dict["extra_state"]["ema_fp32_params"] = self.ema.fp32_params
        if not self.cfg.checkpoint.no_save_optimizer_state:
            if self._gathered_optim_state is not None:
                state_dict["last_optimizer_state"] = self._gathered_optim_state
                self._gathered_optim_state = None
            else:
                state_dict["last_optimizer_state"] = self.optimizer.state_dict()
        if self.is_fsdp:
            # save meta data for recombining checkpoint upon loading
            state_dict["fsdp_metadata"] = self.model.local_metadata_dict()
        return state_dict

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""

        logger.info(f"Saving checkpoint to {os.path.abspath(filename)}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)
        if self.should_save_checkpoint_on_current_rank:
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously
            )
        logger.info(f"Finished saving checkpoint to {os.path.abspath(filename)}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)
        if bexists:
            load_on_all_ranks = (
                self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # FSDP requires loading checkpoint shards on all ranks
                or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(
                    filename, load_on_all_ranks=load_on_all_ranks
                )
                last_optim_state = state.get("last_optimizer_state", None)

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory exploding.
                if (
                    not load_on_all_ranks
                    and self.cfg.distributed_training.zero_sharding == "os"
                    and "last_optimizer_state" in state
                    and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            # load model parameters
            try:
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    self.model.set_num_updates(
                        state["optimizer_history"][-1]["num_updates"]
                    )

                self.model.load_state_dict(
                    state["model"], strict=True, model_cfg=self.cfg.model
                )
                # save memory for later steps
                if not self.cfg.ema.store_ema:
                    del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(
                        state["criterion"], strict=True
                    )
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"] if "extra_state" in state else None
            self._optim_history = state["optimizer_history"] if "optimizer_history" in state else []

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), (
                f"Criterion does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['criterion_name']} vs "
                f"{self.get_criterion().__class__.__name__}"
            )
            assert last_optim["optimizer_name"] == self.optimizer.__class__.__name__, (
                f"Optimizer does not match; please reset the optimizer "
                f"(--reset-optimizer). {last_optim['optimizer_name']} vs "
                f"{self.optimizer.__class__.__name__}"
            )

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if self.is_fsdp and not self.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded, skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(
                    last_optim_state
                )
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(
                    last_optim_state
                )

            if not self.cfg.checkpoint.no_load_optim_state:
                self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)
                logger.info(f"Loaded optim_state for {filename}")
            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if (
                itr_state.get("version", 1) >= 2
                and itr_state["iterations_in_epoch"] == 0
            ):
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. "
                        "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(
                        state["model"], build_fp32_params=self.cfg.ema.ema_fp32
                    )
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info(
                                "Building EMA fp32 params from EMA model in checkpoint"
                            )
                            self.ema.build_fp32_params()

            logger.info(
                "Loaded checkpoint {} (epoch {} @ {} updates)".format(
                    filename, epoch, self.get_num_updates()
                )
            )

        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state

    def get_train_iterator(
        self,
        epoch,
        combine=True,
        load_dataset=True,
        data_selector=None,
        shard_batch_itr=True,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over the training set for a given epoch."""
        if load_dataset:
            logger.info("loading train data for epoch {}".format(epoch))
            self.task.load_dataset(
                self.cfg.dataset.train_subset,
                epoch=epoch,
                combine=combine,
                data_selector=data_selector
            )
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(self.cfg.dataset.train_subset),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
                self.cfg.dataset.max_tokens,
            ),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=(self.cfg.common.seed + epoch)
            if self.cfg.dataset.update_ordered_indices_seed
            else self.cfg.common.seed,
            num_shards=self.data_parallel_world_size if shard_batch_itr else 1,
            shard_id=self.data_parallel_rank if shard_batch_itr else 0,
            num_workers=self.cfg.dataset.num_workers,
            epoch=epoch,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=self.cfg.optimization.skip_remainder_batch,
            ensure_equal_batch=self.cfg.dataset.ensure_equal_batch,
            disable_shuffling=False
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def get_valid_iterator(
        self,
        subset,
        disable_iterator_cache=False,
    ):
        """Return an EpochBatchIterator over given validation subset for a given epoch."""
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.dataset(subset),
            max_tokens=self.cfg.dataset.max_tokens_valid,
            max_sentences=self.cfg.dataset.batch_size_valid,
            max_positions=utils.resolve_max_positions(
                self.task.max_positions(),
                self.model.max_positions(),
            ),
            ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
            seed=self.cfg.common.seed,
            num_shards=self.data_parallel_world_size,
            shard_id=self.data_parallel_rank,
            num_workers=self.cfg.dataset.num_workers,
            # always pass a fixed "epoch" to keep validation data consistent
            # across training epochs
            epoch=1,
            data_buffer_size=self.cfg.dataset.data_buffer_size,
            disable_iterator_cache=disable_iterator_cache,
            skip_remainder_batch=False,
            disable_shuffling=True
        )
        self.reset_dummy_batch(batch_iterator.first_batch)
        return batch_iterator

    def begin_epoch(self, epoch):
        """Called at the beginning of each epoch."""
        logger.info("begin training epoch {}".format(epoch))

        self.lr_step_begin_epoch(epoch)

        # task specific setup per epoch
        self.task.begin_epoch(epoch, self.get_model())

    def begin_valid_epoch(self, epoch, subset):
        """Called at the beginning of each validation epoch."""

        # task specific setup per validation epoch
        self.task.begin_valid_epoch(epoch, self.get_model(), subset)

    def reset_dummy_batch(self, batch):
        self._dummy_batch = batch

    @metrics.aggregate("train")
    def train_step(self, samples, raise_oom=False, empty_cache=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        metrics.log_start_time("train_wall", priority=800, round=0)

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        # if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
        #     extra_kwargs["ema_model"] = self.ema.get_model()

        # forward and backward pass
        logging_outputs, sample_size = [], 0
        for i, sample in enumerate(samples):  # delayed update loop
            sample, is_dummy_batch = self._prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.data_parallel_world_size > 1
                    and hasattr(self.model, "no_sync")
                    and i < len(samples) - 1
                    # The no_sync context manager results in increased memory
                    # usage with FSDP, since full-size gradients will be
                    # accumulated on each GPU. It's typically a better tradeoff
                    # to do the extra communication with FSDP.
                    and not self.is_fsdp
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            metrics.log_start_time("train_wall_step", priority=780, round=0)
            try:
                with maybe_no_sync():
                    # forward and backward
                    loss, sample_size_i, logging_output = self.task.train_step(
                        sample=sample,
                        model=self.model,
                        criterion=self.criterion,
                        optimizer=self.optimizer,
                        update_num=self.get_num_updates(),
                        ignore_grad=is_dummy_batch,
                        **extra_kwargs,
                    )
                    del loss

                logging_outputs.append(logging_output)
                sample_size += sample_size_i

                # emptying the CUDA cache after the first step can
                # reduce the chance of OOM
                if self.cuda and empty_cache:
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                raise e

            metrics.log_stop_time("train_wall_step")

        if is_dummy_batch:
            if torch.is_tensor(sample_size):
                sample_size.zero_()
            else:
                sample_size *= 0.0

        if torch.is_tensor(sample_size):
            sample_size = sample_size.float()
        else:
            sample_size = float(sample_size)

        # gather logging outputs from all replicas
        if self._sync_stats():
            train_time = self._local_cumulative_training_time()
            (
                logging_outputs,
                (
                    sample_size,
                    total_train_time,
                ),
            ) = self._aggregate_logging_outputs(
                logging_outputs, sample_size, train_time, ignore=is_dummy_batch
            )
            self._cumulative_training_time = (
                total_train_time / self.data_parallel_world_size
            )

        overflow = False
        try:
            with torch.autograd.profiler.record_function("reduce-grads"):
                # reduce gradients across workers
                self.optimizer.all_reduce_grads(self.model)
                if utils.has_parameters(self.criterion):
                    self.optimizer.all_reduce_grads(self.criterion)

            with torch.autograd.profiler.record_function("multiply-grads"):
                # multiply gradients by (data_parallel_size / sample_size) since
                # DDP normalizes by the number of data parallel workers for
                # improved fp16 precision.
                # Thus we get (sum_of_gradients / sample_size) at the end.
                # In case of fp16, this step also undoes loss scaling.
                # (Debugging note: Some optimizers perform this scaling on the
                # fly, so inspecting model.parameters() or optimizer.params may
                # still show the original, unscaled gradients.)
                numer = self.data_parallel_world_size if self._sync_stats() else 1
                self.optimizer.multiply_grads(numer / (sample_size or 1.0))
                # Note: (sample_size or 1.0) handles the case of a zero gradient, in a
                # way that avoids CPU/device transfers in case sample_size is a GPU or
                # TPU object. The assumption is that the gradient itself is also 0.

            with torch.autograd.profiler.record_function("clip-grads"):
                # clip grads
                grad_norm = self.clip_grad_norm(self.cfg.optimization.clip_norm)

            # check that grad norms are consistent across workers
            if self.cfg.distributed_training.ddp_backend != "slowmo":
                self._check_grad_norms(grad_norm)
            if not torch.isfinite(grad_norm).all():
                # in case of AMP, if gradients are Nan/Inf then
                # optimizer step is still required
                if self.cfg.common.amp:
                    overflow = True
                else:
                    # check local gradnorm single GPU case, trigger NanDetector
                    raise FloatingPointError("gradients are Nan/Inf")

            with torch.autograd.profiler.record_function("optimizer"):
                # take an optimization step
                self.task.optimizer_step(
                    self.optimizer, model=self.model, update_num=self.get_num_updates()
                )
                if self.cfg.common.amp and overflow:
                    if self._amp_retries == self.cfg.common.amp_batch_retries:
                        logger.info("AMP: skipping this batch.")
                        self._amp_retries = 0
                    else:
                        self._amp_retries += 1
                        return self.train_step(
                            samples, raise_oom
                        )  # recursion to feed in same batch

        except FloatingPointError:
            # re-run the forward and backward pass with hooks attached to print
            # out where it fails
            self.zero_grad()
            with NanDetector(self.get_model()):
                for _, sample in enumerate(samples):
                    sample, _ = self._prepare_sample(sample)
                    self.task.train_step(
                        sample,
                        self.model,
                        self.criterion,
                        self.optimizer,
                        self.get_num_updates(),
                        ignore_grad=False,
                        **extra_kwargs,
                    )
            raise
        except OverflowError as e:
            overflow = True
            logger.info(
                f"NOTE: gradient overflow detected, ignoring gradient, {str(e)}"
            )
            grad_norm = torch.tensor(0.0).cuda()
            self.zero_grad()
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._log_oom(e)
                logger.error("OOM during optimization, irrecoverable")
            raise e

        # Some distributed wrappers (e.g., SlowMo) need access to the optimizer
        # after the step
        if hasattr(self.model, "perform_slowmo"):
            self.model.perform_slowmo(
                self.optimizer.optimizer, getattr(self.optimizer, "fp32_params", None)
            )

        logging_output = None
        if not overflow or self.cfg.distributed_training.ddp_backend == "slowmo":
            self.set_num_updates(self.get_num_updates() + 1)

            if self.cfg.ema.store_ema:
                # Step EMA forward with new model.
                self.ema.step(
                    self.get_model(),
                    self.get_num_updates(),
                )
                metrics.log_scalar(
                    "ema_decay",
                    self.ema.get_decay(),
                    priority=10000,
                    round=5,
                    weight=0,
                )

            if self.cuda and self.cuda_env is not None:
                # log minimum free memory over the iteration
                self._log_gpu_mem_stats()

            # log stats
            logging_output = self._reduce_and_log_stats(
                logging_outputs, sample_size, grad_norm
            )

            # clear CUDA cache to reduce memory fragmentation
            if (
                self.cuda
                and self.cfg.common.empty_cache_freq > 0
                and (
                    (self.get_num_updates() + self.cfg.common.empty_cache_freq - 1)
                    % self.cfg.common.empty_cache_freq
                )
                == 0
            ):
                torch.cuda.empty_cache()

        if (self.cfg.common.fp16 or self.cfg.common.amp) and not self.cfg.common.bf16:
            metrics.log_scalar(
                "loss_scale",
                (
                    self.optimizer.scaler.loss_scale
                    if self.cfg.common.fp16
                    else self.optimizer.scaler.get_scale()
                ),
                priority=700,
                round=4,
                weight=0,
            )
            metrics.log_scalar(
                "scale_window",
                (
                    self.optimizer.scaler.scale_window
                    if self.cfg.common.fp16
                    else self.optimizer.scaler.get_growth_interval()
                ),
                priority=700,
                round=4,
                weight=0,
            )

        metrics.log_stop_time("train_wall")
        return logging_output

    @metrics.aggregate("valid")
    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""

        # If EMA is enabled through store_ema=True
        # and task.uses_ema is True, pass the EMA model as a keyword
        # argument to the task.
        extra_kwargs = {}
        if self.cfg.ema.store_ema and getattr(self.task, "uses_ema", False):
            extra_kwargs["ema_model"] = self.ema.get_model()

        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()
            self.zero_grad()

            sample, is_dummy_batch = self._prepare_sample(sample)

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion, is_dummy_batch, **extra_kwargs
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log_oom(e)
                    if not raise_oom:
                        logger.warning(
                            "ran out of memory in validation step, retrying batch"
                        )
                        for p in self.model.parameters():
                            if p.grad is not None:
                                p.grad = None  # free some memory
                        if self.cuda:
                            torch.cuda.empty_cache()
                        return self.valid_step(sample, raise_oom=True)
                raise e

            logging_outputs = [logging_output]
            if is_dummy_batch:
                if torch.is_tensor(sample_size):
                    sample_size.zero_()
                else:
                    sample_size *= 0.0

        # gather logging outputs from all replicas
        if self.data_parallel_world_size > 1:
            logging_outputs, (sample_size,) = self._aggregate_logging_outputs(
                logging_outputs,
                sample_size,
                ignore=is_dummy_batch,
            )

        # log validation stats
        logging_output = self._reduce_and_log_stats(logging_outputs, sample_size)
        return logging_output

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step_begin_epoch(self, epoch):
        """Adjust the learning rate at the beginning of the epoch."""
        self.lr_scheduler.step_begin_epoch(epoch)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_reinit(self, total_updates, num_updates):
        self.lr_scheduler.reinit(total_updates, num_updates)

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate at the end of the epoch."""
        self.lr_scheduler.step(epoch, val_loss)
        # prefer updating the LR based on the number of steps
        return self.lr_step_update()

    def lr_step_update(self):
        """Update the learning rate after each update."""
        new_lr = self.lr_scheduler.step_update(self.get_num_updates())
        if isinstance(new_lr, dict):
            for k, v in new_lr.items():
                metrics.log_scalar(f"lr_{k}", v, weight=0, priority=300)
            new_lr = new_lr.get("default", next(iter(new_lr.values())))
        else:
            metrics.log_scalar("lr", new_lr, weight=0, priority=300)
        return new_lr

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_criterion(self):
        """Get the (non-wrapped) criterion instance."""
        return self._criterion

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates
        self.lr_step_update()
        metrics.log_scalar("num_updates", self._num_updates, weight=0, priority=200)

    def clip_grad_norm(self, clip_norm):
        def agg_norm_fn(total_norm):
            total_norm = total_norm.cuda().float() ** 2
            total_norm = distributed_utils.all_reduce(
                total_norm, group=self.data_parallel_process_group
            )
            return total_norm**0.5

        should_agg_norm = self.is_fsdp and (
            self.data_parallel_process_group is not None
            or torch.distributed.is_initialized()
        )
        return self.optimizer.clip_grad_norm(
            clip_norm, aggregate_norm_fn=agg_norm_fn if should_agg_norm else None
        )

    def cumulative_training_time(self):
        if self._cumulative_training_time is None:
            # single GPU
            return self._local_cumulative_training_time()
        else:
            return self._cumulative_training_time

    def _local_cumulative_training_time(self):
        """Aggregate training time in seconds."""
        return time.time() - self._start_time + self._previous_training_time

    def _fp_convert_sample(self, sample):
        def apply_half(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.half)
            return t

        def apply_bfloat16(t):
            if t.dtype is torch.float32:
                return t.to(dtype=torch.bfloat16)
            return t

        if self.cfg.common.bf16:
            sample = utils.apply_to_sample(apply_bfloat16, sample)

        if self.cfg.common.fp16:
            sample = utils.apply_to_sample(apply_half, sample)

        return sample

    def _prepare_sample(self, sample, is_dummy=False):
        if sample == "DUMMY":
            raise Exception(
                "Trying to use an uninitialized 'dummy' batch. This usually indicates "
                "that the total number of batches is smaller than the number of "
                "participating GPUs. Try reducing the batch size or using fewer GPUs."
            )

        if sample is None or len(sample) == 0:
            assert (
                self._dummy_batch is not None and len(self._dummy_batch) > 0
            ), "Invalid dummy batch: {}".format(self._dummy_batch)
            sample, _ = self._prepare_sample(self._dummy_batch, is_dummy=True)
            return sample, True

        # Given that PCIe/NVLink bandwidth is significantly smaller than DRAM bandwidth
        # it makes sense to do the format conversion on the CPU and then transfer
        # a smaller buffer to the device. This also saves GPU memory capacity.

        if self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self.cuda:
            sample = utils.move_to_cuda(sample)

        if not self.cfg.common.on_cpu_convert_precision:
            sample = self._fp_convert_sample(sample)

        if self._dummy_batch == "DUMMY":
            self._dummy_batch = sample

        return sample, False

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.cfg.common.seed + self.get_num_updates()
        utils.set_torch_seed(seed)

    def _sync_stats(self):
        # Return True if it's using multiple GPUs and DDP
        if self.data_parallel_world_size == 1:
            return False
        else:
            return True

    def _log_oom(self, exc):
        msg = "OOM: Ran out of memory with exception: {}".format(exc)
        logger.warning(msg)
        if torch.cuda.is_available() and hasattr(torch.cuda, "memory_summary"):
            for device_idx in range(torch.cuda.device_count()):
                logger.warning(torch.cuda.memory_summary(device=device_idx))
        sys.stderr.flush()

    def _aggregate_logging_outputs(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        if self.task.__class__.logging_outputs_can_be_summed(self.get_criterion()):
            return self._fast_stat_sync_sum(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )
        else:
            return self._all_gather_list_sync(
                logging_outputs, *extra_stats_to_sum, ignore=ignore
            )

    def _all_gather_list_sync(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. all_gather_list_sync is
        suitable when logging outputs are complex types.
        """
        if ignore:
            logging_outputs = []
        results = list(
            zip(
                *distributed_utils.all_gather_list(
                    [logging_outputs] + list(extra_stats_to_sum),
                    max_size=getattr(self.cfg.common, "all_gather_list_size", 16384),
                    group=self.data_parallel_process_group,
                )
            )
        )
        logging_outputs, extra_stats_to_sum = results[0], results[1:]
        logging_outputs = list(chain.from_iterable(logging_outputs))
        extra_stats_to_sum = [sum(s) for s in extra_stats_to_sum]
        return logging_outputs, extra_stats_to_sum

    def _fast_stat_sync_sum(
        self,
        logging_outputs: List[Dict[str, Any]],
        *extra_stats_to_sum,
        ignore=False,
    ):
        """
        Sync logging outputs across workers. fast_stat_sync_sum is
        faster than all_gather_list_sync, but is only suitable when
        logging outputs are scalars and can be summed. Note that
        *logging_outputs* cannot contain any nested dicts/lists.
        """
        data = {}
        for i, stat in enumerate(extra_stats_to_sum):
            data["extra_stats_" + str(i)] = stat
        if len(logging_outputs) > 0:
            log_keys = list(logging_outputs[0].keys())
            for k in log_keys:
                if not ignore:
                    v = sum(log[k] for log in logging_outputs if k in log)
                else:
                    v = logging_outputs[0][k]
                    v = torch.zeros_like(v) if torch.is_tensor(v) else 0
                data["logging_outputs_" + k] = v
        else:
            log_keys = None

        data = distributed_utils.all_reduce_dict(
            data, device=self.device, group=self.data_parallel_process_group
        )

        extra_stats_to_sum = [
            data["extra_stats_" + str(i)] for i in range(len(extra_stats_to_sum))
        ]
        if log_keys is not None:
            logging_outputs = [{k: data["logging_outputs_" + k] for k in log_keys}]
        else:
            logging_outputs = []
        return logging_outputs, extra_stats_to_sum

    def _check_grad_norms(self, grad_norm):
        """Check that grad norms are consistent across workers."""
        if self._grad_norm_buf is not None:
            self._grad_norm_buf.zero_()
            self._grad_norm_buf[self.data_parallel_rank] = grad_norm
            distributed_utils.all_reduce(
                self._grad_norm_buf, group=self.data_parallel_process_group
            )

            def is_consistent(tensor):
                max_abs_diff = torch.max(torch.abs(tensor - tensor[0]))
                return (
                    (
                        torch.isfinite(tensor).all()
                        and (max_abs_diff / (tensor[0] + 1e-6) < 1e-6).all()
                    )
                    or (self.cfg.common.amp and not torch.isfinite(tensor).all())
                    # in case of amp non-finite grads are fine
                )

            if not is_consistent(self._grad_norm_buf):
                pretty_detail = "\n".join(
                    "rank {:3d} = {:.8f}".format(r, n)
                    for r, n in enumerate(self._grad_norm_buf.tolist())
                )
                error_detail = "grad_norm across the workers:\n{}\n".format(
                    pretty_detail
                )
                # use FloatingPointError to trigger NanDetector
                raise FloatingPointError(
                    "Fatal error: gradients are inconsistent between workers. "
                    "Try --ddp-backend=legacy_ddp. "
                    "Or are you mixing up different generation of GPUs in training?"
                    + "\n"
                    + "-" * 80
                    + "\n{}\n".format(error_detail)
                    + "-" * 80
                )

    def _reduce_and_log_stats(self, logging_outputs, sample_size, grad_norm=None):
        # standard code
        if grad_norm is not None and (
            not torch.is_tensor(grad_norm) or torch.isfinite(grad_norm)
        ):
            metrics.log_speed("ups", 1.0, priority=100, round=2)
            metrics.log_scalar("gnorm", grad_norm, priority=400, round=3)
            if self.cfg.optimization.clip_norm > 0:
                metrics.log_scalar(
                    "clip",
                    torch.where(
                        grad_norm > self.cfg.optimization.clip_norm,
                        grad_norm.new_tensor(100),
                        grad_norm.new_tensor(0),
                    ),
                    priority=500,
                    round=1,
                )

        with metrics.aggregate() as agg:
            if logging_outputs is not None:
                self.task.reduce_metrics(logging_outputs, self.get_criterion())
                del logging_outputs

            # extra warning for criterions that don't properly log a loss value
            if "loss" not in agg:
                if "loss" not in self._warn_once:
                    self._warn_once.add("loss")
                    logger.warning(
                        "Criterion.reduce_metrics did not log a 'loss' value, "
                        "which may break some functionality"
                    )
                metrics.log_scalar("loss", -1)

            # support legacy interface
            logging_output = agg.get_smoothed_values()
            logging_output["sample_size"] = sample_size
            for key_to_delete in ["ppl", "wps", "wpb", "bsz"]:
                if key_to_delete in logging_output:
                    del logging_output[key_to_delete]

            return logging_output

    def _log_gpu_mem_stats(self):
        # log minimum free memory over the iteration
        cuda_gb_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        cuda_gb_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        cuda_gb_free = self.cuda_env.total_memory_in_GB - cuda_gb_allocated
        metrics.log_scalar(
            "cuda_gb_allocated", cuda_gb_allocated, priority=1500, round=1, weight=0
        )
        metrics.log_scalar(
            "cuda_gb_reserved", cuda_gb_reserved, priority=1500, round=1, weight=0
        )
        metrics.log_scalar(
            "cuda_gb_free", cuda_gb_free, priority=1500, round=1, weight=0
        )


def _catalog_shared_params(module, memo=None, prefix=""):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ("." if prefix else "") + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ("." if prefix else "") + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def _get_module_by_path(module, path):
    path = path.split(".")
    for name in path:
        module = getattr(module, name)
    return module


def _set_module_by_path(module, path, value):
    path = path.split(".")
    for name in path[:-1]:
        module = getattr(module, name)
    setattr(module, path[-1], value)
