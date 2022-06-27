from pykeen.training import SLCWATrainingLoop
from pykeen.training.training_loop import SubBatchingNotSupportedError, NoTrainingBatchError, _get_optimizer_kwargs, _get_lr_scheduler_kwargs

import gc
import os
import logging
import pathlib
import time
from tempfile import NamedTemporaryFile
from typing import Any, List, Mapping, Optional, Union

from torch.optim.optimizer import Optimizer
from tqdm.autonotebook import tqdm, trange

from pykeen.training.callbacks import (
    GradientAbsClippingTrainingCallback,
    GradientNormClippingTrainingCallback,
    MultiTrainingCallback,
    StopperTrainingCallback,
    TrackerTrainingCallback,
    TrainingCallbackHint,
    TrainingCallbackKwargsHint,
)
from pykeen.models import RGCN
from pykeen.stoppers import Stopper
from pykeen.triples import CoreTriplesFactory
from pykeen.utils import (
    format_relative_comparison,
    get_batchnorm_modules,
    get_preferred_device,
)

logger = logging.getLogger(__name__)


class SLCWATrainingLoopModified(SLCWATrainingLoop):
    def _train(  # noqa: C901
        self,
        triples_factory: CoreTriplesFactory,
        num_epochs: int = 1,
        batch_size: Optional[int] = None,
        slice_size: Optional[int] = None,
        label_smoothing: float = 0.0,
        sampler: Optional[str] = None,
        continue_training: bool = False,
        only_size_probing: bool = False,
        use_tqdm: bool = True,
        use_tqdm_batch: bool = True,
        tqdm_kwargs: Optional[Mapping[str, Any]] = None,
        stopper: Optional[Stopper] = None,
        sub_batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        save_checkpoints: bool = False,
        checkpoint_path: Union[None, str, pathlib.Path] = None,
        checkpoint_frequency: Optional[int] = None,
        checkpoint_on_failure_file_path: Union[None, str, pathlib.Path] = None,
        best_epoch_model_file_path: Optional[pathlib.Path] = None,
        last_best_epoch: Optional[int] = None,
        drop_last: Optional[bool] = None,
        callbacks: TrainingCallbackHint = None,
        callback_kwargs: TrainingCallbackKwargsHint = None,
        gradient_clipping_max_norm: Optional[float] = None,
        gradient_clipping_norm_type: Optional[float] = None,
        gradient_clipping_max_abs_value: Optional[float] = None,
        pin_memory: bool = True,
    ) -> Optional[List[float]]:
        """Train the KGE model, see docstring for :func:`TrainingLoop.train`."""
        if self.optimizer is None:
            raise ValueError("optimizer must be set before running _train()")
        # When using early stopping models have to be saved separately at the best epoch, since the training loop will
        # due to the patience continue to train after the best epoch and thus alter the model
        if (
            stopper is not None
            and not only_size_probing
            and last_best_epoch is None
            and best_epoch_model_file_path is None
        ):
            # Create a path
            best_epoch_model_file_path = pathlib.Path(NamedTemporaryFile().name)
        best_epoch_model_checkpoint_file_path: Optional[pathlib.Path] = None

        if isinstance(self.model, RGCN) and sampler != "schlichtkrull":
            logger.warning(
                'Using RGCN without graph-based sampling! Please select sampler="schlichtkrull" instead of %s.',
                sampler,
            )

        #################### Initialize ESNS sampler #######################
        # It would be nicer to just put this into a callback. However, I have not found a way to access triples_factory then.
        ns_kwargs = dict(self.negative_sampler_kwargs)
        if (ns_kwargs["n_triples_for_ns_qual_analysis"] > 0) and (ns_kwargs["ns_qual_analysis_every"] > 0): 
            ns_kwargs["model"] = self.model
            ns_kwargs["logging_level"] = "ERROR"
            self.negative_sampling_instance = self.negative_sampler(**ns_kwargs, mapped_triples=triples_factory.mapped_triples)
        ####################################################################


        # Prepare all of the callbacks
        callback = MultiTrainingCallback(callbacks=callbacks, callback_kwargs=callback_kwargs)
        # Register a callback for the result tracker, if given
        if self.result_tracker is not None:
            callback.register_callback(TrackerTrainingCallback())
        # Register a callback for the early stopper, if given
        # TODO should mode be passed here?
        if stopper is not None:
            callback.register_callback(
                StopperTrainingCallback(
                    stopper,
                    triples_factory=triples_factory,
                    last_best_epoch=last_best_epoch,
                    best_epoch_model_file_path=best_epoch_model_file_path,
                )
            )
        if gradient_clipping_max_norm is not None:
            callback.register_callback(
                GradientNormClippingTrainingCallback(
                    max_norm=gradient_clipping_max_norm,
                    norm_type=gradient_clipping_norm_type,
                )
            )
        if gradient_clipping_max_abs_value is not None:
            callback.register_callback(GradientAbsClippingTrainingCallback(clip_value=gradient_clipping_max_abs_value))

        callback.register_training_loop(self)

        # Take the biggest possible training batch_size, if batch_size not set
        batch_size_sufficient = False
        if batch_size is None:
            if self.automatic_memory_optimization:
                # Using automatic memory optimization on CPU may result in undocumented crashes due to OS' OOM killer.
                if self.model.device.type == "cpu":
                    batch_size = 256
                    batch_size_sufficient = True
                    logger.info(
                        "Currently automatic memory optimization only supports GPUs, but you're using a CPU. "
                        "Therefore, the batch_size will be set to the default value '{batch_size}'",
                    )
                else:
                    batch_size, batch_size_sufficient = self.batch_size_search(triples_factory=triples_factory)
            else:
                batch_size = 256
                logger.info(f"No batch_size provided. Setting batch_size to '{batch_size}'.")

        # This will find necessary parameters to optimize the use of the hardware at hand
        if (
            not only_size_probing
            and self.automatic_memory_optimization
            and not batch_size_sufficient
            and not continue_training
        ):
            # return the relevant parameters slice_size and batch_size
            sub_batch_size, slice_size = self.sub_batch_and_slice(
                batch_size=batch_size, sampler=sampler, triples_factory=triples_factory
            )

        if sub_batch_size is None or sub_batch_size == batch_size:  # by default do not split batches in sub-batches
            sub_batch_size = batch_size
        elif get_batchnorm_modules(self.model):  # if there are any, this is truthy
            raise SubBatchingNotSupportedError(self.model)

        model_contains_batch_norm = bool(get_batchnorm_modules(self.model))
        if batch_size == 1 and model_contains_batch_norm:
            raise ValueError("Cannot train a model with batch_size=1 containing BatchNorm layers.")

        if drop_last is None:
            drop_last = model_contains_batch_norm

        # Force weight initialization if training continuation is not explicitly requested.
        if not continue_training:
            # Reset the weights
            self.model.reset_parameters_()
            # afterwards, some parameters may be on the wrong device
            self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

            # Create new optimizer
            optimizer_kwargs = _get_optimizer_kwargs(self.optimizer)
            self.optimizer = self.optimizer.__class__(
                params=self.model.get_grad_params(),
                **optimizer_kwargs,
            )

            if self.lr_scheduler is not None:
                # Create a new lr scheduler and add the optimizer
                lr_scheduler_kwargs = _get_lr_scheduler_kwargs(self.lr_scheduler)
                self.lr_scheduler = self.lr_scheduler.__class__(self.optimizer, **lr_scheduler_kwargs)
        elif not self.optimizer.state:
            raise ValueError("Cannot continue_training without being trained once.")

        # Ensure the model is on the correct device
        self.model.to(get_preferred_device(self.model, allow_ambiguity=True))

        if num_workers is None:
            num_workers = 0

        _use_outer_tqdm = not only_size_probing and use_tqdm
        _use_inner_tqdm = _use_outer_tqdm and use_tqdm_batch

        # When size probing, we don't want progress bars
        if _use_outer_tqdm:
            # Create progress bar
            _tqdm_kwargs = dict(desc=f"Training epochs on {self.device}", unit="epoch")
            if tqdm_kwargs is not None:
                _tqdm_kwargs.update(tqdm_kwargs)
            epochs = trange(self._epoch + 1, 1 + num_epochs, **_tqdm_kwargs, initial=self._epoch, total=num_epochs)
        elif only_size_probing:
            epochs = range(1, 1 + num_epochs)
        else:
            epochs = range(self._epoch + 1, 1 + num_epochs)

        logger.debug(f"using stopper: {stopper}")

        train_data_loader = self._create_training_data_loader(
            triples_factory,
            batch_size,
            drop_last,
            num_workers,
            pin_memory,
            sampler=sampler,
        )
        if len(train_data_loader) == 0:
            raise NoTrainingBatchError()
        if drop_last and not only_size_probing:
            logger.info(
                "Dropping last (incomplete) batch each epoch (%s batches).",
                format_relative_comparison(part=1, total=len(train_data_loader)),
            )

        # Save the time to track when the saved point was available
        last_checkpoint = time.time()

        # Training Loop
        for epoch in epochs:
            # When training with an early stopper the memory pressure changes, which may allow for errors each epoch
            try:
                # Enforce training mode
                self.model.train()

                # Accumulate loss over epoch
                current_epoch_loss = 0.0

                # Batching
                # Only create a progress bar when not in size probing mode
                if _use_inner_tqdm:
                    batches = tqdm(
                        train_data_loader,
                        desc=f"Training batches on {self.device}",
                        leave=False,
                        unit="batch",
                    )
                else:
                    batches = train_data_loader

                # Flag to check when to quit the size probing
                evaluated_once = False

                num_training_instances = 0
                for batch in batches:
                    # Recall that torch *accumulates* gradients. Before passing in a
                    # new instance, you need to zero out the gradients from the old instance
                    self.optimizer.zero_grad()

                    # Get batch size of current batch (last batch may be incomplete)
                    current_batch_size = self._get_batch_size(batch)
                    _sub_batch_size = sub_batch_size or current_batch_size

                    # accumulate gradients for whole batch
                    for start in range(0, current_batch_size, _sub_batch_size):
                        stop = min(start + _sub_batch_size, current_batch_size)

                        # forward pass call
                        batch_loss = self._forward_pass(
                            batch,
                            start,
                            stop,
                            current_batch_size,
                            label_smoothing,
                            slice_size,
                        )
                        current_epoch_loss += batch_loss
                        num_training_instances += stop - start
                        callback.on_batch(epoch=epoch, batch=batch, batch_loss=batch_loss)

                    # when called by batch_size_search(), the parameter update should not be applied.
                    if not only_size_probing:
                        callback.pre_step()

                        # update parameters according to optimizer
                        self.optimizer.step()

                    # After changing applying the gradients to the embeddings, the model is notified that the forward
                    # constraints are no longer applied
                    self.model.post_parameter_update()

                    # For testing purposes we're only interested in processing one batch
                    if only_size_probing and evaluated_once:
                        break

                    callback.post_batch(epoch=epoch, batch=batch)

                    evaluated_once = True

                del batch
                del batches
                gc.collect()
                self.optimizer.zero_grad()
                self._free_graph_and_cache()

                # When size probing we don't need the losses
                if only_size_probing:
                    return None

                # Update learning rate scheduler
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(epoch=epoch)

                # Track epoch loss
                if self.model.loss.reduction == "mean":
                    epoch_loss = current_epoch_loss / num_training_instances
                else:
                    epoch_loss = current_epoch_loss / len(train_data_loader)
                self.losses_per_epochs.append(epoch_loss)

                # Print loss information to console
                if _use_outer_tqdm:
                    epochs.set_postfix(
                        {
                            "loss": self.losses_per_epochs[-1],
                            "prev_loss": self.losses_per_epochs[-2] if epoch > 1 else float("nan"),
                        }
                    )

                # Save the last successful finished epoch
                self._epoch = epoch

                ############### QUALITY ANALYSIS #########################
                if (ns_kwargs["n_triples_for_ns_qual_analysis"] > 0) and (ns_kwargs["ns_qual_analysis_every"] > 0): 
                    self.negative_sampling_instance.quality_analysis(epoch)
                ##########################################################

            # When the training loop failed, a fallback checkpoint is created to resume training.
            except (MemoryError, RuntimeError) as e:
                # During automatic memory optimization only the error message is of interest
                if only_size_probing:
                    raise e

                logger.warning(f"The training loop just failed during epoch {epoch} due to error {str(e)}.")
                if checkpoint_on_failure_file_path:
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_on_failure_file_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )
                    logger.warning(
                        "However, don't worry we got you covered. PyKEEN just saved a checkpoint when this "
                        f"happened at '{checkpoint_on_failure_file_path}'. To resume training from the checkpoint "
                        f"file just restart your code and pass this file path to the training loop or pipeline you "
                        f"used as 'checkpoint_file' argument.",
                    )
                # Delete temporary best epoch model
                if best_epoch_model_file_path is not None and best_epoch_model_file_path.is_file():
                    os.remove(best_epoch_model_file_path)
                raise e

            # Includes a call to result_tracker.log_metrics
            callback.post_epoch(epoch=epoch, epoch_loss=epoch_loss)

            # If a checkpoint file is given, we check whether it is time to save a checkpoint
            if save_checkpoints and checkpoint_path is not None:
                minutes_since_last_checkpoint = (time.time() - last_checkpoint) // 60
                # MyPy overrides are because you should
                if (
                    minutes_since_last_checkpoint >= checkpoint_frequency  # type: ignore
                    or self._should_stop
                    or epoch == num_epochs
                ):
                    # When there wasn't a best epoch the checkpoint path should be None
                    if last_best_epoch is not None and best_epoch_model_file_path is not None:
                        best_epoch_model_checkpoint_file_path = best_epoch_model_file_path
                    self._save_state(
                        path=checkpoint_path,
                        stopper=stopper,
                        best_epoch_model_checkpoint_file_path=best_epoch_model_checkpoint_file_path,
                        triples_factory=triples_factory,
                    )  # type: ignore
                    last_checkpoint = time.time()

            if self._should_stop:
                if last_best_epoch is not None and best_epoch_model_file_path is not None:
                    self._load_state(path=best_epoch_model_file_path)
                    # Delete temporary best epoch model
                    if pathlib.Path.is_file(best_epoch_model_file_path):
                        os.remove(best_epoch_model_file_path)
                return self.losses_per_epochs


        callback.post_train(losses=self.losses_per_epochs)

        # If the stopper didn't stop the training loop but derived a best epoch, the model has to be reconstructed
        # at that state
        if stopper is not None and last_best_epoch is not None and best_epoch_model_file_path is not None:
            self._load_state(path=best_epoch_model_file_path)
            # Delete temporary best epoch model
            if pathlib.Path.is_file(best_epoch_model_file_path):
                os.remove(best_epoch_model_file_path)

        return self.losses_per_epochs

    


