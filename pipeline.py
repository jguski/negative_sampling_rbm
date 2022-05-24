import logging
import pathlib
import time
import hashlib
from dataclasses import dataclass, field
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    Tuple,
    Union,
    cast,
)

import torch
from class_resolver.utils import OneOrManyHintOrType, OneOrManyOptionalKwargs
from torch.optim.optimizer import Optimizer

from pykeen.constants import PYKEEN_CHECKPOINTS, USER_DEFINED_CODE
from pykeen.datasets import get_dataset
from pykeen.datasets.base import Dataset
from pykeen.evaluation import Evaluator, MetricResults, evaluator_resolver
from pykeen.losses import Loss, loss_resolver
from pykeen.lr_schedulers import LRScheduler, lr_scheduler_resolver
from pykeen.models import Model, make_model_cls, model_resolver
from pykeen.nn.modules import Interaction
from pykeen.optimizers import optimizer_resolver
from pykeen.regularizers import Regularizer, regularizer_resolver
from pykeen.sampling import NegativeSampler, negative_sampler_resolver
from pykeen.stoppers import EarlyStopper, Stopper, stopper_resolver
from pykeen.trackers import ResultTracker, resolve_result_trackers
from pykeen.training import SLCWATrainingLoop, TrainingLoop, training_loop_resolver
from pykeen.triples import CoreTriplesFactory
from pykeen.typing import Hint, HintType, MappedTriples
from pykeen.utils import (
    random_non_negative_int,
    resolve_device,
    set_random_seed,
)
from pykeen.pipeline import PipelineResult

logger = logging.getLogger(__name__)


def triple_hash(*triples: MappedTriples) -> Mapping[str, str]:
    """Slow triple hash using sha512 and conversion to Python."""
    return dict(sha512=hashlib.sha512(str(sorted(sum((t.tolist() for t in triples), []))).encode("utf8")).hexdigest())

def _get_model_defaults(model: Model) -> Mapping[str, Any]:
    """Get the model's default parameters."""
    return {
        name: param.default
        for name, param in model_resolver.signature(model).parameters.items()
        # skip special parameters
        if name != "self"
        and param.kind not in {param.VAR_POSITIONAL, param.VAR_KEYWORD}
        and param.default != param.empty
    }

def _build_model_helper(
    *,
    model,
    model_kwargs,
    loss,
    loss_kwargs,
    _device,
    _random_seed,
    regularizer,
    regularizer_kwargs,
    training_triples_factory,
) -> Tuple[Model, Mapping[str, Any]]:
    """Collate model-specific parameters and initialize the model from it."""
    if model_kwargs is None:
        model_kwargs = {}
    model_kwargs = dict(model_kwargs)
    model_kwargs.setdefault("random_seed", _random_seed)

    if regularizer is not None:
        # FIXME this should never happen.
        if "regularizer" in model_kwargs:
            _regularizer = model_kwargs.pop("regularizer")
            logger.warning(
                f"Cannot specify regularizer in kwargs ({regularizer}) and model_kwargs ({_regularizer})."
                "removing from model_kwargs",
            )
        model_kwargs["regularizer"] = regularizer_resolver.make(regularizer, regularizer_kwargs)

    if "loss" in model_kwargs:
        _loss = model_kwargs.pop("loss")
        if loss is None:
            loss = _loss
        else:
            logger.warning(
                f"Cannot specify loss in kwargs ({loss}) and model_kwargs ({_loss}). removing from model_kwargs.",
            )
    model_kwargs["loss"] = loss_resolver.make(loss, loss_kwargs)

    if not isinstance(model, Model):
        for param_name, param_default in _get_model_defaults(model).items():
            model_kwargs.setdefault(param_name, param_default)

    return (
        model_resolver.make(
            model,
            triples_factory=training_triples_factory,
            **model_kwargs,
        ),
        model_kwargs,
    )


def pipeline(  # noqa: C901
    *,
    # 1. Dataset
    dataset: Union[None, str, Dataset, Type[Dataset]] = None,
    dataset_kwargs: Optional[Mapping[str, Any]] = None,
    training: Hint[CoreTriplesFactory] = None,
    testing: Hint[CoreTriplesFactory] = None,
    validation: Hint[CoreTriplesFactory] = None,
    evaluation_entity_whitelist: Optional[Collection[str]] = None,
    evaluation_relation_whitelist: Optional[Collection[str]] = None,
    # 2. Model
    model: Union[None, str, Model, Type[Model]] = None,
    model_kwargs: Optional[Mapping[str, Any]] = None,
    interaction: Union[None, str, Interaction, Type[Interaction]] = None,
    interaction_kwargs: Optional[Mapping[str, Any]] = None,
    dimensions: Union[None, int, Mapping[str, int]] = None,
    # 3. Loss
    loss: HintType[Loss] = None,
    loss_kwargs: Optional[Mapping[str, Any]] = None,
    # 4. Regularizer
    regularizer: HintType[Regularizer] = None,
    regularizer_kwargs: Optional[Mapping[str, Any]] = None,
    # 5. Optimizer
    optimizer: HintType[Optimizer] = None,
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    clear_optimizer: bool = True,
    # 5.1 Learning Rate Scheduler
    lr_scheduler: HintType[LRScheduler] = None,
    lr_scheduler_kwargs: Optional[Mapping[str, Any]] = None,
    # 6. Training Loop
    training_loop: HintType[TrainingLoop] = None,
    training_loop_kwargs: Optional[Mapping[str, Any]] = None,
    negative_sampler: HintType[NegativeSampler] = None,
    negative_sampler_kwargs: Optional[Mapping[str, Any]] = None,
    # 7. Training (ronaldo style)
    epochs: Optional[int] = None,
    training_kwargs: Optional[Mapping[str, Any]] = None,
    stopper: HintType[Stopper] = None,
    stopper_kwargs: Optional[Mapping[str, Any]] = None,
    # 8. Evaluation
    evaluator: HintType[Evaluator] = None,
    evaluator_kwargs: Optional[Mapping[str, Any]] = None,
    evaluation_kwargs: Optional[Mapping[str, Any]] = None,
    # 9. Tracking
    result_tracker: OneOrManyHintOrType[ResultTracker] = None,
    result_tracker_kwargs: OneOrManyOptionalKwargs = None,
    # Misc
    metadata: Optional[Dict[str, Any]] = None,
    device: Hint[torch.device] = None,
    random_seed: Optional[int] = None,
    use_testing_data: bool = True,
    evaluation_fallback: bool = False,
    filter_validation_when_testing: bool = True,
    use_tqdm: Optional[bool] = None,
) -> PipelineResult:
    """Train and evaluate a model.
    :param dataset:
        The name of the dataset (a key for the :data:`pykeen.datasets.dataset_resolver`) or the
        :class:`pykeen.datasets.Dataset` instance. Alternatively, the training triples factory (``training``), testing
        triples factory (``testing``), and validation triples factory (``validation``; optional) can be specified.
    :param dataset_kwargs:
        The keyword arguments passed to the dataset upon instantiation
    :param training:
        A triples factory with training instances or path to the training file if a a dataset was not specified
    :param testing:
        A triples factory with training instances or path to the test file if a dataset was not specified
    :param validation:
        A triples factory with validation instances or path to the validation file if a dataset was not specified
    :param evaluation_entity_whitelist:
        Optional restriction of evaluation to triples containing *only* these entities. Useful if the downstream task
        is only interested in certain entities, but the relational patterns with other entities improve the entity
        embedding quality.
    :param evaluation_relation_whitelist:
        Optional restriction of evaluation to triples containing *only* these relations. Useful if the downstream task
        is only interested in certain relation, but the relational patterns with other relations improve the entity
        embedding quality.
    :param model:
        The name of the model, subclass of :class:`pykeen.models.Model`, or an instance of
        :class:`pykeen.models.Model`. Can be given as None if the ``interaction`` keyword is used.
    :param model_kwargs:
        Keyword arguments to pass to the model class on instantiation
    :param interaction: The name of the interaction class, a subclass of :class:`pykeen.nn.modules.Interaction`,
        or an instance of :class:`pykeen.nn.modules.Interaction`. Can not be given when there is also a model.
    :param interaction_kwargs:
        Keyword arguments to pass during instantiation of the interaction class. Only use with ``interaction``.
    :param dimensions:
        Dimensions to assign to the embeddings of the interaction. Only use with ``interaction``.
    :param loss:
        The name of the loss or the loss class.
    :param loss_kwargs:
        Keyword arguments to pass to the loss on instantiation
    :param regularizer:
        The name of the regularizer or the regularizer class.
    :param regularizer_kwargs:
        Keyword arguments to pass to the regularizer on instantiation
    :param optimizer:
        The name of the optimizer or the optimizer class. Defaults to :class:`torch.optim.Adagrad`.
    :param optimizer_kwargs:
        Keyword arguments to pass to the optimizer on instantiation
    :param clear_optimizer:
        Whether to delete the optimizer instance after training. As the optimizer might have additional memory
        consumption due to e.g. moments in Adam, this is the default option. If you want to continue training, you
        should set it to False, as the optimizer's internal parameter will get lost otherwise.
    :param lr_scheduler:
        The name of the lr_scheduler or the lr_scheduler class.
        Defaults to :class:`torch.optim.lr_scheduler.ExponentialLR`.
    :param lr_scheduler_kwargs:
        Keyword arguments to pass to the lr_scheduler on instantiation
    :param training_loop:
        The name of the training loop's training approach (``'slcwa'`` or ``'lcwa'``) or the training loop class.
        Defaults to :class:`pykeen.training.SLCWATrainingLoop`.
    :param training_loop_kwargs:
        Keyword arguments to pass to the training loop on instantiation
    :param negative_sampler:
        The name of the negative sampler (``'basic'`` or ``'bernoulli'``) or the negative sampler class.
        Only allowed when training with sLCWA.
        Defaults to :class:`pykeen.sampling.BasicNegativeSampler`.
    :param negative_sampler_kwargs:
        Keyword arguments to pass to the negative sampler class on instantiation
    :param epochs:
        A shortcut for setting the ``num_epochs`` key in the ``training_kwargs`` dict.
    :param training_kwargs:
        Keyword arguments to pass to the training loop's train function on call
    :param stopper:
        What kind of stopping to use. Default to no stopping, can be set to 'early'.
    :param stopper_kwargs:
        Keyword arguments to pass to the stopper upon instantiation.
    :param evaluator:
        The name of the evaluator or an evaluator class. Defaults to :class:`pykeen.evaluation.RankBasedEvaluator`.
    :param evaluator_kwargs:
        Keyword arguments to pass to the evaluator on instantiation
    :param evaluation_kwargs:
        Keyword arguments to pass to the evaluator's evaluate function on call
    :param result_tracker: Either none (will result in a Python result tracker),
        a single tracker (as either a class, instance, or string for class name), or a list
        of trackers (as either a class, instance, or string for class name
    :param result_tracker_kwargs: Either none (will use all defaults), a single dictionary
        (will be used for all trackers), or a list of dictionaries with the same length
        as the result trackers
    :param metadata:
        A JSON dictionary to store with the experiment
    :param use_testing_data:
        If true, use the testing triples. Otherwise, use the validation triples. Defaults to true - use testing triples.
    :param device: The device or device name to run on. If none is given, the device will be looked up with
        :func:`pykeen.utils.resolve_device`.
    :param random_seed: The random seed to use. If none is specified, one will be assigned before any code
        is run for reproducibility purposes. In the returned :class:`PipelineResult` instance, it can be accessed
        through :data:`PipelineResult.random_seed`.
    :param evaluation_fallback:
        If true, in cases where the evaluation failed using the GPU it will fall back to using a smaller batch size or
        in the last instance evaluate on the CPU, if even the smallest possible batch size is too big for the GPU.
    :param filter_validation_when_testing:
        If true, during the evaluating of the test dataset, validation triples are added to the set of known positive
        triples, which are filtered out when performing filtered evaluation following the approach described by
        [bordes2013]_. This should be explicitly set to false only in the scenario that you are training a single
        model using the pipeline and evaluating with the testing set, but never using the validation set for
        optimization at all. This is a very atypical scenario, so it is left as true by default to promote
        comparability to previous publications.
    :param use_tqdm:
        Globally set the usage of tqdm progress bars. Typically more useful to set to false, since the training
        loop and evaluation have it turned on by default.
    :returns: A pipeline result package.
    :raises ValueError:
        If a negative sampler is specified with LCWA
    :raises TypeError:
        If an invalid argument type is given for ``evaluation_kwargs["additional_filter_triples"]``
    """
    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs = dict(training_kwargs)

    # To allow resuming training from a checkpoint when using a pipeline, the pipeline needs to obtain the
    # used random_seed to ensure reproducible results
    checkpoint_name = training_kwargs.get("checkpoint_name")
    if checkpoint_name is not None:
        checkpoint_directory = pathlib.Path(training_kwargs.get("checkpoint_directory", PYKEEN_CHECKPOINTS))
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_directory / checkpoint_name
        if checkpoint_path.is_file():
            checkpoint_dict = torch.load(checkpoint_path)
            _random_seed = checkpoint_dict["random_seed"]
            logger.info("loaded random seed %s from checkpoint.", _random_seed)
            # We have to set clear optimizer to False since training should be continued
            clear_optimizer = False
        else:
            logger.info(f"=> no training loop checkpoint file found at '{checkpoint_path}'. Creating a new file.")
            if random_seed is None:
                _random_seed = random_non_negative_int()
                logger.warning(f"No random seed is specified. Setting to {_random_seed}.")
            else:
                _random_seed = random_seed
    elif random_seed is None:
        _random_seed = random_non_negative_int()
        logger.warning(f"No random seed is specified. Setting to {_random_seed}.")
    else:
        _random_seed = random_seed  # random seed given successfully
    set_random_seed(_random_seed)

    _result_tracker = resolve_result_trackers(result_tracker, result_tracker_kwargs)

    if not metadata:
        metadata = {}
    title = metadata.get("title")

    # Start tracking
    _result_tracker.start_run(run_name=title)

    _device: torch.device = resolve_device(device)
    logger.info(f"Using device: {device}")

    dataset_instance: Dataset = get_dataset(
        dataset=dataset,
        dataset_kwargs=dataset_kwargs,
        training=training,
        testing=testing,
        validation=validation,
    )
    if dataset is not None:
        _result_tracker.log_params(
            dict(
                dataset=dataset_instance.get_normalized_name(),
                dataset_kwargs=dataset_kwargs,
            )
        )
    else:  # means that dataset was defined by triples factories
        _result_tracker.log_params(
            dict(
                dataset=USER_DEFINED_CODE,
                training=training if isinstance(training, str) else USER_DEFINED_CODE,
                testing=testing if isinstance(training, str) else USER_DEFINED_CODE,
                validation=validation if isinstance(training, str) else USER_DEFINED_CODE,
            )
        )

    training, testing, validation = dataset_instance.training, dataset_instance.testing, dataset_instance.validation
    # evaluation restriction to a subset of entities/relations
    if any(f is not None for f in (evaluation_entity_whitelist, evaluation_relation_whitelist)):
        testing = testing.new_with_restriction(
            entities=evaluation_entity_whitelist,
            relations=evaluation_relation_whitelist,
        )
        if validation is not None:
            validation = validation.new_with_restriction(
                entities=evaluation_entity_whitelist,
                relations=evaluation_relation_whitelist,
            )

    model_instance: Model
    if model is not None and interaction is not None:
        raise ValueError("can not pass both a model and interaction")
    elif model is None and interaction is None:
        raise ValueError("must pass one of model or interaction")
    elif interaction is not None:
        if dimensions is None:
            raise ValueError("missing dimensions")
        model = make_model_cls(
            interaction=interaction,
            dimensions=dimensions,
            interaction_kwargs=interaction_kwargs,
        )

    if isinstance(model, Model):
        model_instance = cast(Model, model)
        # TODO should training be reset?
        # TODO should kwargs for loss and regularizer be checked and raised for?
    else:
        model_instance, model_kwargs = _build_model_helper(
            model=model,
            model_kwargs=model_kwargs,
            loss=loss,
            loss_kwargs=loss_kwargs,
            regularizer=regularizer,
            regularizer_kwargs=regularizer_kwargs,
            _device=_device,
            _random_seed=_random_seed,
            training_triples_factory=training,
        )

    model_instance = model_instance.to(_device)

    # Log model parameters
    _result_tracker.log_params(
        params=dict(
            model=model_instance.__class__.__name__,
            model_kwargs=model_kwargs,
        ),
    )

    # Log loss parameters
    _result_tracker.log_params(
        params=dict(
            # the loss was already logged as part of the model kwargs
            # loss=loss_resolver.normalize_inst(model_instance.loss),
            loss_kwargs=loss_kwargs
        ),
    )

    # Log regularizer parameters
    _result_tracker.log_params(
        params=dict(regularizer_kwargs=regularizer_kwargs),
    )

    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_instance = optimizer_resolver.make(
        optimizer,
        optimizer_kwargs,
        params=model_instance.get_grad_params(),
    )
    for key, value in optimizer_instance.defaults.items():
        optimizer_kwargs.setdefault(key, value)
    _result_tracker.log_params(
        params=dict(
            optimizer=optimizer_instance.__class__.__name__,
            optimizer_kwargs=optimizer_kwargs,
        ),
    )

    lr_scheduler_instance: Optional[LRScheduler]
    if lr_scheduler is None:
        lr_scheduler_instance = None
    else:
        lr_scheduler_instance = lr_scheduler_resolver.make(
            lr_scheduler,
            lr_scheduler_kwargs,
            optimizer=optimizer_instance,
        )
        _result_tracker.log_params(
            params=dict(
                lr_scheduler=lr_scheduler_instance.__class__.__name__,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
            ),
        )

    training_loop_cls = training_loop_resolver.lookup(training_loop)
    if training_loop_kwargs is None:
        training_loop_kwargs = {}

    if negative_sampler is None:
        negative_sampler_cls = None
    elif not issubclass(training_loop_cls, SLCWATrainingLoop):
        raise ValueError("Can not specify negative sampler with LCWA")
    else:
        negative_sampler_cls = negative_sampler_resolver.lookup(negative_sampler)
        training_loop_kwargs = dict(training_loop_kwargs)
        if "ESNS" in str((negative_sampler_cls).__name__):
            negative_sampler_kwargs["model"] = model_instance
        training_loop_kwargs.update(
            negative_sampler=negative_sampler_cls,
            negative_sampler_kwargs=negative_sampler_kwargs,
        )
        
        _result_tracker.log_params(
            params=dict(
                negative_sampler=negative_sampler_cls.__name__,
                negative_sampler_kwargs=negative_sampler_kwargs,
            ),
        )
    training_loop_instance = training_loop_cls(
        model=model_instance,
        triples_factory=training,
        optimizer=optimizer_instance,
        lr_scheduler=lr_scheduler_instance,
        result_tracker=_result_tracker,
        **training_loop_kwargs,
    )
    _result_tracker.log_params(
        params=dict(
            training_loop=training_loop_instance.__class__.__name__,
            training_loop_kwargs=training_loop_kwargs,
        ),
    )

    if evaluator_kwargs is None:
        evaluator_kwargs = {}
    evaluator_kwargs = dict(evaluator_kwargs)
    evaluator_instance: Evaluator = evaluator_resolver.make(evaluator, evaluator_kwargs)
    _result_tracker.log_params(
        params=dict(
            evaluator=evaluator_instance.__class__.__name__,
            evaluator_kwargs=evaluator_kwargs,
        ),
    )

    if evaluation_kwargs is None:
        evaluation_kwargs = {}
    evaluation_kwargs = dict(evaluation_kwargs)

    # Stopping
    if "stopper" in training_kwargs and stopper is not None:
        raise ValueError("Specified stopper in training_kwargs and as stopper")
    if "stopper" in training_kwargs:
        stopper = training_kwargs.pop("stopper")
    if stopper_kwargs is None:
        stopper_kwargs = {}
    stopper_kwargs = dict(stopper_kwargs)

    # Load the evaluation batch size for the stopper, if it has been set
    _evaluation_batch_size = evaluation_kwargs.get("batch_size")
    if _evaluation_batch_size is not None:
        stopper_kwargs.setdefault("evaluation_batch_size", _evaluation_batch_size)

    stopper_instance: Stopper = stopper_resolver.make(
        stopper,
        model=model_instance,
        evaluator=evaluator_instance,
        training_triples_factory=training,
        evaluation_triples_factory=validation,
        result_tracker=_result_tracker,
        **stopper_kwargs,
    )

    if epochs is not None:
        training_kwargs["num_epochs"] = epochs
    if use_tqdm is not None:
        training_kwargs["use_tqdm"] = use_tqdm
    training_kwargs.setdefault("num_epochs", 5)
    training_kwargs.setdefault("batch_size", 256)
    _result_tracker.log_params(params=training_kwargs)

    # Add logging for debugging
    configuration = _result_tracker.get_configuration()
    logging.debug("Run Pipeline based on following config:")
    for key, value in configuration.items():
        logging.debug(f"{key}: {value}")

    # Train like Cristiano Ronaldo
    training_start_time = time.time()
    losses = training_loop_instance.train(
        triples_factory=training,
        stopper=stopper_instance,
        clear_optimizer=clear_optimizer,
        **training_kwargs,
    )
    assert losses is not None  # losses is only none if it's doing search mode
    training_end_time = time.time() - training_start_time
    step = training_kwargs.get("num_epochs")
    _result_tracker.log_metrics(metrics=dict(total_training=training_end_time), step=step, prefix="times")

    if use_testing_data:
        mapped_triples = testing.mapped_triples
    elif validation is None:
        raise ValueError("no validation triples available")
    else:
        mapped_triples = validation.mapped_triples

    # Build up a list of triples if we want to be in the filtered setting
    additional_filter_triples_names = dict()
    if evaluator_instance.filtered:
        additional_filter_triples: List[MappedTriples] = [
            training.mapped_triples,
        ]
        additional_filter_triples_names["training"] = triple_hash(training.mapped_triples)

        # If the user gave custom "additional_filter_triples"
        popped_additional_filter_triples = evaluation_kwargs.pop("additional_filter_triples", [])
        if popped_additional_filter_triples:
            additional_filter_triples_names["custom"] = triple_hash(*popped_additional_filter_triples)
        if isinstance(popped_additional_filter_triples, (list, tuple)):
            additional_filter_triples.extend(popped_additional_filter_triples)
        elif torch.is_tensor(popped_additional_filter_triples):  # a single MappedTriple
            additional_filter_triples.append(popped_additional_filter_triples)
        else:
            raise TypeError(
                f'Invalid type for `evaluation_kwargs["additional_filter_triples"]`:'
                f" {type(popped_additional_filter_triples)}",
            )

        # Determine whether the validation triples should also be filtered while performing test evaluation
        if use_testing_data and filter_validation_when_testing and validation is not None:
            if isinstance(stopper, EarlyStopper):
                logging.info(
                    "When evaluating the test dataset after running the pipeline with early stopping, the validation"
                    " triples are added to the set of known positive triples which are filtered out when performing"
                    " filtered evaluation following the approach described by (Bordes et al., 2013).",
                )
            else:
                logging.info(
                    "When evaluating the test dataset, validation triples are added to the set of known positive"
                    " triples which are filtered out when performing filtered evaluation following the approach"
                    " described by (Bordes et al., 2013).",
                )
            additional_filter_triples.append(validation.mapped_triples)
            additional_filter_triples_names["validation"] = triple_hash(validation.mapped_triples)

        # TODO consider implications of duplicates
        evaluation_kwargs["additional_filter_triples"] = additional_filter_triples

    # Evaluate
    # Reuse optimal evaluation parameters from training if available, only if the validation triples are used again
    if evaluator_instance.batch_size is not None or evaluator_instance.slice_size is not None and not use_testing_data:
        evaluation_kwargs["batch_size"] = evaluator_instance.batch_size
        evaluation_kwargs["slice_size"] = evaluator_instance.slice_size
    if use_tqdm is not None:
        evaluation_kwargs["use_tqdm"] = use_tqdm
    # Add logging about evaluator for debugging
    _result_tracker.log_params(
        params=dict(
            evaluation_kwargs={
                k: (additional_filter_triples_names if k == "additional_filter_triples" else v)
                for k, v in evaluation_kwargs.items()
            }
        )
    )
    evaluate_start_time = time.time()
    metric_results: MetricResults = _safe_evaluate(
        model=model_instance,
        mapped_triples=mapped_triples,
        evaluator=evaluator_instance,
        evaluation_kwargs=evaluation_kwargs,
        evaluation_fallback=evaluation_fallback,
    )
    evaluate_end_time = time.time() - evaluate_start_time
    _result_tracker.log_metrics(metrics=dict(final_evaluation=evaluate_end_time), step=step, prefix="times")
    _result_tracker.log_metrics(
        metrics=metric_results.to_dict(),
        step=step,
    )
    _result_tracker.end_run()

    return PipelineResult(
        random_seed=_random_seed,
        model=model_instance,
        training=training,
        training_loop=training_loop_instance,
        losses=losses,
        stopper=stopper_instance,
        configuration=configuration,
        metric_results=metric_results,
        metadata=metadata,
        train_seconds=training_end_time,
        evaluate_seconds=evaluate_end_time,
    )


def _safe_evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluator: Evaluator,
    evaluation_kwargs: Dict[str, Any],
    evaluation_fallback: bool = False,
) -> MetricResults:
    """Evaluate with a potentially safe fallback to CPU.
    :param model: The model
    :param mapped_triples: Mapped triples from the evaluation set (test or valid)
    :param evaluator: An evaluator
    :param evaluation_kwargs: Kwargs for the evaluator (might get modified in place)
    :param evaluation_fallback:
        If true, in cases where the evaluation failed using the GPU it will fall back to using a smaller batch size or
        in the last instance evaluate on the CPU, if even the smallest possible batch size is too big for the GPU.
    :return: A metric result
    :raises MemoryError:
        If it is not possible to evaluate the model on the hardware at hand with the given parameters.
    :raises RuntimeError:
        If CUDA ran into OOM issues trying to evaluate the model on the hardware at hand with the given parameters.
    """
    while True:
        try:
            metric_results: MetricResults = evaluator.evaluate(
                model=model,
                mapped_triples=mapped_triples,
                **evaluation_kwargs,
            )
        except (MemoryError, RuntimeError) as e:
            # If the evaluation still fail using the CPU, the error is raised
            if model.device.type != "cuda" or not evaluation_fallback:
                raise e

            # When the evaluation failed due to OOM on the GPU due to a batch size set too high, the evaluation is
            # restarted with PyKEEN's automatic memory optimization
            elif "batch_size" in evaluation_kwargs:
                logging.warning(
                    "You tried to evaluate the current model on %s with batch_size=%d which was too big for %s.",
                    model.device,
                    evaluation_kwargs["batch_size"],
                    model.device,
                )
                logging.warning("Will activate the built-in PyKEEN memory optimization to find a suitable batch size.")
                del evaluation_kwargs["batch_size"]

            # When the evaluation failed due to OOM on the GPU even with automatic memory optimization, the evaluation
            # is restarted using the cpu
            else:  # 'batch_size' not in evaluation_kwargs
                logging.warning(
                    "Tried to evaluate the current model on %s, but the model and the dataset are too big for the "
                    "%s memory currently available.",
                    model.device,
                    model.device,
                )
                logging.warning(
                    "Will revert to using the CPU for evaluation, which will increase the evaluation time "
                    "significantly.",
                )
                model.to_cpu_()
        else:
            break  # evaluation was successful, don't continue the ``while True`` loop

    return metric_results