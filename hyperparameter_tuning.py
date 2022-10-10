import os
from modified_pykeen.hpo_modified import hpo_pipeline
from modified_pykeen.slcwa_modified import SLCWATrainingLoop, SLCWATrainingLoopModified
from negative_samplers import ESNSStandard, ESNSRelaxed, ESNSRidle
from losses.custom_losses import ShiftLogLoss
from pykeen.datasets import get_dataset

experiments = [
    {"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "basic"},
    {"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "bernoulli"},
    {"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_standard", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "YAGO3-10", "negative_sampler": "basic"},
    {"model": "TransE", "dataset": "YAGO3-10", "negative_sampler": "bernoulli"},
    {"model": "TransE", "dataset": "YAGO3-10", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "YAGO3-10", "negative_sampler": "esns_standard", "similarity_metric": "absolute"},
]


neg_samplers_dict = {"basic": "basic", "bernoulli": "bernoulli", "esns_relaxed": ESNSRelaxed, "esns_ridle": ESNSRidle, "esns_standard": ESNSStandard}

index_column_size=100
index_path_base = "EII"
sampling_size = 100
q_set_size = 50
batch_size=1024
lr = 0.001
regularization_weight = 0.0001
# no quality analysis of selected triples for hpo
n_triples_for_ns_qual_analysis=0
ns_qual_analysis_every=0

results_path_base = "Output/hpo"
num_epochs = 1000
n_trials = 25
device = "gpu"

embedding_dim_range = dict(type='categorical', choices=[50,100,200])
shift_range = dict(type=int, low=0, high=25, step=1)

if not os.path.isdir(results_path_base):
    os.makedirs(results_path_base)


for exp in experiments:

    exp_name = "-".join(list(exp.values()))
    exp["exp_name"] = exp_name
    print("Starting hpo for {}".format(exp_name))

    if "esns" in exp["negative_sampler"]:
        negative_sampler_kwargs=dict(
            index_column_size=index_column_size,
            sampling_size=sampling_size,
            q_set_size=q_set_size,
            similarity_metric=exp["similarity_metric"],
            n_triples_for_ns_qual_analysis=n_triples_for_ns_qual_analysis,
            ns_qual_analysis_every=ns_qual_analysis_every,
            logging_level="INFO",
            num_negs_per_pos=1,
            dataset=exp["dataset"]
        )
        if "rbm_layer" in exp.keys():
            negative_sampler_kwargs["rbm_layer"] = exp["rbm_layer"]
        training_loop=SLCWATrainingLoopModified
    else:
        negative_sampler_kwargs=dict(num_negs_per_pos=1)
        training_loop = SLCWATrainingLoop

    # load dataset beforehand so that validation set can be used for evaluation instead of testing set
    dataset_instance = get_dataset(dataset=exp["dataset"])

    results = hpo_pipeline(
        n_trials=n_trials,
        training=dataset_instance.training,
        validation=dataset_instance.validation,
        testing=dataset_instance.validation,
        model=exp["model"],
        model_kwargs=dict(
            scoring_fct_norm=1
        ),
        model_kwargs_ranges=dict(
            embedding_dim=embedding_dim_range,
        ),
        negative_sampler=neg_samplers_dict[exp["negative_sampler"]],
        negative_sampler_kwargs=negative_sampler_kwargs,
        # Training configuration
        training_kwargs=dict(
            num_epochs=num_epochs,
            use_tqdm_batch=False,
            batch_size=batch_size,
        ),  
        #training_kwargs_ranges=dict(
        #    batch_size=batch_size_range
        #),
        loss=ShiftLogLoss,
        loss_kwargs_ranges=dict(
            shift=shift_range
        ),
        optimizer="Adam",
        optimizer_kwargs=dict(
            lr=lr,
        ),
        #optimizer_kwargs_ranges=dict(
        #    lr=lr_range
        #  ),
        #regularizer_kwargs_ranges=dict(
        #    weights=dict(type='categorical', choices=[0.1,0.01, 0.001])
        #),
        training_loop=training_loop,
        regularizer="LpRegularizer",
        regularizer_kwargs=dict(
            weight=regularization_weight,
        ),
        # Runtime configuration
        device='gpu',
        stopper="early",
        stopper_kwargs=dict(frequency=20, patience=2, relative_delta=0.002, metric="inverse_harmonic_mean_rank")
    )

    save_path = results_path_base + "/" + exp_name
    os.makedirs(save_path)
    results.save_to_directory(save_path)
