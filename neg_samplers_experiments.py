import os
import json
import logging
import numpy as np
from modified_pykeen.pipeline_modified import pipeline
from modified_pykeen.slcwa_modified import SLCWATrainingLoop, SLCWATrainingLoopModified
from negative_samplers import *
from losses.custom_losses import ShiftLogLoss

model = "TransE"
dataset = "FB15k-237"

experiments = [
    # {"model": model, "dataset": dataset, "negative_sampler": "basic"},
    # {"model": model, "dataset": dataset, "negative_sampler": "bernoulli"},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_standard", "similarity_metric": "absolute"},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_standard", "similarity_metric": "absolute", "index_column_size": 0},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_standard", "similarity_metric": "absolute", "index_column_size": 1000},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "reconstructed"},
    # {"model": model, "dataset": dataset, "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "compressed"},
    {"model": model, "dataset": dataset, "negative_sampler": "esns_baseline_no_exploration", "index_column_size": 1000},
    {"model": model, "dataset": dataset, "negative_sampler": "esns_relaxed_no_exploration", "similarity_metric": "absolute", "index_column_size": 1000},
    {"model": model, "dataset": dataset, "negative_sampler": "esns_ridle_no_exploration", "similarity_metric": "cosine", "rbm_layer": "reconstructed", "index_column_size": 1000},
    {"model": model, "dataset": dataset, "negative_sampler": "esns_ridle_no_exploration", "similarity_metric": "cosine", "rbm_layer": "compressed", "index_column_size": 1000}
]


neg_samplers_dict = {"basic": "basic", 
    "bernoulli": "bernoulli", 
    "esns_relaxed": ESNSRelaxed, 'esns_relaxed_no_exploration': ESNSRelaxedNoExploration,
    "esns_ridle": ESNSRidle, 'esns_ridle_no_exploration': ESNSRidleNoExploration,
    "esns_standard": ESNSStandard, "esns_baseline_no_exploration": ESNSBaselineNoExploration}

n_iterations=3
#sampling_size=100 # these are the default values
#q_set_size=50
# do not perform NS quality analysis during KGE fitting
n_triples_for_ns_qual_analysis=0
ns_qual_analysis_every=0

# set paths
results_path_base = "Output/Results"
checkpoint_path = "Output/Checkpoints"
hpo_path = "Output/hpo"
quality_analysis_path = "Output/NS_quality_analysis"
num_epochs=1000
device="gpu"

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(results_path_base):
    os.makedirs(results_path_base)

#logger = logging.getLogger()
logging.getLogger().setLevel(logging.INFO)

# loop over specified experiments
for exp in experiments:
    # parameter setup
    exp_name = "-".join(str(v) for v in list(exp.values()))
    exp["exp_name"] = exp_name
    logging.info("Training for {}".format(exp_name))

    # initialize parameters according to experiment and pipeline_config.json of best hpo pipeline
    if "index_column_size" in exp.keys():
        index_column_size = exp["index_column_size"]
    else:
        index_column_size = 100
    
    try:
        parameters_path = hpo_path + "/" + "-".join(list(exp.values())[0:2]) + "/best_pipeline/pipeline_config.json"
        hpo = json.load(open(parameters_path))
        embedding_dim = hpo["pipeline"]["model_kwargs"]["embedding_dim"]
        shift = hpo["pipeline"]["loss_kwargs"]["shift"]
        lr = hpo["pipeline"]["optimizer_kwargs"]["lr"]
        batch_size = hpo["pipeline"]["training_kwargs"]["batch_size"]
        regularization_weight = hpo["pipeline"]["regularizer_kwargs"]["weight"]
        num_negs_per_pos = hpo["pipeline"]["negative_sampler_kwargs"]["num_negs_per_pos"]
        logging.info("Loaded parameters from hpo run (shift={}, embedding_dim={}, lr={}, batch_size={}, regularization_weight={}, num_negs_per_pos={}).".format(shift, embedding_dim, lr, batch_size, regularization_weight, num_negs_per_pos))
    except FileNotFoundError:
        # if no pipeline_config.json file found: Use default parameters instead
        embedding_dim = 200
        shift = 0
        lr = 0.001
        batch_size = 1024
        regularization_weight = 0.0001
        num_negs_per_pos = 1
        logging.warning("No file found under {}. Using default hyperparameters instead (shift={}, embedding_dim={}, lr={}, batch_size={}, regularization_weight={}, num_negs_per_pos={}).".format(parameters_path,shift, embedding_dim, lr, batch_size, regularization_weight, num_negs_per_pos))

    # set ESNS parameters
    if "esns" in exp["negative_sampler"]:
        negative_sampler_kwargs=dict(
            index_column_size=index_column_size,
            max_index_column_size=index_column_size,
            n_triples_for_ns_qual_analysis=n_triples_for_ns_qual_analysis,
            ns_qual_analysis_every=ns_qual_analysis_every,
            logging_level="INFO"
        )
        if "similarity_metric" in exp.keys():
            negative_sampler_kwargs["similarity_metric"]=exp["similarity_metric"]
        if "rbm_layer" in exp.keys():
            negative_sampler_kwargs["rbm_layer"] = exp["rbm_layer"]
        training_loop=SLCWATrainingLoopModified
    else:
        negative_sampler_kwargs=dict()
        training_loop = SLCWATrainingLoop
    negative_sampler_kwargs["num_negs_per_pos"] = num_negs_per_pos
    
    # get a reproducible list of seeds for the iterations
    np.random.seed(42)
    seeds=np.random.random_integers(0, 2000000000, n_iterations)

    # loop to train model n_iterations times

    for it in range(n_iterations):
        if "esns" in exp["negative_sampler"]:
            negative_sampler_kwargs['ns_qual_analysis_path'] = quality_analysis_path + "/" + exp_name + "/iteration_{:02d}".format(it)
        
        seed=seeds[it]

        # run PyKEEN pipeline with fixed seed
        results= pipeline(
                dataset=exp["dataset"],
                model=exp["model"],
                model_kwargs=dict(
                    embedding_dim=embedding_dim,
                ),
                negative_sampler=neg_samplers_dict[exp["negative_sampler"]],
                negative_sampler_kwargs=negative_sampler_kwargs,
                # Training configuration
                training_kwargs=dict(
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    use_tqdm_batch=False,
                    checkpoint_name=os.getcwd()+ '/' + checkpoint_path + '/' + exp_name +'.pt',
                ),  
                loss=ShiftLogLoss,
                loss_kwargs=dict(
                    shift = shift
                ),
                optimizer_kwargs=dict(
                    lr=lr,
                ),
                training_loop=training_loop,
                regularizer="LpRegularizer",
                regularizer_kwargs=dict(
                    weight=regularization_weight,
                ),
                # Runtime configuration
                random_seed=seed,
                device=device,
                stopper="early",
                stopper_kwargs=dict(frequency=40, patience=2, relative_delta=0.002, metric="inverse_harmonic_mean_rank"),
                evaluation_entity_whitelist=None
            )

        save_path = results_path_base + "/" + exp_name + "/iteration_{:02d}".format(it)
        os.makedirs(save_path, exist_ok=True)
        results.save_to_directory(save_path)

        # remove checkpoint so it will not be loaded in next iteration
        os.remove(checkpoint_path + '/' + exp_name +'.pt')
