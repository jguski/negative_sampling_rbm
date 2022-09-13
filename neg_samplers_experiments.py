import os
import json
import logging
import numpy as np
from modified_pykeen.pipeline_modified import pipeline
from modified_pykeen.slcwa_modified import SLCWATrainingLoop, SLCWATrainingLoopModified
from negative_samplers import ESNSStandard, ESNSRelaxed, ESNSRidle
from losses.custom_losses import ShiftLogLoss

experiments = [
    #{"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "basic"},
    #{"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_standard", "similarity_metric": "absolute"},
    #{"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    #{"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "reconstructed"},
    #{"model": "TransE", "dataset": "FB15k-237", "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "compressed"},
    {"model": "TransE", "dataset": "WN18RR", "negative_sampler": "basic"},
    {"model": "TransE", "dataset": "WN18RR", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "WN18RR", "negative_sampler": "esns_standard", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "WN18RR", "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "reconstructed"},
    {"model": "TransE", "dataset": "WN18RR", "negative_sampler": "esns_ridle", "similarity_metric": "cosine", "rbm_layer": "compressed"},
]



neg_samplers_dict = {"basic": "basic", "esns_relaxed": ESNSRelaxed, "esns_ridle": ESNSRidle, "esns_standard": ESNSStandard}

n_iterations=3
index_column_size=100
index_path_base = "EII"
sampling_size=100
q_set_size=50
n_triples_for_ns_qual_analysis=20
ns_qual_analysis_every=20


results_path_base = "Output/Results"
checkpoint_path = "Output/Checkpoints"
hpo_path = "Output/hpo"
quality_analysis_path = "Output/NS_quality_analysis"
num_epochs=1000
device="gpu"

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

logger = logging.getLogger()

for exp in experiments:
    # parameter setup
    exp_name = "-".join(list(exp.values()))
    exp["exp_name"] = exp_name
    print("Training for {}".format(exp_name))

    try:
        parameters_path = hpo_path + "/" + exp_name + "/best_pipeline/pipeline_config.json"
        hpo = json.load(open(parameters_path))
        embedding_dim = hpo["pipeline"]["model_kwargs"]["embedding_dim"]
        lr = hpo["pipeline"]["optimizer_kwargs"]["lr"]
        batch_size = hpo["pipeline"]["training_kwargs"]["batch_size"]
        margin = hpo["pipeline"]["loss_kwargs"]["margin"]
    except FileNotFoundError:
        logger.warning("No file found under {}. Using default hyperparameters instead.".format(parameters_path))
        embedding_dim = 100
        lr = 0.001
        batch_size = 64
        margin = 1


    if "esns" in exp["negative_sampler"]:
        negative_sampler_kwargs=dict(
            index_column_size=index_column_size,
            sampling_size=sampling_size,
            q_set_size=q_set_size,
            similarity_metric=exp["similarity_metric"],
            n_triples_for_ns_qual_analysis=n_triples_for_ns_qual_analysis,
            ns_qual_analysis_every=ns_qual_analysis_every,
            logging_level="INFO"
        )
        training_loop=SLCWATrainingLoopModified
    else:
        negative_sampler_kwargs=dict()
        training_loop = SLCWATrainingLoop
    
    # get a reproducible list of seeds for the iterations
    np.random.seed(42)
    seeds=np.random.random_integers(0, 2000000000, n_iterations)
    # loop to train model n_iterations times
    for it in range(n_iterations):
        if "esns" in exp["negative_sampler"]:
            negative_sampler_kwargs['ns_qual_analysis_path'] = quality_analysis_path + "/" + exp_name + "/iteration_{:02d}".format(it)
        
        seed=seeds[it]

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
                loss_kwargs=dict(
                    margin=margin,
                ),
                optimizer_kwargs=dict(
                    lr=lr,
                ),
                training_loop=training_loop,
                regularizer="LpRegularizer",
                regularizer_kwargs=dict(
                    weight=0.0001,
                ),
                # Runtime configuration
                random_seed=seed,
                device=device,
                stopper="early",
                stopper_kwargs=dict(frequency=10, patience=2, relative_delta=0.002, metric="inverse_harmonic_mean_rank")
            )

        save_path = results_path_base + "/" + exp_name + "/iteration_{:02d}".format(it)
        os.makedirs(save_path, exist_ok=True)
        results.save_to_directory(save_path)

        # remove checkpoint so it will not be loaded in next iteration
        os.remove(checkpoint_path + '/' + exp_name +'.pt')
