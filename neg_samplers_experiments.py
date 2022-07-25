import os
from modified_pykeen.pipeline_modified import pipeline
from modified_pykeen.slcwa_modified import SLCWATrainingLoop, SLCWATrainingLoopModified
from negative_samplers.esns_relaxed import ESNSRelaxed
from negative_samplers.esns_ridle import ESNSRidle

experiments = [
    #{"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_ridle", "similarity_metric": "cosine"},
    #{"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_relaxed", "similarity_metric": "jacccard"},
    #{"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "WN18", "negative_sampler": "esns_ridle", "similarity_metric": "cosine"},
    {"model": "TransE", "dataset": "WN18", "negative_sampler": "esns_relaxed", "similarity_metric": "jacccard"},
]


neg_samplers_dict = {"basic": "basic", "esns_relaxed": ESNSRelaxed, "esns_ridle": ESNSRidle}

index_column_size=1000
index_path_base = "EII"
sampling_size=100
q_set_size=50
n_triples_for_ns_qual_analysis=5
ns_qual_analysis_every=20


results_path_base = "Output/Results"
checkpoint_path = "Output/Checkpoints"
num_epochs=100
device="gpu"

if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)


for exp in experiments:

    exp_name = "-".join(list(exp.values()))
    exp["exp_name"] = exp_name
    print("Training for {}".format(exp_name))

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
    

    results= pipeline(
        dataset=exp["dataset"],
        model=exp["model"],
        negative_sampler=neg_samplers_dict[exp["negative_sampler"]],
        negative_sampler_kwargs=negative_sampler_kwargs,
        # Training configuration
        training_kwargs=dict(
            num_epochs=num_epochs,
            use_tqdm_batch=False,
            checkpoint_name=os.getcwd()+ '/' + checkpoint_path + '/' + exp_name +'.pt',
        ),  
        # Runtime configuration
        training_loop=training_loop,
        random_seed=1235,
        device=device,
        stopper="early",
        stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002)
    )

    save_path = results_path_base + "/" + exp_name
    os.makedirs(save_path)
    results.save_to_directory(save_path)