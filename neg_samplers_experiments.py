import os
from pipeline import pipeline
from esns_relaxed import ESNSRelaxed

experiments = [
    {"model": "TransE", "dataset": "WN18", "negative_sampler": "basic"},
    {"model": "TransE", "dataset": "WN18", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "WN18", "negative_sampler": "esns_relaxed", "similarity_metric": "jaccard"},
    {"model": "TransE", "dataset": "FB15k", "negative_sampler": "basic"},
    {"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_relaxed", "similarity_metric": "absolute"},
    {"model": "TransE", "dataset": "FB15k", "negative_sampler": "esns_relaxed", "similarity_metric": "jaccard"}
]


neg_samplers_dict = {"basic": "basic", "esns_relaxed": ESNSRelaxed}

index_column_size=1000
index_path_base = "EII"
sampling_size=100
q_set_size=50

results_path_base = "results"
checkpoint_path = "checkpoints"
num_epochs=100
device="cpu"

if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)


for exp in experiments:

    exp_name = "-".join(list(exp.values()))
    exp["exp_name"] = exp_name
    print("Training for {}".format(exp_name))

    if "esns" in exp["negative_sampler"]:
        negative_sampler_kwargs=dict(
            index_column_size=index_column_size,
            index_path=index_path_base + "/" + exp["dataset"],
            sampling_size=sampling_size,
            q_set_size=q_set_size,
            similarity_metric=exp["similarity_metric"]
        )
    else:
        negative_sampler_kwargs=dict()
    

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
        random_seed=1235,
        device=device,
    )

    save_path = results_path_base + "/" + exp_name
    os.makedirs(save_path)
    results.save_to_directory(save_path)