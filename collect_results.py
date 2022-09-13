import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Aggregate results for the iterations and configurations of an experiment.")

parser.add_argument('--experiment', type=str, required=True, help="Path of the results for one experiment.")

args = parser.parse_args()

results_dir = "Output/Results/" + args.experiment
experiments = os.listdir(results_dir)

eval_metrics = pd.DataFrame(columns=["Experiment", "MRR", "Hits@10"])

for exp in experiments:
    plt.figure(figsize=(16,9))
    max_loss = 0
    min_loss = 1
    
    mrr = []
    hits10 = []
    

    for iteration in os.listdir(results_dir + "/" + exp):
        results = json.load(open(results_dir + "/" + exp + "/" + iteration + "/results.json"))
        # if max(results["losses"][50:]) > max_loss:
        #     max_loss =  max(results["losses"][50:])
        # if min(results["losses"][50:]) < min_loss:
        #     min_loss =  min(results["losses"][50:])
        #plt.plot(results["losses"])

        mrr += [results["metrics"]["both"]["realistic"]["inverse_harmonic_mean_rank"]]
        hits10 += [results["metrics"]["both"]["realistic"]["hits_at_10"]]

    eval_metrics = eval_metrics.append({"Experiment": exp, 
        "MRR":np.mean(mrr), 
        "Hits@10": np.mean(hits10)},
        ignore_index=True)


    # plt.ylim(0.8*min_loss, 1.2*max_loss)
    # plt.title("Loss history for {}".format(exp))
    # plt.legend(os.listdir(results_dir + "/" + exp))
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    #plt.savefig(results_dir + "/loss_" + exp + ".png")

eval_metrics.to_csv(results_dir + "/eval_metrics_esns.csv")
