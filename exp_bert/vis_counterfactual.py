import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
from utils.utils import get_dataset, get_dataloader, compute_metrics
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from exp_bert import config
import numpy as np
import argparse
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the results of counterfactual analysis.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../../output_bert/toxigen/base/best_checkpoint",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../../output_bert/toxigen/",
        help="A path to load training and validation data from.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/toxigen/",
        help="A path to load training and validation data from.",
    )
    
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="../vis/vis_bert_toxigen/",
        help="The path to save plots.",
    )
    
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    base_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    test_df = pd.read_csv(args.data_dir + "test.csv").reset_index(drop=True)
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    test_dl = get_dataloader(test_dataset, 32, shuffle=False)
    base_model_metrics = compute_metrics(base_model,test_dl,DEVICE)

    methods = ["IF","TRAK","random"]
    colors = sns.color_palette("Set1", n_colors=len(methods))

    for i,metric in enumerate(["accuracy","loss","fpr","fnr","auc"]):
        os.makedirs(args.path_to_save + metric, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        for method,color in zip(methods, colors):

            with open(f'{args.results_dir}{method}_finetuning/metrics_total.json') as f:
                data = json.load(f)
            
            means_by_k = []
            ks = sorted([int(i) for i in data.keys()])
            ks = [str(i) for i in ks]

            for k in ks:
                m = np.mean(data[k][metric])
                means_by_k.append(m)
            
            plt.plot(['0']+ks, np.hstack([base_model_metrics[i],means_by_k]), color = color, label = method)
            plt.plot()

        plt.xlabel('K Removed Training Samples')
        plt.ylabel(metric)
        plt.title(metric)
        plt.xticks(['0']+ks)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'{args.path_to_save}{metric}/total_{metric}.pdf')



    for group in test_df["group"].unique():
        
        g_indices = test_df.index[test_df["group"] == group].tolist()
        test_dl_group = get_dataloader(test_dataset.select(g_indices), 32)
        base_model_metrics_group = compute_metrics(base_model,test_dl_group,DEVICE)

        for i,metric in enumerate(["accuracy","loss","fpr","fnr","auc"]):

            plt.figure(figsize=(8, 6))
            sns.set_style("whitegrid")
            for method,color in zip(methods, colors):
                with open(f'{args.results_dir}{method}_finetuning/metrics_groups.json') as f:
                    data = json.load(f)
                
                means_by_k = []
                ks = sorted([int(i) for i in data.keys()])
                ks = [str(i) for i in ks]

                for k in ks:
                    m = np.mean(data[k][group][metric])
                    means_by_k.append(m)
                
                plt.plot(['0']+ks, np.hstack([base_model_metrics_group[i],means_by_k]), color = color, label = method)

            plt.xlabel('K Removed Training Samples')
            plt.ylabel(metric)
            plt.title(f"{metric}: {group}")
            plt.xticks(['0']+ks)
            plt.legend()

            plt.tight_layout()
            plt.savefig(f'{args.path_to_save}{metric}/{group}_{metric}.pdf')



if __name__ == "__main__":
    main()
