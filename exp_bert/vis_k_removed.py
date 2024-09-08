import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)
from utils.utils import get_dataset, get_dataloader, compute_accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import config
import numpy as np
import argparse

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
        "--data_dir",
        type=str,
        default="../../data/toxigen/",
        help="A path to load training and validation data from.",
    )
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="../../output_bert/toxigen/",
        help="The path to save scores.",
    )
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")


    base_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    test_df = pd.read_csv(args.data_dir + "test.csv")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    test_dl = get_dataloader(test_dataset, 32, shuffle=False)

    base_model_acc = compute_accuracy(base_model,test_dl,DEVICE)

    colors = sns.color_palette("Set1", n_colors=6)
    methods = ["BM25","cosine","l2","IF","TRAK","random"]
    ks = [50,100,150,200,350,500,650,800, 1100,1400]

    for method,color in zip(methods, colors):
        df_acc_method = pd.read_csv(f'../../output_bert/toxigen/{method}_finetuning/total_accuracy.csv')
        df_acc_method["std"] = df_acc_method["std"]
        df_acc_method = df_acc_method.loc[df_acc_method["k"].isin(ks)]
        plt.errorbar(df_acc_method["k"], df_acc_method["mean"], yerr=df_acc_method["std"], fmt='o', capsize=5, capthick=2, ecolor='gray', color = color)
        plt.plot([0]+df_acc_method["k"].to_list(), np.hstack([base_model_acc,df_acc_method["mean"]]), color = color, label = method)

    plt.xlabel('K Removed Training Samples')
    plt.ylabel('Total Accuracy')
    plt.title('Total Accuracy')
    plt.xticks(ks)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'../vis/vis_bert_toxigen/total_accuracy.png')


    data_groups = pd.DataFrame(columns=["method","k","acc","std"])

    for method in methods:
        df_acc_method_groups = pd.read_csv(f'../../output_bert/toxigen/{method}_finetuning/accuracy_by_groups.csv')
        df_acc_method_groups = df_acc_method_groups.loc[df_acc_method_groups["k"].isin(ks)]
        df_acc_method_groups["method"] = method
        df_acc_method_groups["std"] = df_acc_method_groups["std"]
        data_groups = pd.concat([data_groups,df_acc_method_groups],axis=0)

    for group in data_groups["group"].unique():
        df_group = data_groups.loc[data_groups["group"] == group]
        
        g_indices = test_df.index[test_df["group"] == group].tolist()
        test_dl_group = get_dataloader(test_dataset.select(g_indices), 32)
        base_model_acc_group = compute_accuracy(base_model,test_dl_group,DEVICE)

        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")

        for method,color in zip(methods, colors):
            df_group_method = df_group.loc[df_group["method"] == method]
            plt.errorbar(df_group_method["k"], df_group_method["mean"], yerr=df_group_method["std"], fmt='o', capsize=5, capthick=2, ecolor='gray', color = color)
            plt.plot([0]+df_group_method["k"].to_list(), np.hstack([base_model_acc_group,df_group_method["mean"]]), color = color, label = method)

        plt.xlabel('K Removed Training Samples')
        plt.ylabel('Accuracy')
        plt.title(f'Group Accuracy: {group}')
        plt.xticks(ks)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../vis/vis_bert_toxigen/{group}_accuracy.png')

if __name__ == "__main__":
    main()
