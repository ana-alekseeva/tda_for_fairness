import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import shutil
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)

from utils.utils import get_dataloader, get_dataset, compute_metrics
from utils.modules import D3M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Run counterfactual analysis and compute accuracies and losses for plotting.")

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
        "--output_dir",
        type=str,
        default="../../output_bert/toxigen/",
        help="The path to save scores.",
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
    """
    This function is used to run the counterfactual experiments.
    """
    args = parse_args()
    
    base_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    val_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"val")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    train_dl = get_dataloader(test_dataset, config.TRAIN_BATCH_SIZE,shuffle=False)
    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE,shuffle=False)

    train_group_indices = pd.read_csv(args.data_dir + "train.csv")['group'].astype('category').cat.codes.tolist()
    test_df = pd.read_csv(args.data_dir + "test.csv")
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()

    os.makedirs(f"{args.path_to_save}attr_scores", exist_ok=True)
    
    df_tda_scores_stat = pd.DataFrame(columns=["sample","min","mean","max", "std"])
    df_tda_scores_group_stat = pd.DataFrame(columns=["group","min","mean","max", "std"])

    for method in ["IF", "TRAK"]:
        scores = torch.load(f"{args.output_dir}/{args.method}_scores.pt")
        scores = scores.T
        assert scores.shape[0] < scores.shape[1]

        groups = list(test_df["group"].unique())

        # 1.a. Distribution of TDA scores for training samples (averaged over test samples): histograms and a table of means and stds
        scores_train = scores.mean(axis=0).cpu().numpy()
        assert scores.shape[1] == scores_train[0]

        # 1.b. Distribution of TDA scores for test samples (averaged over trainng samples): histograms and a table of means and stds
        scores_test = scores.mean(axis=1).cpu().numpy()

        # Plotting histograms
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(scores_train, bins=50, alpha=0.7, color='blue', label='Training Samples')
        ax.hist(scores_test, bins=50, alpha=0.7, color='green', label='Test Samples')
        ax.set_title(f'Distribution of {method} Scores')
        ax.legend()
        fig.savefig(f'{args.path_to_save}attr_scores/Distribution_of_{method}_scores.pdf')

        # Table of means and stds
        df_tda_scores_stat.loc[method] = ["train", scores_train.min(), scores_train.mean(), scores_train.max(), scores_train.std()]
        df_tda_scores_stat.loc[method] = ["test", scores_test.min(), scores_test.mean(), scores_test.max(), scores_test.std()]

        # 2. Distribution of TDA scores for training samples (averages) by group: histograms and a table of means and stds
        for group in groups:
            scores_group = scores[test_df["group"] == group].mean(axis=0).cpu().numpy()
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(scores_group, bins=50, alpha=0.7, color='blue', label='Training Samples')
            ax.set_title(f'Distribution of {method} Scores for Group {group}')
            ax.legend()
            fig.savefig(f'{args.path_to_save}attr_scores/Distribution_of_{method}_scores_for_group_{group}.pdf')

            df_tda_scores_group_stat.loc[method] = [group,scores_group.min(), scores_group.mean(), scores_group.max(), scores_group.std()]


        # 3.a. Distribution of group attribution scores for training samples: histograms and a table of means and stds 

        d3m = D3M(
            model=base_model,
            checkpoints=[],
            train_dataloader=train_dl,
            val_dataloader = test_dl,
            group_indices_train=train_group_indices,
            group_indices_val=test_group_indices,
            scores=scores,
            train_set_size=None,
            val_set_size=None)

 

    df_tda_scores_stat.to_csv(f'{args.path_to_save}attr_scores/tda_scores_stat.csv')
    df_tda_scores_stat.to_csv(f'{args.path_to_save}attr_scores/tda_scores_group_stat.csv')




    # 5. Create a table with metrics by group for test samples and the fine-tuned base model
    df_metrics_group = pd.DataFrame(columns=["accuracy","loss","fpr","fnr","auc"])
    for group in test_df["group"].unique():
    
        g_indices = test_df.index[test_df["group"] == group].tolist()
        test_dl_group = get_dataloader(test_dataset.select(g_indices), 32)
        acc, loss, fpr, fnr, auc = compute_metrics(base_model,test_dl_group,DEVICE)
        df_metrics_group.loc[group] = [acc, loss, fpr, fnr, auc]

    df_metrics_group.to_csv(f'{args.path_to_save}attr_scores/metrics_group.csv')    





    def get_dataloader_group(group):
        g_indices = test_df.index[test_df["group"] == group].tolist()
        g_test_dl = get_dataloader(test_dataset.select(g_indices), config.TEST_BATCH_SIZE)
        return g_test_dl

    test_dl_groups = {group:get_dataloader_group(group) for group in groups} 
    
    for k in ks:
        print(k)

        if args.method == "random":
            debiased_train_idx = random.sample(range(len(train_dataset)), len(train_dataset)-k)
        else:
            debiased_train_idx = d3m.debias(num_to_discard=k)
        

        for i in range(args.num_runs):
            new_folder = f"{args.output_dir}{args.method}_finetuning/{k}/{i}/"
            os.makedirs(new_folder, exist_ok=True)
    
            seed = random.randint(0,1000)
            finetune_model(train_dataset.select(debiased_train_idx),val_dataset, new_folder, random_seed=seed)
        
            model_path = f"{new_folder}/best_checkpoint"
            model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
            model.eval()
        
            accuracy, loss, fpr, fnr, auc = compute_metrics(model, test_dl, DEVICE)
            
            dict_metrics[k]['accuracy'].append(accuracy)
            dict_metrics[k]['loss'].append(loss)
            dict_metrics[k]['fpr'].append(fpr)
            dict_metrics[k]['fnr'].append(fnr)
            dict_metrics[k]['auc'].append(auc)
            
            for group in groups:
                accuracy,loss, fpr, fnr, auc = compute_metrics(model, test_dl_groups[group], DEVICE)
                dict_metrics_groups[k][group]['accuracy'].append(accuracy)
                dict_metrics_groups[k][group]['loss'].append(loss)
                dict_metrics_groups[k][group]['fpr'].append(fpr)
                dict_metrics_groups[k][group]['fnr'].append(fnr)
                dict_metrics_groups[k][group]['auc'].append(auc)

            shutil.rmtree(new_folder)


        with open(f"{args.output_dir}{args.method}_finetuning/metrics_total_{k}.json", 'w') as f:
            json.dump(dict_metrics, f)

        with open(f"{args.output_dir}{args.method}_finetuning/metrics_groups_{k}.json", 'w') as f:
            json.dump(dict_metrics_groups, f)

if __name__ == "__main__":
    main()