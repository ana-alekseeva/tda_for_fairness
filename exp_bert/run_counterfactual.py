from exp_bert import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import random
import shutil
import json
import numpy as np
import argparse
import os

from exp_bert.train import finetune_model
from utils.utils import get_dataloader, get_dataset, compute_metrics
from utils.modules import D3M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42) # for random.randint

def parse_args():
    parser = argparse.ArgumentParser(description="Run counterfactual analysis and compute accuracies and losses for plotting.")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../output_bert/toxigen/base/best_checkpoint",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/toxigen/",
        help="A path to load training and validation data from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output_bert/toxigen/",
        help="The path to save scores.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="IF",
        help="The method that was used in the first module.",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="The number of independent runs.",
    )
    args = parser.parse_args()

    return args
  
def main(): 
    """
    This function is used to run the counterfactual experiments.
    """
    args = parse_args()
    
    seeds = [i for i in range(args.num_runs)]
    ks = [50,100,200,300,400,500, 1000, 3000, 5000, 7000]
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    val_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"val")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    train_dl = get_dataloader(train_dataset, config.TRAIN_BATCH_SIZE,shuffle=False)
    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE,shuffle=False)

    train_group_indices = pd.read_csv(args.data_dir + "train.csv")['group'].astype('category').cat.codes.tolist()
    test_df = pd.read_csv(args.data_dir + "test.csv")
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()


    os.makedirs(f"{args.output_dir}{args.method}_finetuning", exist_ok=True)

    if args.method != "random":
        scores = torch.load(f"{args.output_dir}/{args.method}_scores.pt")
        scores = scores.T

        d3m = D3M(
            model=finetuned_model,
            checkpoints=[],
            train_dataloader=train_dl,
            val_dataloader = test_dl,
            group_indices_train=train_group_indices,
            group_indices_val=test_group_indices,
            scores=scores,
            train_set_size=None,
            val_set_size=None)

    dict_metrics = {k:
                     {
                        "accuracy":[],
                        "loss":[],
                        "fpr":[],
                        "fnr":[],
                        "auc":[]
                     }
                    for k in ks
                    }
    
    groups = test_df['group'].unique()

    dict_metrics_groups = {k: {
                    group:
                     {
                        "accuracy":[],
                        "loss":[],
                        "fpr":[],
                        "fnr":[],
                        "auc":[]
                     }
                    for group in groups}
                    for k in ks
                    }
    
    def get_dataloader_group(group):
        g_indices = test_df.index[test_df["group"] == group].tolist()
        g_test_dl = get_dataloader(test_dataset.select(g_indices), config.TEST_BATCH_SIZE,shuffle=False)
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
    
            seed = seeds[i]
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
