import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import random
import shutil

import numpy as np
import argparse

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)

from exp_bert.train import finetune_model
from utils.utils import get_dataloader, get_dataset, compute_accuracy_and_loss
from utils.modules import D3M

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42) # for random.randint

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
    args = parser.parse_args()

    return args
  
def main(): 
    """
    This function is used to run the counterfactual experiments.
    """
    args = parse_args()
    
    ks = [50,100,150,200,350,500,650,800, 1100,1400]

    pretrained_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME,num_labels = 2).to(DEVICE)
    finetuned_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    val_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"val")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    train_dl = get_dataloader(test_dataset, config.TRAIN_BATCH_SIZE)
    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE)

    train_group_indices = pd.read_csv(args.data_dir + "train.csv")['group'].astype('category').cat.codes.tolist()
    test_df = pd.read_csv(args.data_dir + "test.csv")
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()

    for method in ["random"]:

        os.makedirs(f"{args.output_dir}{method}_finetuning", exist_ok=True)

        if method != "random":
            scores = torch.load(f"{args.output_dir}/{method}_scores.pt")
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

        df_acc = pd.DataFrame(columns = ["k","mean","std"])
        df_loss = pd.DataFrame(columns = ["k","mean","std"])
        df_acc_groups = pd.DataFrame(columns = ["group","k","mean","std"])
        df_loss_groups = pd.DataFrame(columns = ["group","k","mean","std"])
        groups = test_df['group'].unique()
        
        def get_dataloader_group(group):
            g_indices = test_df.index[test_df["group"] == group].tolist()
            g_test_dl = get_dataloader(test_dataset.select(g_indices), config.TEST_BATCH_SIZE)
            return g_test_dl

        test_dl_groups = {group:get_dataloader_group(group) for group in groups} 
        
        for k in ks:
            print(k)

            if method == "random":
                debiased_train_idx = random.sample(range(len(train_dataset)), len(train_dataset)-k)
            else:
                debiased_train_idx = d3m.debias(num_to_discard=k)
            

            accuracies = []
            accuracies_groups = {group:[] for group in groups}

            losses = []
            losses_groups = {group:[] for group in groups}

            for i in range(5): # 5 runs for error bars
                new_folder = f"{args.output_dir}{method}_finetuning/{k}/{i}/"
                os.makedirs(new_folder, exist_ok=True)
        
                seed = random.randint(0,1000)
                finetune_model(train_dataset.select(debiased_train_idx),val_dataset,new_folder, random_seed=seed)
            
                model_path = f"{new_folder}/best_checkpoint"
                model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
                model.eval()
            
                accuracy,loss = compute_accuracy_and_loss(model, test_dl, DEVICE)
                accuracies.append(accuracy)
                losses.append(loss)
                
                for group in groups:
                    accuracy,loss = compute_accuracy_and_loss(model, test_dl_groups[group], DEVICE)
                    accuracies_groups[group].append(accuracy)
                    losses_groups[group].append(loss)

                shutil.rmtree(new_folder)

            # Compute mean and standard error for each model
            mean_acc = np.mean(accuracies)
            std_errors_acc = np.std(accuracies)

            mean_loss = np.mean(losses)
            std_errors_loss = np.std(losses)
           
            df_acc.loc[len(df_acc)] = [k,mean_acc, std_errors_acc]
            df_loss.loc[len(df_loss)] = [k,mean_loss, std_errors_loss]

            for group in groups:
                mean_acc = np.mean(accuracies_groups[group])
                std_errors_acc = np.std(accuracies_groups[group])
                
                mean_loss = np.mean(losses_groups[group])
                std_errors_loss = np.std(losses_groups[group])

                df_acc_groups.loc[len(df_acc_groups)] = [group, k, mean_acc, std_errors_acc]
                df_loss_groups.loc[len(df_loss_groups)] = [group, k, mean_loss, std_errors_loss]


        df_acc.to_csv(f"{args.output_dir}{method}_finetuning/total_accuracy.csv",index=False)
        df_acc_groups.to_csv(f"{args.output_dir}{method}_finetuning/accuracy_by_groups.csv",index=False)

        df_loss.to_csv(f"{args.output_dir}{method}_finetuning/loss.csv",index=False)
        df_loss_groups.to_csv(f"{args.output_dir}{method}_finetuning/loss_by_groups.csv",index=False)



if __name__ == "__main__":
    main()
