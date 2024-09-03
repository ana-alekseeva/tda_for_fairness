import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import pandas as pd
import random
import shutil

import numpy as np
import argparse

from exp_bert.train import finetune_model
from utils.utils import get_dataloader, get_dataset
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
    args = parser.parse_args()

    return args
  
def main(): 
    """
    This function is used to run the counterfactual experiments.
    """
    args = parse_args()
    
    ks = list(range(200,1000,200)) + list(range(1000,10000,1000)) # there are 26,000 samples in the training set

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

    for method in ["BM25","cosine","l2","IF","TRAK"]:

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
                val_set_size=None,
                device=DEVICE)

        df_acc = pd.DataFrame(columns = ["k","mean","std"])
        df_acc_groups = pd.DataFrame(columns = ["group","k","mean","std"])
        groups = test_df['group'].unique()
        
        
        def get_dataloader_group(group):
            g_indices = test_df.index[test_df["group"] == group].tolist()
            g_test_dl = get_dataloader(test_dataset.select(g_indices), config.TEST_BATCH_SIZE)
            return g_test_dl

        test_dl_groups = {group:get_dataloader_group(group) for group in groups} 
        
        for k in ks:
            print(k)
            new_folder = f"{args.output_dir}{method}_finetuning/{k}"
            os.mkdir(new_folder, exist_ok=True)
            
            debiased_train_idx = d3m.debias(num_to_discard=k)
            finetune_model(train_dataset.select(debiased_train_idx),val_dataset,new_folder, random_seed=42)
            
            checkpoints = os.listdir(new_folder)
            num_checkpoints = len(checkpoints[5:])
            model_accuracies = []
            model_accuracies_groups = {group:[] for group in groups}
        
            for checkpoint in checkpoints[5:]:
                # Load the model
                model_path = f"{new_folder}/{checkpoint}"
                model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
                model.eval()
            
                accuracy = utils.compute_accuracy(model, test_dl, DEVICE)
                model_accuracies.append(accuracy)
                
                for group in groups:
                    accuracy = utils.compute_accuracy(model, test_dl_groups[group], DEVICE)
                    model_accuracies_groups[group].append(accuracy)


            shutil.rmtree(new_folder)

            # Compute mean and standard error for each model
            mean = np.mean(model_accuracies)
            std_errors = np.std(model_accuracies) / np.sqrt(num_checkpoints)
           
            df_acc.loc[len(df_acc)] = [k,mean, std_errors]

            for group in groups:
                mean = np.mean(model_accuracies_groups[group])
                std_errors = np.std(model_accuracies_groups[group]) / np.sqrt(num_checkpoints)
                df_acc_groups.loc[len(df_acc_groups)] = [group, k, mean, std_errors]


        df_acc.to_csv(f"../../output/{method}_finetuning/total_accuracy.csv",index=False)
        df_acc_groups.to_csv(f"../../output/{method}_finetuning/accuracy_by_groups.csv",index=False)



    n = train_df.shape[0]
    df_acc = pd.DataFrame(columns = ["k","mean","std"])
    df_acc_groups = pd.DataFrame(columns = ["group","k","mean","std"])                                    
    for k in ks:
        print(k)
        new_folder = f"../../output/random_finetuning/{k}"
        os.mkdir(new_folder)
        random_indices = random.sample(range(n), n-k)
        finetune_model(train_dataset.select(random_indices), val_dataset,pretrained_model, tokenizer,new_folder)

        checkpoints = os.listdir(new_folder)
        num_checkpoints = len(checkpoints[5:])
        model_accuracies = []
        model_accuracies_groups = {group:[] for group in groups}

        for checkpoint in checkpoints[5:]:
            # Load the model
            model_path = f"{new_folder}/{checkpoint}"
            model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
            model.eval()

            accuracy = utils.compute_accuracy(model, test_dl, DEVICE)
            model_accuracies.append(accuracy)

            for group in groups:
                accuracy = utils.compute_accuracy(model, test_dl_groups[group], DEVICE)
                model_accuracies_groups[group].append(accuracy)

        shutil.rmtree(new_folder)

        # Compute mean and standard error for each model
        mean = np.mean(model_accuracies)
        std_errors = np.std(model_accuracies) / np.sqrt(num_checkpoints)

        df_acc.loc[len(df_acc)] = [k,mean, std_errors]

        for group in groups:
            mean = np.mean(model_accuracies_groups[group])
            std_errors = np.std(model_accuracies_groups[group]) / np.sqrt(num_checkpoints)
            df_acc_groups.loc[len(df_acc_groups)] = [group, k, mean, std_errors]


        df_acc.to_csv(f"../../output/random_finetuning/total_accuracy.csv",index=False)
        df_acc_groups.to_csv(f"../../output/random_finetuning/accuracy_by_groups.csv",index=False)