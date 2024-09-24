from datasets import load_dataset
import argparse
import pandas as pd
from exp_bert import config
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from exp_bert.train import finetune_model
from utils.utils import compute_metrics, get_dataloader,get_dataset
import torch
from utils.modules import D3M
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Ordinary dataset balancing.")

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
        "--path_to_save",
        type=str,
        default="res/toxigen/",
        help="The path to save generated datasets.",
    )
    parser.add_argument(
        "--path_to_save_model",
        type=str,
        default="../output_bert/toxigen/",
        help="The path to save generated datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output_bert/toxigen/",
        help="The path to save scores.",
    )
    args = parser.parse_args()

    return args
  

def main():
    args = parse_args()

    train_df = pd.read_csv(args.data_dir+"train.csv").reset_index(drop=True)
    test_df = pd.read_csv(args.data_dir+"test.csv").reset_index(drop=True)
    val_df = pd.read_csv(args.data_dir+"val.csv")

    # Create a balanced dataset

    df_train_balanced = (train_df.groupby(['group','label'],group_keys=False)
                                 .apply(lambda x: 
                                        x.sample(train_df.group.value_counts().min()),
                                        random_state=args.seed)
                                    .reset_index(drop=True)
                )

    df_val_balanced = (val_df.groupby(['group','label'],group_keys=False)
                             .apply(lambda x: 
                                    x.sample(val_df.group.value_counts().min()),
                                    random_state=args.seed)
                            .reset_index(drop=True)
                )   
    
    samples_removed = train_df.shape[0] - df_train_balanced.shape[0]

    df_train_balanced.to_csv(args.data_dir+"train_balanced.csv",index=False)
    df_val_balanced.to_csv(args.data_dir+"val_balanced.csv",index=False)

    # Load the datasets
    raw_datasets = load_dataset('csv', data_files={'train':args.data_dir+'train_balanced.csv', 'val':args.data_dir+'val_balanced.csv','test':args.data_dir+'test.csv'})
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    column_names = raw_datasets["train"].column_names
    assert "text" in column_names
    text_column_name = "text"

    def preprocess_function(examples):
        result = tokenizer(examples[text_column_name], padding="max_length", max_length=config.MAX_LENGTH, truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        remove_columns=column_names,
    )

   
    train_dataset_balanced = raw_datasets["train"]
    val_dataset_balanced = raw_datasets["val"]
    test_dataset = raw_datasets["test"]  

    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE)

    new_folder = args.path_to_save_model + "base_balanced/"
    finetune_model(train_dataset_balanced,val_dataset_balanced, new_folder, random_seed=config.RANDOM_STATE)  

    model_path = f"{new_folder}/best_checkpoint"
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
    model.eval()
        
    accuracy_b, loss_b, fpr_b, fnr_b, auc_b = compute_metrics(model, test_dl, DEVICE)

    comparison_table = pd.DataFrame(columns=["accuracy","loss","fpr","fnr","auc", "samples_removed"])
    comparison_table.loc["balanced"] = [accuracy_b, loss_b, fpr_b, fnr_b, auc_b, samples_removed]

    def get_dataloader_group(group):
        g_indices = test_df.index[test_df["group"] == group].tolist()
        g_test_dl = get_dataloader(test_dataset.select(g_indices), config.TEST_BATCH_SIZE,shuffle=False)
        return g_test_dl
    
    groups = test_df["group"].unique()
    test_dl_groups = {group:get_dataloader_group(group) for group in groups} 
    
    for group in groups:
        accuracy,loss, fpr, fnr, auc = compute_metrics(model, test_dl_groups[group], DEVICE)
        samples_removed = train_df[train_df["group"] == group].shape[0] - df_train_balanced[df_train_balanced["group"] == group].shape[0]
        comparison_table.loc["balanced" + "_" + group] = [accuracy, loss, fpr, fnr, auc, samples_removed]


    # Remove all harmful samples from the training set

    finetuned_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir, num_labels = 2).to(DEVICE)
    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    val_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"val")
    train_dl = get_dataloader(train_dataset, config.TRAIN_BATCH_SIZE,shuffle=False)
    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE,shuffle=False)

    train_group_indices = pd.read_csv(args.data_dir + "train.csv")['group'].astype('category').cat.codes.tolist()
    test_df = pd.read_csv(args.data_dir + "test.csv")
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()


    for method in ["IF", "TRAK"]:
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
        
        debiased_train_idx = d3m.debias(use_heuristic = True)
        samples_removed = train_df.shape[0] - len(debiased_train_idx)

        new_folder = f"{args.output_dir}{args.method}_finetuning/heuristic/"
        os.makedirs(new_folder, exist_ok=True)
    
        finetune_model(train_dataset.select(debiased_train_idx),val_dataset, new_folder, random_seed=config.RANDOM_STATE)
        
        model_path = f"{new_folder}/best_checkpoint"
        model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
        model.eval()
        
        accuracy, loss, fpr, fnr, auc = compute_metrics(model, test_dl, DEVICE)
        comparison_table.loc[method] = [accuracy, loss, fpr, fnr, auc, samples_removed]

        for group in groups:
            accuracy,loss, fpr, fnr, auc = compute_metrics(model, test_dl_groups[group], DEVICE)
            samples_removed = train_df[train_df["group"] == group].shape[0] - train_df.loc[debiased_train_idx][train_df["group"] == group].shape[0]
            comparison_table.loc[method + "_" + group] = [accuracy, loss, fpr, fnr, auc, samples_removed]

    comparison_table.to_csv(args.path_to_save + "comparison_table.csv", index=False)

if __name__ == "__main__":
    main()