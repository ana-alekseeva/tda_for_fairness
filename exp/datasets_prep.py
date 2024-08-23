import pandas as pd
from typing import List
from datasets import load_dataset
import config
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os
import torch


def prepare_toxigen(path_to_data="../../data/toxigen/",test_samples_per_group = 50):

    if not os.path.exists(path_to_data + "train.csv"):
        train_df = pd.read_csv(path_to_data+"annotated_train.csv") 
        train_df['label'] = 1*(train_df['toxicity_human'] > 2)
        train_df = train_df.dropna(subset=["text","label","target_group"])
        train_df["text"] = [i[2:-1] for i in train_df["text"]]
        train_df = train_df.loc[train_df["text"].str.len() != 0]
        train_df["label"] = train_df["label"].replace({"hate":1,"neutral":0}).astype(int)
        train_df = train_df.loc[:,["text","label","target_group"]]

        train_df.to_csv(path_to_data+"train.csv", index=False)

    if not os.path.exists(path_to_data + "val.csv"):
        val_df = pd.read_csv(path_to_data+"annotated_test.csv")
        val_df["target_group"] = val_df["target_group"].replace('black folks / african-americans', 'black/african-american folks')
        val_df['label'] = 1*(val_df['toxicity_human'] > 2)
        val_df = val_df.dropna(subset=["text","label","target_group"])
        val_df = val_df.loc[val_df["text"].str.len() != 0]
        val_df["label"] = val_df["label"].astype(int)
        val_df = val_df.loc[:,["text","label","target_group"]]

        val_df.to_csv(path_to_data+"val.csv", index=False)

    if not os.path.exists(path_to_data + "test.csv"):
        test_df = load_dataset("toxigen/toxigen-data", "annotations")["train"].to_pandas()
        test_df.columns = [i.replace("Input.","") for i in test_df.columns]
        test_df = test_df.loc[:,["text","binary_prompt_label","target_group"]].groupby("text").first().reset_index()
        test_df["label"] = test_df["binary_prompt_label"]
        test_df = test_df.drop("binary_prompt_label",axis=1).loc[:,["text","label","target_group"]]
        test_df["text"] = [i[2:-1] for i in test_df["text"]]
        test_df = test_df.dropna(subset=["text","label","target_group"])
        test_df = test_df.loc[test_df["text"].str.len() != 0]
        test_df = test_df.groupby(['target_group','label'], group_keys=False).apply(lambda x: x.sample(int(test_samples_per_group/2)))
        
        test_df.to_csv(path_to_data+"test.csv", index=False)


def get_toxigen_dataset(
   # data_name: str,
    split: str,
    indices: List[int] = None,
):
    assert split in ["train", "val", "test"]
    path = "../../data/toxigen/"
    raw_datasets = load_dataset('csv', data_files={'train':path+'train.csv', 'val':path+'val.csv','test':path+'test.csv'})

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    column_names = raw_datasets["train"].column_names
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

    if split == "train":
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    if split == "val":
        eval_dataset = raw_datasets["val"]
        ds = eval_dataset
    if split == "test":
        eval_dataset = raw_datasets["test"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)
    return ds

def get_dataloader(dataset, batch_size):
    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch]),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch]),
        }
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)