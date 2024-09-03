import pandas as pd
import numpy as np
from torch.utils.data import  DataLoader
from datasets import load_dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

def get_dataloader(dataset, batch_size, shuffle=True):
    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([item['input_ids'] for item in batch]),
            'attention_mask': torch.tensor([item['attention_mask'] for item in batch]),
            'token_type_ids': torch.tensor([item['token_type_ids'] for item in batch]),
            'labels': torch.tensor([item['labels'] for item in batch]),
        }
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,collate_fn=collate_fn)


def get_dataset(
    tokenizer: object,
    max_length: int,
    data_path: str,
    split: str,
    indices: List[int] = None,
):
    assert split in ["train", "val", "test"]
    raw_datasets = load_dataset('csv', data_files={'train':data_path+'train.csv', 'val':data_path+'val.csv','test':data_path+'test.csv'})

    column_names = raw_datasets["train"].column_names
    assert "text" in column_names
    text_column_name = "text"

    def preprocess_function(examples):
        result = tokenizer(examples[text_column_name], padding="max_length", max_length=max_length, truncation=True)
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





def plot_distr_by_group(df,var='label',title="",path_to_save="../vis/"):
    """
    Plot the distribution of samples by group.
    """
    # Create a count of samples for each combination of label and target group
    grouped_data = df.groupby(['group', var]).size().unstack()

    # Set up the plot
    plt.figure(figsize=(12, 6))

    # Create the grouped bar plot
    grouped_data.plot(kind='bar', ax=plt.gca())

    # Customize the plot
    if var=='label':
        plt.title('Distribution of Labels by Group')
        plt.legend(['Neutral (0)', 'Hate (1)'])
    if var=='generation_method':
        plt.title('Distribution of Generation Methods by Group')
        plt.legend(['ALICE', 'top-k'])
    plt.xlabel('Group')
    plt.ylabel('Number of Samples')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add value labels on top of each bar
    for i in range(len(grouped_data)):
        for j in range(len(grouped_data.columns)):
            value = grouped_data.iloc[i, j]
            plt.text(i, value, str(value), ha='center', va='bottom')

    # Adjust layout to prevent cutoff
    plt.tight_layout()
    plt.savefig(f'{path_to_save}distr_by_group_{var}_{title}.pdf')



def compute_accuracy(model, dataloader, device="cuda"):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k:batch[k].to(device) for k in batch.keys()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(pred)
            true_labels.extend(batch['labels'].cpu().numpy())

    return sum(np.array(true_labels) == np.array(predictions) ) / len(predictions)

def compute_predictions(model, dataloader, device="cuda"):
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k:batch[k].to(device) for k in batch.keys()}
            outputs = model(**batch)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(pred)

    return predictions
