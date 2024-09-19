import pandas as pd
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)
from utils.utils import get_dataset, get_dataloader, compute_accuracy
from utils.modules import D3M
import config
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

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    test_df = pd.read_csv(args.data_dir + "test.csv")
    train_df  = pd.read_csv(args.data_dir + "train.csv")
    train_group_indices = train_df['group'].astype('category').cat.codes.tolist()
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()

    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    test_dl = get_dataloader(test_dataset, 32, shuffle=False)
    train_dl = get_dataloader(train_dataset, 32, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)    
    model.eval()

    for method in ["IF","TRAK"]:
        scores = torch.load(f"{args.path_to_save}{method}_scores.pt")
        scores = scores.T

        d3m = D3M(
                    model=model,
                    checkpoints=[],
                    train_dataloader=train_dl,
                    val_dataloader = test_dl,
                    group_indices_train=train_group_indices,
                    group_indices_val=test_group_indices,
                    scores=scores,
                    train_set_size=None,
                    val_set_size=None)

        group_losses = d3m.get_group_losses(
                                model=d3m.model,
                                val_dl=d3m.dataloaders["val"],
                                group_indices=d3m.group_indices_val,
                                )

        group_alignment_scores = d3m.compute_group_alignment_scores(
                d3m.scores, d3m.group_indices_val, group_losses
            )

        train_df[method] = group_alignment_scores


    train_df.to_csv(f'{args.path_to_save}examples.csv')

if __name__ == "__main__":
    main()

