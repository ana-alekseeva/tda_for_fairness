import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from utils.utils import get_dataloader, get_dataset
from utils.modules import D3M
from exp_bert import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the matrices of TRAK and IF scores.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../output_bert/toxigen/base/best_checkpoint",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../output_bert/toxigen/",
        help="A path to load training and validation data from.",
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
        default="../vis/vis_bert_toxigen/",
        help="The path to save plots.",
    )
    
    args = parser.parse_args()

    return args

def main(): 
    args = parse_args()

    def plot_and_save_heatmap(a_scores,method):
        a_scores = np.array(a_scores)
        groups = list(train_df.group.unique())
        n = len(groups)
        matrix_avg = np.zeros((n,2))

        for i in range(n):
            for j in range(2):
                train_idx = train_df.loc[(train_df.group == groups[i]) & (train_df.label == j)].index.to_list()
                m = a_scores[train_idx]
                matrix_avg[i,j] = np.mean(m)


        plt.figure(figsize=(2,6))
        
        # Create heatmap
        heatmap = plt.imshow(matrix_avg, cmap='viridis', aspect='auto')
        plt.yticks(ticks=np.arange(n), labels=groups)
        plt.xticks(ticks=[0,1], labels=["neutral","hate"])
        
        # Add labels to the axes
        plt.xlabel('Label')
        plt.ylabel('Training samples')
        plt.title(f"{method} scores")
        
        # Add colorbar on the left side
        cbar = plt.colorbar(heatmap, orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        plt.savefig(f"{args.path_to_save}/heatmap_label_{method}.pdf")

    
    base_model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")
    train_dl = get_dataloader(train_dataset, config.TRAIN_BATCH_SIZE,shuffle=False)
    test_dl = get_dataloader(test_dataset, config.TEST_BATCH_SIZE,shuffle=False)

    train_df = pd.read_csv(args.data_dir + "train.csv").reset_index(drop=True)
    train_group_indices = train_df['group'].astype('category').cat.codes.tolist()
    test_df = pd.read_csv(args.data_dir + "test.csv").reset_index(drop=True)
    test_group_indices = test_df['group'].astype('category').cat.codes.tolist()
    

    for method in ["IF", "TRAK"]:
        try:
            scores = torch.load(f"{args.output_dir}{method}_scores.pt").numpy()
        except:
            scores = torch.load(f"{args.output_dir}{method}_scores.pt")

        d3m = D3M(
            model=base_model,
            checkpoints=[],
            train_dataloader=train_dl,
            val_dataloader = test_dl,
            group_indices_train=train_group_indices,
            group_indices_val=test_group_indices,
            scores=scores.T,
            train_set_size=None,
            val_set_size=None)
        
        group_losses = d3m.get_group_losses(
            model=d3m.model,
            val_dl=d3m.dataloaders["val"],
            group_indices=d3m.group_indices_val,
        )

        a_scores = d3m.compute_group_alignment_scores(d3m.scores, d3m.group_indices_val, group_losses)

        plot_and_save_heatmap(a_scores,method)



if __name__ == "__main__":
    main()
