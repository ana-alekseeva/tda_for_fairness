import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Plot the matrices of TRAK and IF scores.")

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

    def plot_and_save_heatmap(matrix,method):
        groups = list(train_df.group.unique())
        n = len(groups)
        matrix_avg = np.zeros((n,n))

        for i in range(n):
            for j in range(n):
                train_idx = train_df.loc[train_df.group == groups[i]].index.to_list()
                test_idx = test_df.loc[test_df.group == groups[j]].index.to_list()
                m = matrix[train_idx,:]
                m = m[:,test_idx]
                matrix_avg[i,j] = np.mean(m)


        plt.figure(figsize=(6,6))
        
        # Create heatmap
        heatmap = plt.imshow(matrix_avg, cmap='viridis', aspect='auto')
        plt.xticks(ticks=np.arange(n), labels=groups, rotation=45)
        plt.yticks(ticks=np.arange(n), labels=groups)
        
        # Add labels to the axes
        plt.xlabel('Test samples')
        plt.ylabel('Training samples')
        plt.title(f"{method} scores")
        
        # Add colorbar on the left side
        cbar = plt.colorbar(heatmap, orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        plt.savefig(f"{args.path_to_save}/heatmap_{method}.pdf")
    
    scores_IF = torch.load(f"{args.output_dir}/IF_scores.pt").numpy().T
    scores_TRAK = torch.load(f"{args.output_dir}/TRAK_scores.pt").T

    test_df = pd.read_csv(args.data_dir + "test.csv")
    train_df = pd.read_csv(args.data_dir + "train.csv")


    plot_and_save_heatmap(scores_IF,"IF")
    plot_and_save_heatmap(scores_TRAK,"TRAK")



if __name__ == "__main__":
    main()
