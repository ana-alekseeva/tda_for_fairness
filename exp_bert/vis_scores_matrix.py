import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

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
    """
    This function is used to run the counterfactual experiments.
    """
    args = parse_args()

    def plot_and_save_heatmap(matrix,method):
        plt.figure(figsize=(10,10))
        
        # Create heatmap
        heatmap = plt.imshow(matrix, cmap='viridis', aspect='auto')
        
        # Add labels to the axes
        plt.xlabel('Training samples')
        plt.ylabel('Test samples')
        plt.title(f"{method} scores")
        
        # Add colorbar on the left side
        cbar = plt.colorbar(heatmap, orientation='vertical')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

        plt.savefig(f"{args.path_to_save}/heatmap_{method}.pdf")
    
    scores_IF = torch.load(f"{args.output_dir}/IF_scores.pt").numpy().T
    scores_TRAK = torch.load(f"{args.output_dir}/TRAK_scores.pt").numpy().T

    test_df = pd.read_csv(args.data_dir + "test.csv")
    train_df = pd.read_csv(args.data_dir + "train.csv")

    train_sorted_idx = train_df.sort_values(by='group').index
    test_sorted_idx = test_df.sort_values(by='group').index


    plot_and_save_heatmap(scores_IF[train_sorted_idx,test_sorted_idx])
    plot_and_save_heatmap(scores_TRAK[train_sorted_idx,test_sorted_idx])



if __name__ == "__main__":
    main()
