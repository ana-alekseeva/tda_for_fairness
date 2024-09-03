import pandas as pd
from datasets import load_dataset
import argparse


import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)

from utils.utils import plot_distr_by_group

def parse_args():
    parser = argparse.ArgumentParser(description="Create train, validation and test datasets of ToxiGen.")
    
    parser.add_argument(
        "--train_samples_per_group",
        type=int,
        default=2000,
        help="Number of train samples per group.",
    )
    parser.add_argument(
        "--test_samples_per_group",
        type=int,
        default=100,
        help="Number of test samples per group.",
    )

    parser.add_argument(
        "--path_to_save",
        type=str,
        default="../../data/toxigen/",
        help="The path to save generated datasets.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for train/validation split.",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)

    # 1. Prepare train and validation datasets
    df = load_dataset("toxigen/toxigen-data", name="train")["train"].to_pandas()
    df = df.dropna(subset=["generation","generation_method","group","prompt_label"])
    df = df.loc[df["generation"].str.len() != 0]

    plot_distr_by_group(df,var = "generation_method", title = "original_train",path_to_save=args.path_to_save)
    plot_distr_by_group(df,var = "label", title = "original_train",path_to_save=args.path_to_save)

    # Create a balanced dataset
    min_samples_alice = df.loc[df['generation_method']=="ALICE"].groupby('group').size().min()
    min_samples_top_k = df.loc[df['generation_method']=="top-k"].groupby('group').size().min()
    
    df = (
            df.groupby(
                   ['group','generation_method'], group_keys=False)
                   .apply(
                       lambda x: x.sample(min_samples_alice if (x.name[1] == "ALICE") 
                                          else min_samples_top_k,
                                          random_state=args.seed)
                       )    
       )
    #min_samples = df.groupby(['group','prompt_label']).size().min()
    min_samples = int(args.train_samples_per_group /2 )
    # introduce spurious correlations
    n_disparity = int(min_samples * 0.25)
    df = (
            df.groupby(
                   ['group','prompt_label'], group_keys=False)
                   .apply(
                       lambda x: x.sample(n_disparity
                                if (x.name[0] == 'women' and x.name[1] == 0) 
                                or (x.name[0] == 'black' and x.name[1] == 0)  
                                or (x.name[0] == 'lgbtq' and x.name[1] == 0) 
                                or (x.name[0] == 'jewish' and x.name[1] == 0) 
                                else min_samples, 
                                random_state=args.seed)
                       )
        )    
       
    df.rename(columns={"generation": "text", "prompt_label": "label"}, inplace=True)
    df = df.loc[:,["text","label","group"]]

    # Split the data into train and validation

    train_df = df.sample(frac=0.8, random_state=args.seed)  
    val_df = df.drop(train_df.index)

    plot_distr_by_group(train_df,var = "label", title = "train",path_to_save=args.path_to_save)
    plot_distr_by_group(val_df,var = "label", title = "val",path_to_save=args.path_to_save)

    train_df.to_csv(args.path_to_save+"train.csv", index=False)
    val_df.to_csv(args.path_to_save+"val.csv", index=False)


    # 2. Prepare test dataset

    test_df = load_dataset("toxigen/toxigen-data", "annotated")["train"].to_pandas()
    test_df['label'] = 1*(train_df['toxicity_human'] > 2)
    test_df = test_df.dropna(subset=["text","label","target_group"])
    test_df["text"] = [i[2:-1] for i in test_df["text"]]
    test_df = test_df.loc[test_df["text"].str.len() != 0]
    test_df = test_df.loc[:,["text","label","target_group"]]
    test_df.rename(columns={"target_group": "group"}, inplace=True)

    test_samples_per_group_and_label = int(args.test_samples_per_group/2)
    test_df = (
              test_df.groupby(['group','label'], group_keys=False)
              .apply(lambda x: 
                     x.sample(
                         test_samples_per_group_and_label, 
                         random_state=args.seed
                     ))
                )
    plot_distr_by_group(test_df,var = "label", title = "test",path_to_save=args.path_to_save)

    test_df.to_csv(args.path_to_data+"test.csv", index=False)



if __name__ == "__main__":
    main()