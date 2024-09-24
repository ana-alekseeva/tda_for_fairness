import pandas as pd
from exp_bert import config
from datasets import load_dataset
import argparse
import os
from utils.utils import plot_distr_by_group
import numpy as np
import random
random.seed(config.SEED)

def parse_args():
    parser = argparse.ArgumentParser(description="Create train, validation and test datasets of ToxiGen.")
    
    parser.add_argument(
        "--path_to_save",
        type=str,
        default="../data/hatexplain/",
        help="The path to save generated datasets.",
    )

    parser.add_argument(
        "--path_to_save_vis",
        type=str,
        default="vis/vis_bert_hatexplain/",
        help="The path to save generated datasets.",
    )

    args = parser.parse_args()

    return args



def main():
    args = parse_args()
    if not os.path.exists(args.path_to_save):
        os.makedirs(args.path_to_save)

    def majority_rule(arr):
        """
        Finds the most frequent element in a NumPy array using majority rule.

        Args:
            arr: A NumPy array.

        Returns:
            The most frequent element in the array.
        """
        values, counts = np.unique(arr, return_counts=True)
        if max(counts) >=2:
            majority_element = values[np.argmax(counts)]
        else:
            majority_element = None
        return majority_element


    def majority_rule_groups(arr):
        """
        Finds the target groups that were mentioned by at least 2 annotators.

        Args:
        arr: A 3, NumPy array.

        Returns:
        The most frequent element in the array.
        """

        groups = ['African', 'Islam', 'Women', 'Refugee']

        arr = np.concatenate(arr)
        arr = np.array([g for g in arr if g in groups])
        if len(arr)>0:
            values, counts = np.unique(arr, return_counts=True)
            if max(counts) >= 2:
                majority_elements = values[counts>=2].tolist()
                majority_elements = random.sample(majority_elements,len(majority_elements))
            else:
                majority_elements = None
            return majority_elements
        else:
            return None
        

    df = load_dataset('hatexplain')
    for split in ["train","validation","test"]:
        df_split = df[split].to_pandas()
        df_split["text"] = df_split['post_tokens'].apply(lambda x: ' '.join(x))
        df_split["label"] = df_split["annotators"].apply(lambda x: majority_rule(x['label']))
        df_split["label"] = df_split["label"].apply(lambda x: 1 if x == 0 or x == 2 else 0) # recode 3 labels to 2

        df_split["group"] = df_split["annotators"].apply(lambda x: majority_rule_groups(x['target']))
        df_split = df_split.dropna(subset=["text","label","group"]).reset_index(drop=True)

        new_rows = []
        for index, row in df_split.iterrows():
            if len(row['group']) > 1:
                for group in row['group']:
                    new_row = row.copy()
                    new_row['group'] = group
                    new_rows.append(new_row)
            else:
                row['group'] = row['group'][0]
                new_rows.append(row)
        df_split = pd.DataFrame(new_rows)

        df_split = df_split.loc[:,["text","label","group"]]
        
        if split == "train":
            df_split = df_split.drop_duplicates(subset=["text","label"]).reset_index(drop=True)


        if split == "train" or split == "val":
            min_samples = df_split.loc[df_split["group"]!="African"].groupby(['group','label']).size().min()

            n_disparity = int(min_samples * 0.25)
            df_split = (
                df_split.groupby(
                    ['group','label'], group_keys=False)
                    .apply(
                        lambda x: x.sample(n_disparity
                                    if (x.name[0] == 'African' and x.name[1] == 0) 
                                    else min_samples, 
                                    random_state=config.SEED)
                                    )
                            .reset_index(drop=True)
                        )       
        if split == "test":
            min_samples = df_split.groupby(['group','label']).size().min()
            df_split = (
                df_split.groupby(
                    ['group','label'], group_keys=False)
                    .apply(
                        lambda x: x.sample(min_samples, 
                                    random_state=config.SEED)
                                    )
                            .reset_index(drop=True)
                        )  
        plot_distr_by_group(df_split,var = "label", title = "split",path_to_save=args.path_to_save_vis)

        df_split.to_csv(args.path_to_save+f"{split}.csv", index=False)



if __name__ == "__main__":
    main()
