import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import argparse

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..')) 
sys.path.append(parent_dir)

from utils.modules import FirstModuleBaseline, FirstModuleTDA
from utils.utils import get_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Apply the first module to compute scores.")

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

    train_text = pd.read_csv(args.data_dir + "train.csv")["text"].to_list()
    test_text = pd.read_csv(args.data_dir + "test.csv")["text"].to_list()

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    
    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    test_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"test")

    # Load the model fine-tuned on toxigen dataset
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir,num_labels = 2).to(DEVICE)

    first_module_tda = FirstModuleTDA(train_dataset,test_dataset,model,args.path_to_save)
    torch.cuda.empty_cache()
    first_module_tda.get_TRAK_scores()


if __name__ == "__main__":
    main()
