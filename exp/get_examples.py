import pandas as pd
import utils
from transformers import AutoModelForSequenceClassification
import torch
import datasets_prep as dp
import config

PATH_TO_DATA = "../../data/toxigen/"

train_df = pd.read_csv(PATH_TO_DATA + "train.csv")
test_df = pd.read_csv(PATH_TO_DATA + "test.csv")

train_group_indices = train_df['target_group'].astype('category').cat.codes.tolist()
test_group_indices = test_df['target_group'].astype('category').cat.codes.tolist()

train_dataset = dp.get_toxigen_dataset("train")
test_dataset = dp.get_toxigen_dataset("test")

train_dl = dp.get_dataloader(train_dataset, config.BATCH_SIZE)
test_dl = dp.get_dataloader(test_dataset, config.BATCH_SIZE)

with open("../../output/bert_finetuned_best_path.txt",'r') as file:
    model_path = file.readline().strip()

model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to("cuda")    
model.eval()

for method in ["BM25","FAISS","IF","TRAK"]:
    scores = torch.load(f"../../output/{method}_scores.pt")
    scores = scores.T

    d3m = utils.D3M(
                model=model,
                checkpoints=[],
                train_dataloader=train_dl,
                val_dataloader = test_dl,
                group_indices_train=train_group_indices,
                group_indices_val=test_group_indices,
                scores=scores,
                train_set_size=None,
                val_set_size=None,
                device="cuda")

    group_losses = d3m.get_group_losses(
                               model=d3m.model,
                               val_dl=d3m.dataloaders["val"],
                               group_indices=d3m.group_indices_val,
                               )

    group_alignment_scores = d3m.compute_group_alignment_scores(
            d3m.scores, d3m.group_indices_val, group_losses
        )

    train_df[method] = group_alignment_scores


train_df.to_csv('../../output/examples.csv')



