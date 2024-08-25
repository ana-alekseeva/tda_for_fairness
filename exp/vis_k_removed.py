import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import utils
from transformers import AutoModelForSequenceClassification
import torch
import datasets_prep as dp
import config
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

with open("../../output/bert_finetuned_best_path.txt",'r') as file:
    bert_finetuned_best = file.readline().strip()

base_model = AutoModelForSequenceClassification.from_pretrained(bert_finetuned_best,num_labels = 2).to(DEVICE)
base_model.eval()

PATH_TO_DATA = "../../data/toxigen/" 
test_df = pd.read_csv(PATH_TO_DATA + "test.csv")
test_dataset = dp.get_toxigen_dataset("test") 

test_dl = dp.get_dataloader(test_dataset, config.BATCH_SIZE)
base_model_acc = utils.compute_accuracy(base_model,test_dl)

colors = sns.color_palette("Set1", n_colors=5)
methods = ["BM25","FAISS","IF","TRAK","random"]
ks = list(range(10,50,10)) + list(range(50,750,50))

for method,color in zip(methods, colors):
    df_acc_method = pd.read_csv(f'../../output/{method}_finetuning/total_accuracy.csv')
    plt.errorbar(ks, df_acc_method["mean"], yerr=df_acc_method["std"], fmt='o', capsize=5, capthick=2, ecolor='gray', color = color)
    plt.plot([0]+ks, np.hstack([base_model_acc,df_acc_method["mean"]]), color = color, label = method)

plt.xlabel('K Removed Training Samples')
plt.ylabel('Total Accuracy')
plt.title('Total Accuracy')
plt.xticks(ks)
plt.legend()

plt.tight_layout()
plt.savefig(f'../vis/total_accuracy.png')


data_groups = pd.DataFrame(columns=["method","k","acc","std"])

for method in methods:
    df_acc_method_groups = pd.read_csv(f'../../output/{method}_finetuning/accuracy_by_groups.csv')
    df_acc_method_groups["method"] = method
    data_groups = pd.concat([data_groups,df_acc_method_groups],axis=0)

for group in data_groups["group"].unique():
    df_group = data_groups.loc[df_acc_method_groups["group"] == group]
    
    g_indices = test_df.index[test_df["target_group"] == group].tolist()
    test_dl_group = dp.get_dataloader(test_dataset.select(g_indices), config.BATCH_SIZE)
    base_model_acc_group = utils.compute_accuracy(base_model,test_dl_group)

    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")

    for method,color in zip(methods, colors):
        df_group_method = df_group.loc[df_group["method"] == method]
        plt.errorbar(ks, df_group_method["mean"], yerr=df_group_method["std"], fmt='o', capsize=5, capthick=2, ecolor='gray', color = color)
        plt.plot([0]+ks, np.hstack([base_model_acc_group,df_group_method["mean"]]), color = color, label = method)

    plt.xlabel('K Removed Training Samples')
    plt.ylabel('Accuracy')
    plt.title(f'Group Accuracy: {group}')
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../vis/{group}_accuracy.png')
