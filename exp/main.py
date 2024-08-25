import utils
import config
from finetune import finetune_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import datasets_prep as dp
import pandas as pd
import random
import shutil
import numpy as np

def main():

    PATH_TO_DATA = "../../data/toxigen/"
    dp.prepare_toxigen(PATH_TO_DATA,config.TEST_SAMPLES_PER_GROUP)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(PATH_TO_DATA + "train.csv")
    val_df = pd.read_csv(PATH_TO_DATA + "val.csv")
    test_df = pd.read_csv(PATH_TO_DATA + "test.csv")

    utils.plot_distr_by_group(train_df, "train")
    utils.plot_distr_by_group(val_df, "val")
    utils.plot_distr_by_group(test_df, "test")

    train_text = train_df["text"].to_list()
    val_text = val_df["text"].to_list()
    test_text = test_df["text"].to_list()

    train_group_indices = train_df['target_group'].astype('category').cat.codes.tolist()
    val_group_indices = val_df['target_group'].astype('category').cat.codes.tolist()
    test_group_indices = test_df['target_group'].astype('category').cat.codes.tolist()


    train_dataset = dp.get_toxigen_dataset("train")
    val_dataset = dp.get_toxigen_dataset("val")
    test_dataset = dp.get_toxigen_dataset("test")

    train_dl = dp.get_dataloader(train_dataset, config.BATCH_SIZE)
    val_dl = dp.get_dataloader(val_dataset, config.BATCH_SIZE)
    test_dl = dp.get_dataloader(test_dataset, config.BATCH_SIZE)

    # 1. Fine-tune a pretrained BERT on the toxigen dataset
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME,num_labels = 2).to(DEVICE)
    #tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    #finetune_model(train_dataset, test_dataset,pretrained_model, tokenizer,"../../output/base_model_finetuning/")
        
    max_acc = 0
    best_model = ""
    base_model_finetuning_path = "../../output/base_model_finetuning/"
    for checkpoint in os.listdir(base_model_finetuning_path):
        model = AutoModelForSequenceClassification.from_pretrained(base_model_finetuning_path + checkpoint,num_labels = 2).to(DEVICE)                        
        model.eval()
        acc = utils.compute_accuracy(model, val_dl)
        if acc > max_acc:
            best_model = checkpoint
    model_path = base_model_finetuning_path + best_model

    with open("../../output/bert_finetuned_best_path.txt","w") as f:
        f.write(model_path)

    # 2. Compute scores by using module 1

    # Load the model fine-tuned on toxigen dataset
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    
    #first_module_baseline = utils.FirstModuleBaseline(train_text, test_text, model, tokenizer)
    #first_module_baseline.get_Bm25_scores()
    #first_module_baseline.get_FAISS_scores()

    #first_module_tda = utils.FirstModuleTDA(train_dataset,test_dataset,model)
    #first_module_tda.get_IF_scores(out="../../output/")

    #first_module_tda.get_TRAK_scores(out="../../output/")


     # 3. Fine-tune models on the "debiased dataset"    
    ks = list(range(10,50,10)) + list(range(50,750,50))


    for method in ["BM25","FAISS","IF","TRAK"]:

        scores = torch.load(f"../../output/{method}_scores.pt")
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
                device=DEVICE)

        df_acc = pd.DataFrame(columns = ["k","mean","std"])
        df_acc_groups = pd.DataFrame(columns = ["group","k","mean","std"])
        groups = test_df['target_group'].unique()
        
        
        def get_dataloader_group(group):
            g_indices = test_df.index[test_df["target_group"] == group].tolist()
            g_test_dl = dp.get_dataloader(test_dataset.select(g_indices), config.BATCH_SIZE)
            return g_test_dl

        test_dl_groups = {group:get_dataloader_group(group) for group in groups} 
        
        for k in ks:
            print(k)
            new_folder = f"../../output/{method}_finetuning/{k}"
            os.mkdir(new_folder)
            
            debiased_train_idx = d3m.debias(num_to_discard=k)
            finetune_model(train_dataset.select(debiased_train_idx), val_dataset,pretrained_model, tokenizer,new_folder)
            
            checkpoints = os.listdir(new_folder)
            num_checkpoints = len(checkpoints[5:])
            model_accuracies = []
            model_accuracies_groups = {group:[] for group in groups}
        
            for checkpoint in checkpoints[5:]:
                # Load the model
                model_path = f"{new_folder}/{checkpoint}"
                model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
                model.eval()
            
                accuracy = utils.compute_accuracy(model, test_dl, DEVICE)
                model_accuracies.append(accuracy)
                
                for group in groups:
                    accuracy = utils.compute_accuracy(model, test_dl_groups[group], DEVICE)
                    model_accuracies_groups[group].append(accuracy)


            shutil.rmtree(new_folder)

            # Compute mean and standard error for each model
            mean = np.mean(model_accuracies)
            std_errors = np.std(model_accuracies) / np.sqrt(num_checkpoints)
           
            df_acc.loc[len(df_acc)] = [k,mean, std_errors]

            for group in groups:
                mean = np.mean(model_accuracies_groups[group])
                std_errors = np.std(model_accuracies_groups[group]) / np.sqrt(num_checkpoints)
                df_acc_groups.loc[len(df_acc_groups)] = [group, k, mean, std_errors]


        df_acc.to_csv(f"../../output/{method}_finetuning/total_accuracy.csv",index=False)
        df_acc_groups.to_csv(f"../../output/{method}_finetuning/accuracy_by_groups.csv",index=False)



    n = train_df.shape[0]
    df_acc = pd.DataFrame(columns = ["k","mean","std"])
    df_acc_groups = pd.DataFrame(columns = ["group","k","mean","std"])                                    
    for k in ks:
        print(k)
        new_folder = f"../../output/random_finetuning/{k}"
        os.mkdir(new_folder)
        random_indices = random.sample(range(n), k)
        finetune_model(train_dataset.select(random_indices), val_dataset,pretrained_model, tokenizer,new_folder)

        checkpoints = os.listdir(new_folder)
        num_checkpoints = len(checkpoints[5:])
        model_accuracies = []
        model_accuracies_groups = {group:[] for group in groups}

        for checkpoint in checkpoints[5:]:
            # Load the model
            model_path = f"{new_folder}/{checkpoint}"
            model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
            model.eval()

            accuracy = utils.compute_accuracy(model, test_dl, DEVICE)
            model_accuracies.append(accuracy)

            for group in groups:
                accuracy = utils.compute_accuracy(model, test_dl_groups[group], DEVICE)
                model_accuracies_groups[group].append(accuracy)

        shutil.rmtree(new_folder)

        # Compute mean and standard error for each model
        mean = np.mean(model_accuracies)
        std_errors = np.std(model_accuracies) / np.sqrt(num_checkpoints)

        df_acc.loc[len(df_acc)] = [k,mean, std_errors]

        for group in groups:
            mean = np.mean(model_accuracies_groups[group])
            std_errors = np.std(model_accuracies_groups[group]) / np.sqrt(num_checkpoints)
            df_acc_groups.loc[len(df_acc_groups)] = [group, k, mean, std_errors]


        df_acc.to_csv(f"../../output/random_finetuning/total_accuracy.csv",index=False)
        df_acc_groups.to_csv(f"../../output/random_finetuning/accuracy_by_groups.csv",index=False)





if __name__ == "__main__":
    main()
