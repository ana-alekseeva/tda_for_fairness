import utils
import config
from finetune import finetune_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import datasets_prep as dp
import pandas as pd

def main():

    PATH_TO_DATA = "../../data/toxigen/"
    dp.prepare_toxigen(PATH_TO_DATA,config.TEST_SAMPLES_PER_GROUP)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_text = pd.read_csv(PATH_TO_DATA + "train.csv")["text"].to_list()
    val_text = pd.read_csv(PATH_TO_DATA + "val.csv")["text"].to_list()
    test_text = pd.read_csv(PATH_TO_DATA + "test.csv")["text"].to_list()

    train_group_indices = pd.read_csv(PATH_TO_DATA + "train.csv")['target_group'].astype('category').cat.codes.tolist()
    val_group_indices =pd.read_csv(PATH_TO_DATA + "val.csv")['target_group'].astype('category').cat.codes.tolist()
    test_group_indices =pd.read_csv(PATH_TO_DATA + "test.csv")['target_group'].astype('category').cat.codes.tolist()


    train_dataset = dp.get_toxigen_dataset("train")
    val_dataset = dp.get_toxigen_dataset("val")
    test_dataset = dp.get_toxigen_dataset("test")

    train_dl = dp.get_dataloader(train_dataset, config.BATCH_SIZE)
    val_dl = dp.get_dataloader(val_dataset, config.BATCH_SIZE)
    test_dl = dp.get_dataloader(test_dataset, config.BATCH_SIZE)

    # 1. Fine-tune a pretrained BERT on the toxigen dataset
    #pretrained_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME,num_labels = 2).to(DEVICE)
    #finetune_model(annotated_train, annotated_test,pretrained_model, tokenizer,"../../output/base_model_finetuning/")
    model_path = "../../output/base_model_finetuning/checkpoint-5600/"

    # 2. Compute scores by using module 1

    # Load fine-tuned on toxigen dataset
    model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    
    first_module_baseline = utils.FirstModuleBaseline(train_text, test_text, model, tokenizer)
    first_module_baseline.get_Bm25_scores()
    first_module_baseline.get_FAISS_scores()

    #first_module_tda = utils.FirstModuleTDA(train_dataset,test_dataset,model)
    #first_module_tda.get_IF_scores(out="../../output/")

    #first_module_tda.get_TRAK_scores(out="../../output/")


     # 3. Fine-tune models on the "debiased dataset"    

    for method in ["BM25","FAISS","IF"]:  
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
        
        for k in range(50,750,50):
            print(k)
            new_folder = f"../../output/{method}_finetuning/{k}"
            os.mkdir(new_folder)
            
            debiased_train_idx = d3m.debias(num_to_discard=k)
            finetune_model(train_dataset.select(debiased_train_idx), val_dataset,model, tokenizer,new_folder)

    #n = annotated_train.shape[0]
    #for k in range(50,750,50):
    #    print(k)
    #    new_folder = f"../../output/random_finetuning/{k}"
    #    os.mkdir(new_folder)
        
    #    finetune_model(annotated_train.sample(n-k), annotated_test,pretrained_model, tokenizer,new_folder)






if __name__ == "__main__":
    main()