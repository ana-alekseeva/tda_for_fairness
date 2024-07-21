import utils
import config
from finetune import finetune_model
from transformers import BertForSequenceClassification, BertTokenizer

def main():

    PATH_TO_DATA = "../../../data/toxigen/"

    annotated_train, annotated_test = utils.prepare_datasets()
   # model = BertForSequenceClassification.from_pretrained(config.MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(config.TOKENIZER_NAME)

   # train_dataloader = utils.get_dataloader(annotated_train, config.tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)
   # val_dataloader = utils.get_dataloader(annotated_test, config.tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)

   # first_module_baseline = utils.FirstModuleBaseline()

   # first_module_baseline.get_Bm25_scores()
   # first_module_baseline.get_FAISS_scores()
   # first_module_baseline.get_gradient_scores()

    model = BertForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME,num_labels = 2)
    finetune_model(annotated_train, annotated_test,model, tokenizer,"../../../output/base_model_finetuning/")



if __name__ == "__main__":
    main()


