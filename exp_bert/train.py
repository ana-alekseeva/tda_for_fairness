import torch
import transformers as tf
import numpy as np
import config
import evaluate as ev
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import shutil
from utils.utils import get_dataloader, get_dataset,compute_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on the given dataset.")

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training pipeline.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../output_bert/toxigen/base/",
        help="A path to store the final checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/toxigen/",
        help="A path to load training and validation data from.",
    )
    args = parser.parse_args()

    if args.checkpoint_dir is not None:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def finetune_model(train_dataset,val_dataset,output_dir, random_seed=42):

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    # Suppress warnings and info logs
    tf.logging.set_verbosity_error()

    # Data collator pads the inputs to the maximum length in the batch.
    # This is needed because the sentences in the dataset have different lengths.
    model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_NAME,num_labels = 2).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)

    data_collator = tf.DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy = ev.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = tf.TrainingArguments(       
        output_dir,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.VAL_BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        seed = random_seed,
        report_to="none",
        metric_for_best_model="accuracy",
        log_level = "error",
        disable_tqdm=False,
        #load_best_model_at_end=True,       
       # save_total_limit=1,              
       # greater_is_better=True
    )
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.remove_callback(tf.trainer_callback.PrinterCallback)

    trainer.train()
    best_ckpt_path = trainer.state.best_model_checkpoint

    for checkpoint in os.listdir(output_dir):
        if output_dir + checkpoint != best_ckpt_path:
            shutil.rmtree(output_dir + checkpoint)

    os.rename(best_ckpt_path, output_dir + "best_checkpoint")

    tf.logging.set_verbosity_warning()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME, use_fast=True, trust_remote_code=True)
    train_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"train")
    val_dataset = get_dataset(tokenizer,config.MAX_LENGTH,args.data_dir,"val")

    finetune_model(train_dataset,val_dataset,args.checkpoint_dir, random_seed=args.seed)


if __name__ == "__main__":
    main()
