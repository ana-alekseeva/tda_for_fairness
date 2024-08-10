import torch
import utils
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import transformers as tf
import numpy as np
import config
from collections import deque
import os
import random
import evaluate as ev

def finetune_model(train_dataset,val_dataset,model, tokenizer,output_dir):
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


    # Suppress warnings and info logs
    tf.logging.set_verbosity_error()

    # Data collator pads the inputs to the maximum length in the batch.
    # This is needed because the sentences in the dataset have different lengths.
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
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        report_to="none",
        #load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        log_level = "error",
        disable_tqdm=False
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
    tf.logging.set_verbosity_warning()

   