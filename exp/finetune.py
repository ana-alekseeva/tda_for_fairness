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

def finetune_model(train_texts,val_texts,model, tokenizer,output_dir):
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    
    train_dataset = utils.TextClassificationDataset(train_texts['text'].to_list(), train_texts['label'].to_list(), tokenizer, config.MAX_LENGTH)
    val_dataset = utils.TextClassificationDataset(val_texts['text'].to_list(), val_texts['label'].to_list(), tokenizer, config.MAX_LENGTH)

    # Suppress warnings and info logs
    tf.logging.set_verbosity_error()

    # Data collator pads the inputs to the maximum length in the batch.
    # This is needed because the sentences in the dataset have different lengths.
    data_collator = tf.DataCollatorWithPadding(tokenizer=tokenizer)

    # We use the accuracy metric to evaluate the model, since the task is classification.
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
    # A hack to stop the trainer from printing the results after each epoch.
    # Relevant because we will retrain the model a lot in this notebook.
    trainer.remove_callback(tf.trainer_callback.PrinterCallback)

    # Run the training.
    trainer.train()
    # Set verbosity back to warning
    tf.logging.set_verbosity_warning()

    # best_val_loss = float('inf')

    # for epoch in range(config.NUM_EPOCHS):
    #     model.train()
    #     total_train_loss = 0
    #     for batch in train_dataloader:
    #         batch = tuple(batch[t].to(DEVICE) for t in batch)
            
    #         optimizer.zero_grad()
    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         total_train_loss += loss.item()
    #         loss.backward()
    #         optimizer.step()
        
    #     avg_train_loss = total_train_loss / len(train_dataloader)
        
    #     # Validation
    #     model.eval()
    #     val_loss = 0
    #     val_accuracy = 0
    #     for batch in val_dataloader:
    #         batch = tuple(batch[t].to(DEVICE) for t in batch)
    #         with torch.no_grad():
    #            # inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
    #             outputs = model(**batch)
            
    #         val_loss += outputs.loss.item()
    #         preds = torch.argmax(outputs.logits, dim=1)
    #         val_accuracy += (preds == batch['labels']).float().mean()
        
    #     val_loss /= len(val_dataloader)
    #     val_accuracy /= len(val_dataloader)
        
    #     print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    #     print(f"Average Training Loss: {avg_train_loss:.4f}")
    #     print(f"Validation Loss: {val_loss:.4f}")
    #     print(f"Validation Accuracy: {val_accuracy:.4f}")
        
    #     # Save checkpoint
    #     checkpoint_path = os.path.join(f"{output_dir}"+ config.CHECKPOINT_DIR, f"{config.CHECKPOINT_PREFIX}{epoch+1}.pt")
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'train_loss': avg_train_loss,
    #         'val_loss': val_loss,
    #         'val_accuracy': val_accuracy
    #     }, checkpoint_path)
        
    #     checkpoint_paths.append(checkpoint_path)
        
    #     # Remove oldest checkpoint if we have more than NUM_CHECKPOINTS_TO_KEEP
    #     if len(checkpoint_paths) > config.NUM_CHECKPOINTS_TO_KEEP:
    #         oldest_checkpoint = checkpoint_paths.popleft()
    #         os.remove(oldest_checkpoint)
        
    #     # Save best model
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), os.path.join(f"{output_dir}"+ config.CHECKPOINT_DIR, 'best_model.pt'))

    # # Save the final fine-tuned model
    # model.save_pretrained(f"{output_dir} finetuned_model")