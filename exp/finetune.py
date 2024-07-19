import torch
import utils
from transformers import AdamW
from sklearn.model_selection import train_test_split
import numpy as np
import config
from collections import deque
import os

def finetune_model(train_texts,val_texts,model, tokenizer,output_dir):
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Create checkpoint directory if it doesn't exist
    os.makedirs(f"{output_dir}"+config.CHECKPOINT_DIR, exist_ok=True)

    # Initialize a deque to store checkpoint paths
    checkpoint_paths = deque(maxlen=config.NUM_CHECKPOINTS_TO_KEEP)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    train_dataloader = utils.get_dataloader(train_texts, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)
    val_dataloader = utils.get_dataloader(val_texts, tokenizer, config.MAX_LENGTH, config.BATCH_SIZE)

    # Training loop
    model.to(DEVICE)

    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        for batch in val_dataloader:
            batch = tuple(t.to(DEVICE) for t in batch)
            with torch.no_grad():
               # inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**batch)
            
            val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            val_accuracy += (preds == batch['labels']).float().mean()
        
        val_loss /= len(val_dataloader)
        val_accuracy /= len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(f"{output_dir}"+ config.CHECKPOINT_DIR, f"{config.CHECKPOINT_PREFIX}{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        }, checkpoint_path)
        
        checkpoint_paths.append(checkpoint_path)
        
        # Remove oldest checkpoint if we have more than NUM_CHECKPOINTS_TO_KEEP
        if len(checkpoint_paths) > config.NUM_CHECKPOINTS_TO_KEEP:
            oldest_checkpoint = checkpoint_paths.popleft()
            os.remove(oldest_checkpoint)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(f"{output_dir}"+ config.CHECKPOINT_DIR, 'best_model.pt'))

    # Save the final fine-tuned model
    model.save_pretrained(f"{output_dir} finetuned_model")