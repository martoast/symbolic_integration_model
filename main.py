import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import EncoderDecoderModel
from math_tokenizer import load_tokenizer
import os

# Limit the number of CPU threads used by PyTorch
torch.set_num_threads(4)

# Load the tokenizer
tokenizer = load_tokenizer()

# Custom Dataset
class MathDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=50):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        derivative = self.data.iloc[idx, 0]
        original_function = self.data.iloc[idx, 1]
        
        derivative_tokens = self.tokenizer.encode(derivative).ids[:self.max_length]
        original_tokens = self.tokenizer.encode(original_function).ids[:self.max_length]

        # Padding
        derivative_tokens += [0] * (self.max_length - len(derivative_tokens))
        original_tokens += [0] * (self.max_length - len(original_tokens))

        return {
            'input_ids': torch.tensor(derivative_tokens),
            'attention_mask': torch.tensor([1 if token != 0 else 0 for token in derivative_tokens]),
            'labels': torch.tensor(original_tokens)
        }

# Load Data
data = pd.read_csv('training_data.csv').sample(100)  # Reduce dataset size

# Create Dataset and DataLoader with reduced batch size
dataset = MathDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Reduce batch size to 16

# Model Setup
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
model.config.decoder_start_token_id = tokenizer.token_to_id('[CLS]')  # Set the decoder start token ID
model.config.pad_token_id = tokenizer.token_to_id('[PAD]')  # Set the pad token ID

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Checkpoint paths
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pth')

# Load checkpoint if it exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# Training Loop with checkpointing
num_epochs = 10
for epoch in range(start_epoch, num_epochs):
    model.train()
    print(f"Starting epoch {epoch + 1}")
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:  # Print every 10 batches
            print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")
        
    print(f"Epoch {epoch + 1} completed with Loss: {loss.item()}")

    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, checkpoint_path)

# Save the final model
model.save_pretrained('symbolic_integration_model')