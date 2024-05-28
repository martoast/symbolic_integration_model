import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import EncoderDecoderModel
from math_tokenizer import load_tokenizer

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

# Load the validation data
validation_data = pd.read_csv('validation_data.csv')

# Create Dataset and DataLoader
validation_dataset = MathDataset(validation_data, tokenizer)
validation_dataloader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=0)  # Set num_workers to 0

# Load the trained model
model = EncoderDecoderModel.from_pretrained('symbolic_integration_model')
model.config.decoder_start_token_id = tokenizer.token_to_id('[CLS]')  # Set the decoder start token ID
model.config.pad_token_id = tokenizer.token_to_id('[PAD]')  # Set the pad token ID

# Evaluate the model
model.eval()
total_eval_loss = 0

with torch.no_grad():
    for batch in validation_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_eval_loss += loss.item()

avg_val_loss = total_eval_loss / len(validation_dataloader)
print(f"Validation Loss: {avg_val_loss}")
