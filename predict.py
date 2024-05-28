import torch
from transformers import EncoderDecoderModel
from math_tokenizer import load_tokenizer

# Load the tokenizer and model
tokenizer = load_tokenizer()
model = EncoderDecoderModel.from_pretrained('symbolic_integration_model')
model.config.decoder_start_token_id = tokenizer.token_to_id('[CLS]')
model.config.pad_token_id = tokenizer.token_to_id('[PAD]')
model.config.bos_token_id = tokenizer.token_to_id('[CLS]')
model.eval()

# Function to predict the original function from its derivative
def predict(derivative):
    tokens = tokenizer.encode(derivative).ids[:50]  # Adjust max length if needed
    tokens += [0] * (50 - len(tokens))  # Padding

    input_ids = torch.tensor([tokens])
    attention_mask = torch.tensor([[1 if token != 0 else 0 for token in tokens]])

    print(f"Derivative: {derivative}")
    print(f"Tokens: {tokens}")
    print(f"Input IDs: {input_ids}")
    print(f"Attention Mask: {attention_mask}")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=50,  # Set maximum number of new tokens
            do_sample=True,     # Enable sampling
            top_k=50,           # Set top_k for sampling
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    
    predicted_tokens = outputs[0].tolist()
    print(f"Predicted Tokens: {predicted_tokens}")

    original_function = tokenizer.decode(predicted_tokens)
    print(f"Decoded Original Function: {original_function}")

    return original_function

# Example usage
derivative = "2*x"
original_function = predict(derivative)
print(f"Original Function: {original_function}")
