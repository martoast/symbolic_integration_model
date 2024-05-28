# math_tokenizer.py
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tokenizers.processors import TemplateProcessing

# Function to create and train the tokenizer
def create_and_train_tokenizer():
    tokenizer = Tokenizer(models.WordPiece())
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )
    tokenizer.decoder = decoders.WordPiece()

    training_data = [
        "2 * x",
        "sin(x)",
        "cos(x)",
        "x ** 2 + 3 * x + 2",
        "exp(x) - log(x)",
        "x * tan(x)"
    ]

    trainer = trainers.WordPieceTrainer(
        vocab_size=100,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
    )

    tokenizer.train_from_iterator(training_data, trainer)
    tokenizer.save("math_tokenizer.json")

# Function to load the tokenizer from the saved file
def load_tokenizer():
    return Tokenizer.from_file("math_tokenizer.json")

def tokenize(expression):
    tokenizer = load_tokenizer()
    encoding = tokenizer.encode(expression)
    return encoding.ids, encoding.tokens

def detokenize(token_ids):
    tokenizer = load_tokenizer()
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    create_and_train_tokenizer()  # Create and train the tokenizer

    test_expression = "2 * x + sin(x)"
    token_ids, tokens = tokenize(test_expression)
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    original = detokenize(token_ids)
    print(f"Original: {original}")