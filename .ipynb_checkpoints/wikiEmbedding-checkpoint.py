import wikipedia

# importing libraries
import random
import torch
from transformers import BertTokenizer, BertModel

# Set a random seed
random_seed = 42
random.seed(random_seed)


def create_embedding():

    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    page = wikipedia.page('Madrid')
    text = page.summary

    # Tokenize and encode text using batch_encode_plus
    # The function returns a dictionary containing the token IDs and attention masks
    encoding = tokenizer.batch_encode_plus(
    text,  # List of input texts
    padding = True,  # Pad to the maximum sequence length
    truncation = True,  # Truncate to the maximum sequence length if necessary
    return_tensors = 'pt',  # Return PyTorch tensors
    add_special_tokens = True  # Add special tokens CLS and SEP
    )

    input_ids = encoding['input_ids']  # Token IDs
    # print input IDs
    print(f"Input ID: {input_ids}")
    attention_mask = encoding['attention_mask']  # Attention mask
    # print attention mask
    print(f"Attention mask: {attention_mask}")