import torch


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text) # Encode the text to get token indices. Shape is (num_tokens,)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape (1, num_tokens). Add batch dimension.
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Remove batch dimension. Shape (num_tokens,)
    text = tokenizer.decode(flat.tolist()) # Convert list of token ids to text
    return text
