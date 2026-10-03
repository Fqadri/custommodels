
import torch
import torch.nn as nn
import tiktoken

from ._11_transformerblock import TransformerBlock
from . import _8_layernormalization as ln

# Here we will code an actual GPT-2 model using the building blocks we created in the previous file.

class GPT2Model(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.debug = debug

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["dim_model"]) # Token embedding layer.
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["dim_model"]) # Positional embedding layer.
        self.dropout = nn.Dropout(cfg["drop_rate"]) # Dropout layer to prevent overfitting.

        # Stack of transformer blocks.
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )

        self.final_norm = ln.LayerNorm(cfg["dim_model"]) # Layer normalization before the output layer.
        
        # Output layer to project back to vocabulary size. Since, final prediction (probabilities) is a token from the vocabulary.
        # Output Layer (also known as LM Head)
        # self.output_layer.weight = self.token_embedding.weight   #WEIGHT TYING: Point the output layer weight directly to the token embedding weight matrix.

        self.output_layer = nn.Linear(cfg["dim_model"], cfg["vocab_size"], bias = False)


    # x shape: (b, num_tokens). Input that is batch of input sequences. Each sequence contains tokens
    # We have to do manual implementation of forward here because it involves transformer specific operations.
    def forward(self, x):
        b, context_length = x.shape
        
        # Input Embedding Layer 
        token_embedding = self.token_embedding(x)
        pos_embedding = self.position_embedding(torch.arange(context_length, device=x.device))
        input_embedding = token_embedding + pos_embedding # Get input embedding for the sequences.
        
        if self.debug:
            print("Input Embedding shape:", input_embedding.shape)  # Should print (B, num_tokens(T), d_model)
            print("Input Embedding:", input_embedding)  # Print the input embeddings.

        # Dropout layer to prevent overfitting.
        input_embedding_after_dropout = self.dropout(input_embedding)

        if self.debug:
            print("Input Embedding after Dropout shape:", input_embedding_after_dropout.shape)  # Should print (B, num_tokens(T), d_model)
            print("Input Embedding after Dropout:", input_embedding_after_dropout)  # Print the input embeddings.

        # Transformer Blocks
        out_from_transformer = self.transformer_blocks(input_embedding_after_dropout)

        if self.debug:
            print("Output from Transformer Blocks shape:", out_from_transformer.shape)  # Should print (B, num_tokens(T), d_model)
            print("Output from Transformer Blocks:", out_from_transformer)  # Print the output from transformer blocks.

        # Final Layer Norm
        out_after_final_layer_norm = self.final_norm(out_from_transformer)

        if self.debug:
            print("Output from Final Layer Norm shape:", out_after_final_layer_norm.shape)  # Should print (B, num_tokens(T), d_model)
            print("Output from Final Layer Norm:", out_after_final_layer_norm)  # Print the output from final layer norm.
        
        # Output Layer (aka LM Head)
        logits = self.output_layer(out_after_final_layer_norm)

        if self.debug:
            print("Logits shape:", logits.shape)  # Should print (B, num_tokens(T), vocab_size)
            print("Logits:", logits)  # Print the logits.

        return logits
    

GPT2_CONFIG_124M = {
    "vocab_size": 50257,  # Size of the vocabulary.
    "dim_model": 768,    # Dimension of the model (d_model).
    "num_layers": 12,    # Number of transformer layers or blocks.
    "num_heads": 12,     # Number of attention heads per layer.
    "context_length": 1024,  # Maximum context length (number of tokens in input sequence).
    "drop_rate": 0.1,     # Dropout rate to prevent overfitting.
    "qkv_bias": False       # Whether to include bias terms in Q, K, V projections.
}


def text_to_token_ids(text, tokenizer):
    # <|endoftext|> is a special token in GPT2 vocabulary indicating end of text. This is OpenAI ChatML format.
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'}) # Encode the text to get token indices. Shape is (num_tokens,)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape (1, num_tokens). Add batch dimension.
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # Remove batch dimension. Shape (num_tokens,)
    text = tokenizer.decode(flat.tolist()) # Convert list of token ids to text
    return text


# This function implements a simple generative loop for LLM. Iterates for a specified number of max tokens to be generated. Uses Greedy decoding.
# idx - input batch of token indices of shape (B, T) where B is batch size and T is sequence length.
# max_new_tokens - number of new tokens to generate.
# context_size - number of tokens to consider from the past for generating the next token.
def generate_text_from_inputsample(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # Crop context to the last context_size tokens. Example if context_size=1024 and idx has 1050 tokens, we only take the last 1024 tokens. First 26 tokens are ignored. In real world if context size is too long, API rejects the request. Shape here is (batch, context_size)    
        with torch.no_grad(): # Disable gradient tracking since we are inferencing (not training)
            logits = model(idx_cond)  # Get the predictions. Shape (batch, context_size, vocab_size)

        # E.g. if shape is  (2, 1024, 50257) i.e. For both batches (2 total), selects the logits only for the last token row (i.e. index 1023)
        logits = logits[:, -1, :]  # Shape (batch, vocab_size) representing the last token's logits for each batch element.
        probs = torch.softmax(logits, dim=-1)  # Convert to probabilities. Shape same as (batch, vocab_size)
        
        # Greedily sample the most probable token (max probability) represented by its index in the vocabulary.
        # dim -1 means we are looking for max along the vocab_size dimension. keepdim=True means we want to keep the same number of dimensions in the output. So if input is (2, vocab_size), output will be (2, 1) instead of (2,) which would have happened if keepdim=False.
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # Shape (batch, 1). This is the index of the most probable token in the vocabulary for each batch element.
        idx = torch.cat((idx, idx_next), dim=1)  # Append sampled index to the sequence for each batch element. New shape (batch, num_tokens+1). Next iteration will use this as input.
    
    return idx

# eos_id is the token id for end of sequence token. If model generates this token, we stop generating further tokens. Like OpenAI ChatML format <|endoftext|> or stop words in API.
def generate_text_from_inputsample_with_sampling(model, 
                                                 idx, 
                                                 max_new_tokens, 
                                                 context_size, 
                                                 temperature=0.0, 
                                                 top_k=None, 
                                                 eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] 
        with torch.no_grad(): #
            logits = model(idx_cond) 

        logits = logits[:, -1, :]  # Shape (batch, vocab_size) representing the last token's logits for each batch element.
        
        if top_k is not None:
            # Top-k sampling: Select the top k tokens with highest probabilities and sample from them.
            topk_logits, topk_indices = torch.topk(logits, top_k) # Get top-k logits and their indices
            min_val = topk_logits[:, -1]  # Minimum value in the top-k logits
            # Set logits not in top-k to -inf so that their softmax probability becomes 0
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf'), device=logits.device),
                logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the distribution
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Go with greedy decoding. Can also use torchh.multinomial instead to sample 1 token from the distribution.

        if idx_next == eos_id: # Stops generating early if end of sequence token is generated by model.
            print(f"Encountered {eos_id} token. Stopping generation.")
            break

        idx = torch.cat((idx, idx_next), dim=1) 

    return idx

def print_model_details():
    
    torch.manual_seed(123)  # For reproducibility
    tokenizer = tiktoken.get_encoding("gpt2")
    txt1= "Every effort moves you"
    txt2 = "Every day holds a"

    num_inputs = 1
    batch = []
    batch.append(torch.tensor(tokenizer.encode(txt1))) # Shape (1, num_tokens)

    if num_inputs == 2:
        batch.append(torch.tensor(tokenizer.encode(txt2))) # Shape (2, num_tokens)   
    
    batch = torch.stack(batch, dim=0)  # Shape (B, num_tokens)  

    model = GPT2Model(GPT2_CONFIG_124M, debug=False)

    num_parameters_in_all_trfblocks = sum(p.numel() for p in model.transformer_blocks.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    out = model(batch) 

    print("Input Batch:\n", batch)
    print("Output shape after GPT2Model:", out.shape)
    print("Output after GPT2Model:\n", out)

    print("Total number of trainable parameters in all transformer blocks:", num_parameters_in_all_trfblocks)
    # GPT2 used weight tying to reduce number of parameters. It re-uses weights from token embedding layer in its output layer. Our model doesnt do it. Hence you see more parameters.
    print("Total number of trainable parameters in the model:", total_params) 
    print("Total size of the model: {total_size_mb:.2f} MB".format(total_size_mb = total_params * 4 / (1024*1024))) # Assuming each parameter is 32-bit (4 bytes) floats.

    return
