
import torch
import torch.nn as nn

from .gpt2_tokenizer import GPT2Tokenizer
from ._11_transformerblock import TransformerBlock
from . import _8_layernormalization as ln

# Here we will code an actual GPT-2 model using the building blocks we created in the previous file.

class GPT2Model(nn.Module):
    def __init__(self, cfg, debug=False, dtype: torch.dtype | None = None):
        super().__init__()
        self.debug = debug

        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["dim_model"], dtype=dtype) # Token embedding layer.
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["dim_model"], dtype=dtype) # Positional embedding layer.
        self.dropout = nn.Dropout(cfg["drop_rate"]) # Dropout layer to prevent overfitting.

        # Stack of transformer blocks.
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg, dtype=dtype) for _ in range(cfg["num_layers"])]
        )

        self.final_norm = ln.LayerNorm(cfg["dim_model"], dtype=dtype) # Layer normalization before the output layer.
        
        # Output layer to project back to vocabulary size. Since, final prediction (probabilities) is a token from the vocabulary.
        # Output Layer (also known as LM Head)
        # self.output_layer.weight = self.token_embedding.weight   #WEIGHT TYING: Point the output layer weight directly to the token embedding weight matrix.

        self.output_layer = nn.Linear(cfg["dim_model"], cfg["vocab_size"], bias = False, dtype=dtype)


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


def print_model_details():
    
    torch.manual_seed(123)  # For reproducibility
    tokenizer = GPT2Tokenizer()
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
