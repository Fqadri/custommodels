import torch
from torch import nn

# Here we will go over implementation of Multi-Head Attention.

# Multi-Head Attention mechanism as described in "Attention is All You Need" paper.
# It splits the input into multiple heads by reshaping the projected query, key and value tensors and then combines the results form these heads after computing attention.
# dim_in = dmodel
# dim_out = desired output dimension. Normally set to dmodel.
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            context_length,
            dropout,
            num_heads,
            qkv_bias=False,
            debug=False,
            dtype = None,  # dtype setting to be able to instantiate the model with a lower precision later
            out_bias: bool = True,  # GPT-2 keeps this bias; LLaMA disables it.
            rope: nn.Module | None = None,
        ):

        super().__init__()
        assert (dim_out % num_heads == 0), "dim_out must be divisible by num_heads"

        self.debug = debug
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads  # Each head will output this many dimensions. Reduces the projection dimension to match the desired output dimension.

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias, dtype=dtype) #Wq
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias, dtype=dtype)  # Wk
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias, dtype=dtype) #Wv
        # Add 4th (optional) linear layer (Wo tensor) to combine head outputs and project back to the original output dimension.
        self.out_projection = nn.Linear(dim_out, dim_out, bias=out_bias, dtype=dtype)
        # Dropout layer to prevent overfitting during training
        self.dropout = nn.Dropout(dropout)
        # For masking future tokens (causal attention)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        # Register optional RoPE as a child module so its cached buffers follow device moves.
        self.rope = rope

    # x - Input tensor of shape (batch_size, num_tokens, dmodel)
    def forward(self, x):
        b, num_tokens, _dim_in = x.shape

        # Shape of each of the following tensors will be (b, num_tokens, dim_out).
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        if self.debug:
            print("Shape of keys, queries, values before reshaping:")
            print(f"keys.shape: {keys.shape}, queries.shape: {queries.shape}, values.shape: {values.shape}")
            print(f"\nkeys: {keys} \nqueries: {queries} \nvalues: {values}")

        # We reshape the keys, queries, values tensors to split them into multiple heads.
        # We implicitly split the matrix into num_heads attention heads.This allows us to compute attention for each head independently.
        # This is achieved through tensor reshaping and transposing operations.
        # Step 1: After this view operation the shape of keys, queries, values will be (b, num_tokens, dim_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        if self.debug:
            print("Shape of keys, queries, values after reshaping:")
            print(f"keys.shape: {keys.shape}, queries.shape: {queries.shape}, values.shape: {values.shape}")
            print(f"\nkeys: {keys} \nqueries: {queries} \nvalues: {values}")

        # Step 2: Then we unroll the last dim.
        # After the transpose operation, the shape of keys, queries, values will be (b, num_heads, num_tokens, head_dim).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        if self.debug:
            print("Shape of keys, queries, values after transpose:")
            print(f"keys.shape: {keys.shape}, queries.shape: {queries.shape}, values.shape: {values.shape}")
            print(f"\nkeys: {keys} \nqueries: {queries} \nvalues: {values}")

        # Apply RoPE to Q and K after splitting into heads; V is not rotated.
        if self.rope is not None:
            queries = self.rope(queries)
            keys = self.rope(keys)

        attn_scores = queries @ keys.transpose(2, 3)

        # Apply the causal mask to ensure that each position can only attend to previous positions (including itself)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.nn.functional.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Drop out to prevent overfitting.
        attn_weights = self.dropout(attn_weights)

        context_vector = (attn_weights @ values).transpose(1,2)

        # Reshape (flatten) back into the shape (b, num_tokens, dim_out).
        # Combines heads, where self.dim_out = self.num_heads * self.head_dim.
        context_vector = context_vector.contiguous().view(b, num_tokens, self.dim_out)

        # (Optional) This projection layer is not strictly necessary (because above we have reshaped the context vector) but is commonly used in practice.
        context_vector = self.out_projection(context_vector)

        return context_vector
