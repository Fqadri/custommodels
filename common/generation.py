import torch


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


# With eos_id set, finished rows receive EOS placeholders while other rows continue.
# Return when all rows finish or max_new_tokens is reached; trim each generated continuation at its first EOS.
def generate_text_from_inputsample_with_sampling(model,
                                                 idx,
                                                 max_new_tokens,
                                                 context_size,
                                                 temperature=0.0,
                                                 top_k=None,
                                                 eos_id=None):
    finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]  # Shape (batch, vocab_size) representing the last token's logits for each batch element.

        if top_k is not None:
            # Top-k sampling: Select the top k tokens with highest probabilities and sample from them.
            topk_logits, _topk_indices = torch.topk(logits, top_k) # Get top-k logits and their indices
            min_val = topk_logits[:, -1:]  # Minimum value in the top-k logits
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

        if eos_id is not None:
            idx_next = torch.where(finished.unsqueeze(1), eos_id, idx_next)
            finished |= idx_next.squeeze(1) == eos_id
            if finished.all():
                print(f"All sequences reached {eos_id}. Stopping generation.")
                break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
