# Here we will look at text generation strategies using sampling.
# Without this the model will always generate the same output for a given input context. This is called Greedy decoding.


import matplotlib.pyplot as plt
import torch

vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}


# This is sampling (aka Probabilistic sampling) where we sample (i.e. pick) from the entire probability distribution.
# Here token with highest probability will be picked more often, but other tokens will also be picked sometimes.
def print_sampled_tokens(probas):
    torch.manual_seed(123)  # For reproducibility
    sample = [
        torch.multinomial(probas, num_samples=1).item() for i in range(1_000)
    ]  # num_samples=1 means we pick one token at a time.
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} X {inverse_vocab[i]}")


# Temperature based sampling. This is a way to control the randomness of predictions by scaling the logits before applying softmax.
# Higher temperature results in more random predictions, while lower temperature results in more deterministic predictions.
# Temperature is a hyperparameter that you can tune based on your requirements.
def softmax_with_temperature(logits, temperature):
    """
    Apply softmax with temperature to logits.
    Higher temperature -> more uniform distribution (more exploration)
    Lower temperature -> sharper distribution (more exploitation)
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


def plot_softmax_with_temperature(logits, temperatures):
    scaled_probas = [softmax_with_temperature(logits, temp) for temp in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    _fig, ax = plt.subplots(figsize=(5, 3))
    for i, temp in enumerate(temperatures):
        ax.bar(
            x + i * bar_width,
            scaled_probas[i],
            width=bar_width,
            label=f"Temperature={temp}",
        )
    ax.set_ylabel("Probability")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.show()


# Top-k sampling: Select the top k tokens with highest probabilities and sample from them.
def topk_sampling(logits, tok_k, use_temperature=False, temperature=1.0):
    """
    Top-k sampling: Select the top k tokens with highest probabilities and sample from them.
    """
    topk_logits, topk_indices = torch.topk(
        logits, tok_k
    )  # Get top-k logits and their indices
    print("Top-k logits:", topk_logits)
    print("Top-k indices:", topk_indices)

    min_val = topk_logits[-1]  # Minimum value in the top-k logits
    # Set logits not in top-k to -inf so that their softmax probability becomes 0
    logits = torch.where(
        logits < min_val, torch.full_like(logits, float("-inf")), logits
    )

    if use_temperature:
        topk_probas = softmax_with_temperature(topk_logits, temperature)
    else:
        topk_probas = torch.softmax(topk_logits, dim=0)

    sampled_index_in_topk = torch.multinomial(
        topk_probas, num_samples=1
    ).item()  # Sample one index (1 result only with highest probability) from the top-k probabilities
    sampled_token_id = topk_indices[
        sampled_index_in_topk
    ].item()  # Get the actual token id from the original logits

    return inverse_vocab[sampled_token_id]  # Return the sampled token


def main() -> None:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    print_sampled_tokens(probas)
    plot_softmax_with_temperature(next_token_logits, temperatures=[1, 0.1, 5])
    token = topk_sampling(next_token_logits, tok_k=3, use_temperature=False)
    print(f"Top-k sampling without temperature: {token}")
    token = topk_sampling(
        next_token_logits, tok_k=3, use_temperature=True, temperature=0.1
    )
    print(f"Top-k sampling with temperature: {token}")


if __name__ == "__main__":
    main()
