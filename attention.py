"""Day 1: Scaled dot-product attention from scratch in PyTorch.

Install:  pip install torch
Run:      python day1_attention.py
"""
import math
import torch


def attention(Q, K, V):
    """Return (output, weights).

    Q, K, V shape: (batch, tokens, dim)
    """
    d_k = K.size(-1)

    # 1. Compare every Query with every Key -> (batch, tokens, tokens)
    scores = Q @ K.transpose(-2, -1)

    # 2. Scale by sqrt(d_k) so scores don't grow with dimension size
    scores = scores / math.sqrt(d_k)

    # 3. Softmax turns scores into probabilities (each row sums to 1)
    weights = torch.softmax(scores, dim=-1)

    # 4. Weighted mix of the Values
    output = weights @ V
    return output, weights


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Part 1: basic run ----
    tokens, dim = 6,9
    x = torch.randn(1, tokens, dim)  # batch, tokens, dim

    out, weights = attention(x, x, x)  # self-attention: Q = K = V = x
    print("Output shape:", out.shape)  # expect (1, 5, 8)

    print("\nAttention weights (row = token asking, column = token it looks at):")
    print(weights[0])

    print("\nRow sums (should all be 1.0):")
    print(weights[0].sum(dim=-1))

    # ---- Part 2: try a different number of tokens ----
    x10 = torch.randn(1, 10, dim)
    out10, w10 = attention(x10, x10, x10)
    print("\n10 tokens -> output:", out10.shape, "| weights:", w10.shape)

    # ---- Part 3: why divide by sqrt(dim)? ----
    # Without scaling, big dims give huge scores -> softmax becomes almost
    # one-hot (very "peaky"), and gradients become tiny.
    big_dim = 4
    q = torch.randn(1, 5, big_dim)
    k = torch.randn(1, 5, big_dim)

    raw = q @ k.transpose(-2, -1)
    scaled = raw / math.sqrt(big_dim)

    print("\nScore std WITHOUT scaling:", raw.std().item())
    print("Score std WITH scaling:   ", scaled.std().item())
    print("Max softmax weight WITHOUT scaling:", torch.softmax(raw, -1).max().item())
    print("Max softmax weight WITH scaling:   ", torch.softmax(scaled, -1).max().item())