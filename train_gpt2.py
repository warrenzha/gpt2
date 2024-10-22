import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPT, GPTConfig
import tiktoken


def main():
    # attempt to autodetect a GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig())
    model.eval()
    model.to(device)
    print("Loading... Done!\n")

    # Let's try to generate some text
    num_return_sequences = 5
    max_length = 30

    # prefix tokens
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
    x = tokens.to(device)

    # generate! x is (B, T), where B = 5, T = 8
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:  # max_length=30
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)[0]  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == '__main__':
    main()