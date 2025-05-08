import torch
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt

def model(inputs, size, E, W1, b1, W2, b2):
    # dim=0 is the batches
    emb = E[inputs].view(-1, size) # can only infer one dimension
    hpreact = emb @ W1 + b1
    h1 = torch.tanh(hpreact)
    return h1 @ W2 + b2

if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    special = '.'
    chars = sorted(list(set(''.join(words))))
    stoi = {s:(i+1) for i, s in enumerate(chars)}
    stoi[special] = 0
    itos = {s:i for i, s in stoi.items()}

    # hyperparameters
    gen = torch.Generator().manual_seed(123456)
    block_size = 3
    batch_size = 32
    emb_dims = 10
    hidden = 200
    lr_start = 0.1
    steps = 10000
    epoch = 1
    vocab_size = 27

    # build a dataset
    X, Y = [], []

    for word in words:
        context = [0] * block_size
        for ch in word + special:
            index = stoi[ch]
            X.append(context)
            Y.append(index)
            context = context[1:] + [index]

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    # weights
    E = torch.randn((vocab_size, emb_dims), dtype=torch.float32, requires_grad=True, generator=gen)

    W1 = torch.randn((emb_dims * block_size, hidden), dtype=torch.float32, requires_grad=True, generator=gen)
    b1 = torch.randn(hidden, dtype=torch.float32, requires_grad=True, generator=gen)

    W2 = torch.randn((hidden, vocab_size), dtype=torch.float32, requires_grad=True, generator=gen)
    b2 = torch.randn(vocab_size, dtype=torch.float32, requires_grad=True, generator=gen)

    # training
    parameters = [E, W1, b1, W2, b2]
    num_params = sum(p.numel() for p in parameters)
    print(f"{num_params=}")
    print("Training...")
    sloss = []
    sit = []
    for batch in tqdm(range(steps * epoch)):
        # make a batch
        i = torch.randint(0, X.shape[0], (batch_size,), generator=gen)
        xs = X[i]
        ys = Y[i]

        # forward pass
        logits = model(xs, emb_dims * block_size, *parameters)

        # counts = logits.exp()
        # probs = counts / counts.sum(dim=1, keepdim=True)
        # loss = -probs[torch.arange(32), Y].log().mean()
        # equivalent to ^^
        loss = F.cross_entropy(logits, ys)

        # backward pass
        for p in parameters: p.grad = None
        loss.backward()
        # update
        lr = lr_start / (batch // steps + 1)
        for p in parameters:
            p.data += -lr * p.grad # type: ignore
        # stats
        sit.append(batch)
        sloss.append(loss.item())

    print(sloss[-1])
    # for i in range(0, len(sloss), len(sloss)//10):
    #     print(f"Loss {i}: {sloss[i]}")

    for i in range(10):
        string = []
        context = [0] * block_size
        while True:
            logits = model(context, emb_dims * block_size, *parameters)
            probs = F.softmax(logits, dim=1)
            output = torch.multinomial(probs, 1, replacement=True, generator=gen).item()
            context = context[1:] + [output]
            if output == 0:
                break
            string.append(output)
        print(''.join(itos[s] for s in string))

    # plt.plot(sit, sloss)
    # plt.show()
