import torch
import torch.nn.functional as F
import numpy as np

def simple(freq):
    gen = torch.Generator().manual_seed(2147483647)
    index: int = 0
    while True:
        probs = freq[index].float()
        probs = probs / probs.sum()
        index = int(torch.multinomial(probs, num_samples=1, replacement=True, generator=gen).item())
        if (index == 0): break
        print(itos[index], end="")
    print()

# Likelihood: product of probabilities
# .8 * .8 = .64

if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    special = '.'

    chars = sorted(list(set(''.join(words))))
    stoi = {s:(i+1) for i, s in enumerate(chars)}
    stoi[special] = 0
    itos = {s:i for i, s in stoi.items()}

    size = 27
    freq = np.zeros((size, size), dtype=np.int32)

    # training
    for w in words:
        c = [special] + list(w) + [special]
        for ch1, ch2 in zip(c, c[1:]):
            index1 = stoi[ch1]
            index2 = stoi[ch2]
            freq[index1, index2] += 1

    freq = torch.from_numpy(freq)
    p = (freq.float() + 1)
    p /= p.sum(dim=1, keepdim=True)

    # eval
    nll = 0.0
    n = 0
    # for w in words[:3]:
    for w in ["andrewqj"]:
        c = [special] + list(w) + [special]
        for ch1, ch2 in zip(c, c[1:]):
            index1 = stoi[ch1]
            index2 = stoi[ch2]
            prob = p[index1, index2]
            nll -= torch.log(prob)
            n+=1
            print(f"{ch1}{ch2}: {prob:.4f}")

    nll /= n
    print(f"{nll=}")

    # create training set
    gen = torch.Generator().manual_seed(2147483647)
    inputs, outputs = [], []
    for w in words[:1]:
        c = [special] + list(w) + [special]
        for ch1, ch2 in zip(c, c[1:]):
            index1 = stoi[ch1]
            index2 = stoi[ch2]
            inputs.append(index1)
            outputs.append(index2)

    xs = torch.tensor(inputs)
    # num = xs.nelement() # difference beteween nelement and numel?
    ys = torch.tensor(outputs)

    ohxs = F.one_hot(xs, num_classes=27).float()
    ohys = F.one_hot(ys, num_classes=27).float()
    weights = torch.randn((27, 27), generator=gen, requires_grad=True, dtype=torch.float32)


    for batch in range(10):
        # forward pass
        logits = (ohxs @ weights) # log counts
        counts = logits.exp()
        probs = counts / counts.sum(dim=1, keepdim=True)
        loss = -probs[..., ys].log().mean()
        print(loss)

        # backward pass
        weights.grad = None
        loss.backward()

        # update
        # weights.data += -0.1 * weights.grad # type: ignore
        weights.data += -weights.grad # type: ignore


