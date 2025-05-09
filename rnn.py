import torch
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from typing import List

torch.manual_seed(1)

class Embeddings:
    def __init__(self, vocab_size:int, dims:int):
        self.emb = torch.randn(vocab_size, dims)

    def parameters(self):
        return [self.emb]

    def __call__(self, indices: torch.Tensor):
        return self.emb[indices]

class Linear:
    def __init__(self, fan_in:int, fan_out:int, *, usebias=True):
        self.weights = torch.rand((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = None if not usebias else torch.zeros(fan_out)
        self.fan_in = fan_in
        self.fan_out = fan_out

    def parameters(self):
        params = [self.weights]
        return params if self.bias is None else params + [self.bias]

    def __call__(self, input: torch.Tensor):
        input = input.view(-1, self.fan_in)
        output = input @ self.weights
        output = output if self.bias is None else output + self.bias
        return output

class BatchNorm1D:
    def __init__(self, fan_in:int):
        self.running_mean = torch.zeros(fan_in)
        self.running_std = torch.ones(fan_in)

        self.gamma = torch.ones(fan_in)
        self.beta = torch.zeros(fan_in)

    def parameters(self):
        return [self.gamma, self.beta]

    def __call__(self, input: torch.Tensor, training=True):
        if not training:
            with torch.no_grad():
                return self.gamma * (input - self.running_mean) / self.running_std + self.beta
        else:
            mean = input.mean(dim=0)
            std = input.std(dim=0)

            with torch.no_grad():
                self.running_mean = self.running_mean * 0.999 + mean * 0.001
                self.running_std = self.running_std * 0.999 + std * 0.001

            return self.gamma * (input - mean) / std + self.beta

class Tanh:
    @staticmethod
    def __call__(input: torch.Tensor):
        return input.tanh()

    @staticmethod
    def parameters():
        return []

class RNN:
    def __init__(self, input_size:int, hidden_size:int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_weights = torch.randn((input_size, hidden_size), dtype=torch.float32) * .01
        self.state_weights = torch.randn((hidden_size, hidden_size), dtype=torch.float32) * .01

    def __call__(self, input:torch.Tensor):
        input_t = input.transpose(0, 1)
        seq_len, batch_size, _ = input_t.shape
        state = torch.zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            state = input_t[t] @ self.input_weights + state @ self.state_weights
        return state

    def parameters(self):
        return [self.input_weights, self.state_weights]


class CharRNN:
    def __init__(self, vocab_size:int, embedding_dim:int, block_size:int, hidden:int):
        self.layers = [
            Embeddings(vocab_size, embedding_dim),
            RNN(embedding_dim, hidden),
            BatchNorm1D(hidden),
            Tanh(),
            Linear(hidden, vocab_size)
        ]

    def zero_grad(self):
        for layer in self.layers:
            params = layer.parameters()
            for p in params:
                if p is not None:
                    p.grad = None

    def parameters(self) -> List[torch.Tensor]:
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def __call__(self, input:torch.Tensor, training=True):
        output = self.layers[0](input)
        for layer in self.layers[1:]:
            if isinstance(layer, BatchNorm1D):
                output = layer(output, training)
            else:
                output = layer(output)
        return output

if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    special = '.'
    chars = sorted(list(set(''.join(words))))
    stoi = {s:(i+1) for i, s in enumerate(chars)}
    stoi[special] = 0
    itos = {s:i for i, s in stoi.items()}

    # hyperparameters
    block_size = 3
    batch_size = 32
    emb_dims = 15
    hidden = 500
    lr_start = 0.05
    steps = 20000
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

    # model
    model = CharRNN(vocab_size, emb_dims, block_size, hidden)
    for p in model.parameters(): p.requires_grad = True

    # training
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{num_params=}")
    print("Training...")
    sloss = []
    sit = []
    for batch in tqdm(range(steps * epoch)):
        # make a batch
        i = torch.randint(0, X.shape[0], (batch_size,))
        xs = X[i]
        ys = Y[i]

        # forward pass
        logits = model(xs)
        loss = F.cross_entropy(logits, ys)

        # backward pass
        model.zero_grad()
        loss.backward()
        # update
        lr = lr_start
        for p in model.parameters():
            if p.requires_grad:
                p.data += -lr * p.grad # type: ignore
        # stats
        sit.append(batch)
        sloss.append(loss.item())
        # print(loss.item())

    print(sloss[-1])
    # for i in range(0, len(sloss), len(sloss)//10):
    #     print(f"Loss {i}: {sloss[i]}")

    # Inference
    for i in range(10):
        string = []
        context = [0] * block_size
        while True:
            context_tensor = torch.tensor(context).reshape(1, block_size)
            with torch.no_grad():
                logits = model(context_tensor, training=False)
                probs = F.softmax(logits, dim=1)
            output = torch.multinomial(probs, 1, replacement=True).item()
            context = context[1:] + [output]
            if output == 0:
                break
            string.append(output)
        print(''.join(itos[s] for s in string))

    plt.plot(sit, sloss)
    plt.show()
