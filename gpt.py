import torch
from tqdm import trange
from matplotlib import pyplot as plt

def encode(chars, ctoi): return [ctoi[c] for c in chars]
def decode(indices, itoc): return ''.join([itoc[i] for i in indices])

def get_batch(data, block_size, batch_size):
    indices = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[index: index+block_size] for index in indices])
    y = torch.stack([data[index+1:index + block_size + 1] for index in indices])
    return x, y

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, n_embeddings: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.block_size = block_size # context length
        self.query = torch.nn.Linear(n_embeddings, head_size, bias=False)
        self.key = torch.nn.Linear(n_embeddings, head_size, bias=False)
        self.value = torch.nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(self.block_size, self.block_size)))

    def forward(self, input):
        affinities = (self.query(input) @ self.key(input).transpose(-2, -1)) * self.head_size**-0.5
        # to make this an encoder, remove mask
        wei = torch.masked_fill(affinities, self.tril == 0, float('-inf')) # type: ignore
        wei = torch.nn.functional.softmax(wei, dim=2)
        logits = torch.nn.functional.dropout(wei @ self.value(input), p=0.1)
        return logits

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads:int, n_embeddings:int, head_size:int, block_size:int):
        super().__init__()
        self.heads = torch.nn.ModuleList([ScaledDotProductAttention(n_embeddings, head_size, block_size) for _ in range(num_heads)])

    def forward(self, input):
        return torch.cat([head(input) for head in self.heads], dim=-1)

class TransformerBlock(torch.nn.Module):
    def __init__(self, vocab_size:int, n_embeddings:int, block_size:int, head_size:int, num_heads:int):
        super().__init__()
        self.block_size = block_size
        self.embeddings = torch.nn.Embedding(vocab_size, n_embeddings)
        self.positional_embeddings = torch.nn.Embedding(block_size, n_embeddings) # in the paper its just a wave function
        self.mha = MultiHeadAttention(num_heads, n_embeddings, head_size//num_heads, block_size) 
        self.last_layer = torch.nn.Linear(n_embeddings, vocab_size, bias=False)

    def forward(self, index: torch.Tensor, target: torch.Tensor | None = None):
        token_embeddings = self.embeddings(index)
        positional_embeddings = self.positional_embeddings(torch.arange(index.shape[1]))

        x = token_embeddings + positional_embeddings
        mha = self.mha(x)
        logits = self.last_layer(self.mha(x))

        if target is not None:
            batch_size, block_size, vocab_size = logits.shape
            logits = logits.view(batch_size * block_size, vocab_size)
            targets = target.view(batch_size * block_size)
            loss = torch.nn.functional.cross_entropy(logits, targets)
        else:
            loss = None
        return logits, loss

    def generate(self, index: torch.Tensor, max_new_tokens:int):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits = self.embeddings(index_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=1)
            next_index = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, next_index), dim=1)
        return index


if __name__ == "__main__":
    torch.manual_seed(23412)

    # hyper parameters
    ratio = 0.9
    batch_size = 32
    block_size = 8 # context length
    training_steps = 10000
    n_embeddings = 32
    head_size = 32
    num_heads = 4

    # read data
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Length of dataset: {len(text)}")
    chars = sorted(list(set(text))) # vocab
    ctoi = {c:i for i, c in enumerate(chars)}
    itoc = {i:c for i, c in enumerate(chars)}
    train_len = int(len(text) * ratio)
    vocab_size = len(chars)

    data = torch.tensor(encode(text, ctoi))
    test = data[train_len:]
    train = data[:train_len]

    # define model and optimizer
    model = TransformerBlock(vocab_size, n_embeddings, block_size, head_size, num_heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # training
    losses = []
    print("training...")
    for steps in trange(training_steps):
    # for _ in range(1):
        X, Y = get_batch(train, block_size, batch_size)
        output, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(decode(model.generate(torch.zeros(1, 1, dtype=torch.long), 100)[0].tolist(), itoc))

    plt.plot(losses)
    plt.show()
    print("Loss: ", losses[-1])





