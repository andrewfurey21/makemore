import torch
from tqdm import trange
import matplotlib.pyplot as plt


def encode(chars, ctoi): return [ctoi[c] for c in chars]
def decode(indices, itoc): return ''.join([itoc[i] for i in indices])

def get_batch(data, block_size, batch_size):
    indices = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[index: index+block_size] for index in indices])
    y = torch.stack([data[index+1:index + block_size + 1] for index in indices])
    return x, y

class Bigram(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__();
        self.table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, index: torch.Tensor, target: torch.Tensor):
        logits = self.table(index)
        batch_size, block_size, vocab_size = logits.shape
        logits = logits.view(batch_size * block_size, vocab_size)
        targets = target.view(batch_size * block_size)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, index: torch.Tensor, max_new_tokens:int):
        for _ in range(max_new_tokens):
            logits = self.table(index)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=1)
            next_index = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, next_index), dim=1)
        return index


if __name__ == "__main__":
    torch.manual_seed(23412)
    ratio = 0.9

    batch_size = 64
    block_size = 8 # context length
    training_steps = 100000

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

    model = Bigram(vocab_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    print("training...")
    for steps in trange(training_steps):
        X, Y = get_batch(train, block_size, batch_size)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(decode((model.generate(torch.zeros(1, 1, dtype=torch.long), max_new_tokens=400)[0]).tolist(), itoc))
    plt.plot(losses)
    plt.show()
    






