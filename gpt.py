import torch

def get_batch(data, batch_size, block_size):
    indices = torch.randint(len(data) - block_size, size=batch_size)
    



if __name__ == "__main__":
    torch.manual_seed(1336)

    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Length of dataset: {len(text)}")
    # print(f"{text[:50]=}")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # print(chars)

    ctoi = {c:i for i, c in enumerate(chars)}
    itoc = {i:c for i, c in enumerate(chars)}
    encode = lambda chars: [ctoi[c] for c in chars]
    decode = lambda indices: ''.join([itoc[i] for i in indices])
    # print(decode(encode("andrew is my name")))

    train_size = int(len(text) * .9)
    train_data = text[:train_size]
    test_data = text[train_size:]
    dev_data = text[:100]

    block_size = 8
    batch_size = 4

    x, y = get_batch(dev_data, batch_size, block_size)



