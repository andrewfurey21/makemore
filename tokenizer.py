from typing import List
from tqdm import trange

def get_stats(ints:List[int]):
    counts = {}
    for pair in zip(ints, ints[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ints, pair, newint):
    newints = []
    i = 0
    while i < len(ints):
        if i < len(ints) - 1 and ints[i] == pair[0] and ints[i+1] == pair[1]:
            newints.append(newint)
            i += 2
        else:
            newints.append(ints[i])
            i += 1
    return newints

def compress(tokens):
    num_merges = 40
    for i in range(num_merges):
        stats = get_stats(tokens)
        top_pair = max(stats, key=stats.get) # type: ignore
        new_token = i + 256
        tokens = merge(tokens, top_pair, new_token)
    return tokens

def encode(): pass
def decode(): pass

if __name__ == "__main__":
    file_name = "./english.txt"

    with open(file_name, "r") as f:
        example = f.read()

    tokens = [ord(char) for char in example]
    print(tokens)
    compressed_tokens = compress(tokens)
    print("compression rate: ", len(tokens) / len(compressed_tokens))
    print(compressed_tokens)


