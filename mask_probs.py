import torch

torch.manual_seed(42)

B, T, C = 1, 4, 2

x = torch.arange(0, 8, dtype=torch.float).reshape(B, T, C)
print(x.shape)

# x[b, t] = mean{i<=t} x[b, i]

xbow = torch.zeros(x.shape)

for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, :t+1].mean(dim=0)

w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)
xbow2 = w @ x

tril = torch.tril(torch.ones(T, T))
w1 = torch.zeros(T, T)
w1 = w1.masked_fill(tril == 0, float('-inf'))
w1 = torch.nn.functional.softmax(w1, dim=1)

xbow3 = w1 @ x

print(x)
print(xbow)
print(w)
print(xbow2)
print(xbow3)
