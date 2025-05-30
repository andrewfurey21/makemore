import numpy as np

a = np.arange(0, 9).reshape(3, 3)
b = np.arange(0, 9).reshape(3, 3)
c = np.arange(0, 18).reshape(2, 3, 3)

diagonal = np.einsum('ii->i', a)
trace = np.einsum('ii->', a)
sum = np.einsum('ij->', a)
transpose = np.einsum('ij->ji', a)
elemmul = np.einsum('ij, ij->ij', a, b)
matmul = np.einsum('ij,jk->ik', a, b)

broadadd = np.einsum('...ij,...jk->', c, a)


print("a/b: ", a)
print("diagonal a: ", diagonal)
print("trace a: ", trace)
print("sum a: ", sum)
print("element wise mul: ", elemmul)
print("transpose a: ", transpose)
print("a matmul b: ", matmul)
print("c: ", c)
print("broadcasted element wise add of c and a: ", broadadd)

