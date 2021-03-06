#%% [markdown]
# # 201 Torch and Numpy
#
# View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
# My Youtube Channel: https://www.youtube.com/user/MorvanZhou
#
# Dependencies:
# * torch: 0.4.1
# * numpy: 1.15.4
#
# Details about math operation in torch can be found in: http://pytorch.org/docs/torch.html#math-operations
#

#%%
import torch
import numpy as np

#%%
# convert numpy to tensor or vise versa
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:',
    np_data,  # [[0 1 2], [3 4 5]]
    '\ntorch tensor:',
    torch_data,  #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:',
    tensor2array,  # [[0 1 2], [3 4 5]]
)

#%%
# abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32-bit floating point
print(
    '\nabs',
    '\nnumpy: ',
    np.abs(data),  # [1 2 1 2]
    '\ntorch: ',
    torch.abs(tensor)  # [1 2 1 2]
)

#%%
tensor.abs()

#%%
# sin
print(
    '\nsin',
    '\nnumpy: ',
    np.sin(data),  # [-0.84147098 -0.90929743  0.84147098  0.90929743]
    '\ntorch: ',
    torch.sin(tensor)  # [-0.8415 -0.9093  0.8415  0.9093]
)

#%%
tensor.sigmoid()

#%%
tensor.exp()

#%%
# mean
print(
    '\nmean',
    '\nnumpy: ',
    np.mean(data),  # 0.0
    '\ntorch: ',
    torch.mean(tensor)  # 0.0
)

#%%
# matrix multiplication
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit floating point
# correct method
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ',
    np.matmul(data, data),  # [[7, 10], [15, 22]]
    '\ntorch: ',
    torch.mm(tensor, tensor)  # [[7, 10], [15, 22]]
)

#%%
# incorrect method
data = np.array(data)
tensor = torch.Tensor(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ',
    data.dot(data),  # [[7, 10], [15, 22]]
    '\ntorch: ',
    torch.dot(
        tensor.dot(tensor)
    )  # NOT WORKING! Beware that torch.dot does not broadcast, only works for 1-dimensional tensor
)

#%% [markdown]
# Note that:
#
# torch.dot(tensor1, tensor2) → float
#
# Computes the dot product (inner product) of two tensors. Both tensors are treated as 1-D vectors.

#%%
tensor.mm(tensor)

#%%
tensor * tensor

#%%
torch.dot(torch.Tensor([2, 3]), torch.Tensor([2, 1]))
