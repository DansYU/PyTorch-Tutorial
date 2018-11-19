#%% [markdown]
# # 202 Tensor
#
# View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
# My Youtube Channel: https://www.youtube.com/user/MorvanZhou
#
# Dependencies:
# * torch: 0.4.1
#
# tensor in torch is to build a computational graph,
# but this graph is dynamic compared with a static graph in Tensorflow or Theano.
# So torch does not have placeholder, torch can just pass tensor to the computational graph.
#

#%%
import torch
# from torch.autograd import Variable

#%%
tensor0 = torch.FloatTensor([[1, 2], [3, 4]])  # build a tensor
print(tensor0)  # [torch.FloatTensor of size 2x2]
tensor0.requires_grad_(True)  # by requires_grad flag for compute gradients
print(tensor0)  # [torch.FloatTensor of size 2x2]

variable = torch.tensor(
    [[1, 2], [3, 4]], dtype=torch.float,
    requires_grad=True)  # build a tensor, usually for compute gradients
print(variable)  # [torch.FloatTensor of size 2x2]
#

#%%
tensor0.requires_grad_(False)  # by requires_grad flag for close gradients
t_out = torch.mean(tensor0 * tensor0)  # x^2
v_out = torch.mean(variable * variable)  # x^2
print(t_out)
print(v_out)

#%%
v_out.backward()  # backpropagation from v_out

#%% [markdown]
# $$ v_{out} = {{1} \over {4}} sum(variable^2) $$
#
# the gradients w.r.t the variable,
#
# $$ {d(v_{out}) \over d(variable)} = {{1} \over {4}} 2 variable = {variable \over 2}$$
#
# let's check the result pytorch calculated for us below:

#%%
variable.grad

#%%
variable  # this is data in variable format

#%%
variable.detach()  # this is data in tensor format

#%%
variable.detach().numpy()  # numpy format

#%% [markdown]
# Note that we did `.backward()` on `v_out` but `variable` has been assigned new values on it's `grad`.
#
# As this line
# ```
# v_out = torch.mean(variable*variable)
# ```
# will make a new variable `v_out` and connect it with `variable` in computation graph.

#%%
type(v_out)

#%%
type(v_out.data)
