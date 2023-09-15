```python
A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
E = torch.tensor(1., requires_grad=True)
C = A * B
D = C.exp()
F = D + E
print(F)        
# tensor(3.7183, grad_fn=<AddBackward0>) 打印计算结果，可以看到F的grad_fn指向AddBackward，即产生F的运算
print([x.is_leaf for x in [A, B, C, D, E, F]]) 
# [True, True, False, False, True, False] 打印是否为叶节点，由用户创建，且requires_grad设为True的节点为叶节点
print([x.grad_fn for x in [F, D, C, A]])   
# [<AddBackward0 object at 0x7f972de8c7b8>, <ExpBackward object at 0x7f972de8c278>, <MulBackward0 object at 0x7f972de8c2b0>, None]  每个变量的grad_fn指向产生其算子的backward function，叶节点的grad_fn为空
print(F.grad_fn.next_functions) 
# ((<ExpBackward object at 0x7f972de8c390>, 0), (<AccumulateGrad object at 0x7f972de8c5f8>, 0)) 由于F = D + E， 因此F.grad_fn.next_functions也存在两项，分别对应于D, E两个变量，每个元组中的第一项对应于相应变量的grad_fn，第二项指示相应变量是产生其op的第几个输出。E作为叶节点，其上没有grad_fn，但有梯度累积函数，即AccumulateGrad（由于反传时多出可能产生梯度，需要进行累加）
F.backward(retain_graph=True)   
# 进行梯度反传
print(A.grad, B.grad, E.grad)   
# tensor(1.3591) tensor(5.4366) tensor(1.) 算得每个变量梯度，与求导得到的相符
print(C.grad, D.grad)   # None None 为节约空间，梯度反传完成后，中间节点的梯度并不会保留
```

==grad_fn==梯度函数

==next_functions==计算图中前一个梯度函数

叶子结点没有梯度函数，但是有==AccumulateGrad==，表示是叶子结点进行梯度累加



```python
# 这一例子仅可用于每个op只产生一个输出的情况，且效率很低（由于对于某一节点，每次未等待所有梯度反传至此节点，就直接将本次反传回的梯度直接反传至叶节点）
def autograd(grad_fn, gradient):
    auto_grad = {}
    queue = [[grad_fn, gradient]]
    while queue != []:
        item = queue.pop()
        gradients = item[0](item[1])
        functions = [x[0] for x in item[0].next_functions]    
        if type(gradients) is not tuple:
            gradients = (gradients, )
        for grad, func in zip(gradients, functions):    
            if type(func).__name__ == 'AccumulateGrad':
                if hasattr(func.variable, 'auto_grad'):
                    func.variable.auto_grad = func.variable.auto_grad + grad
                else:
                    func.variable.auto_grad = grad
            else:
                queue.append([func, grad])

A = torch.tensor([3.], requires_grad=True)
B = torch.tensor([2.], requires_grad=True)
C = A ** 2
D = B ** 2
E = C * D
F = D + E

autograd(F.grad_fn, torch.tensor(1))
print(A.auto_grad, B.auto_grad)         # tensor(24., grad_fn=<UnbindBackward>) tensor(40., grad_fn=<AddBackward0>)

# 这一autograd同样可作用于编写的模型，我们将会看到，它与pytorch自带的backward产生了同样的结果
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)
        self.fc3 = nn.Linear(5, 2)
        self.fc4 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x1 = self.fc2(x)
        x2 = self.fc3(x)
        x2 = self.relu(x2)
        x2 = self.fc4(x2)
        return x1 + x2

x = torch.ones([10], requires_grad=True)
mlp = MLP()
mlp_state_dict = mlp.state_dict()

# 自定义autograd
mlp = MLP()
mlp.load_state_dict(mlp_state_dict)
y = mlp(x)
z = torch.sum(y)
autograd(z.grad_fn, torch.tensor(1.))
print(x.auto_grad) # tensor([-0.0121,  0.0055, -0.0756, -0.0747,  0.0134,  0.0867, -0.0546,  0.1121, -0.0934, -0.1046], grad_fn=<AddBackward0>)

mlp = MLP()
mlp.load_state_dict(mlp_state_dict)
y = mlp(x)
z = torch.sum(y)
z.backward()
print(x.grad) # tensor([-0.0121,  0.0055, -0.0756, -0.0747,  0.0134,  0.0867, -0.0546,  0.1121, -0.0934, -0.1046])

```

#### `torch.autograd.gradcheck` （数值梯度检查）

```python
test_input = torch.randn(4, requires_grad=True, dtype=torch.float64)    # tensor([-0.4646, -0.4403,  1.2525, -0.5953], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-4)    # pass
torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-4)    # pass

torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-6)    # pass
torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-6)    # pass
```



#### `torch.autograd.anomaly_mode` （在自动求导时检测错误产生路径）

```python
with autograd.detect_anomaly():
...     inp = torch.rand(10, 10, requires_grad=True)
...     out = run_fn(inp)
...     out.backward()
```

#### `torch.autograd.grad_mode` （设置是否需要梯度）

我们在 inference 的过程中，不希望 autograd 对 tensor 求导，因为求导需要缓存许多中间结构，增加额外的内存/显存开销。在 inference 时，关闭自动求导可实现一定程度的速度提升，并节省大量内存及显存（被节省的不仅限于原先用于梯度存储的部分）。我们可以利用`grad_mode`中的`troch.no_grad()`来关闭自动求导：

#### `model.eval()`与`torch.no_grad()`

这两项实际无关，在 inference 的过程中需要都打开：`model.eval()`令 model 中的`BatchNorm`, `Dropout`等 module 采用 eval mode，保证 inference 结果的正确性，但不起到节省显存的作用；`torch.no_grad()`声明不计算梯度，节省大量内存和显存。

#### `torch.autograd.profiler` （提供function级别的统计信息）

```python
import torch
from torchvision.models import resnet18

x = torch.randn((1, 3, 224, 224), requires_grad=True)
model = resnet18()
with torch.autograd.profiler.profile() as prof:
    for _ in range(100):
        y = model(x)
        y = torch.sum(y)
        y.backward()
# NOTE: some columns were removed for brevity
print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```