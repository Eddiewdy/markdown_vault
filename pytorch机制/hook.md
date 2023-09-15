hook的调用流程

```python
def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
        hook(self, input)
    if torch._C._get_tracing_state():
        result = self._slow_forward(*input, **kwargs)
    else:
        result = self.forward(*input, **kwargs)
    for hook in self._forward_hooks.values():
        hook_result = hook(self, input, result)
        if hook_result is not None:
            raise RuntimeError(
                "forward hooks should never return any values, but '{}'"
                "didn't return None".format(hook))
```

**torch.nn.Module.register_forward_hook**

forward_hook的input output不能修改，hook都不能带返回值

```python
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return x
def farward_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)
if __name__ == "__main__":
    # 初始化网络
    net = Net()
    net.conv1.weight[0].fill_(1)
    net.conv1.weight[1].fill_(2)
    net.conv1.bias.data.zero_()
    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(farward_hook)
    # inference
    fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
    output = net(fake_img)
    # 观察
    print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))
```





**torch.nn.Module.register_backward_hook**

**功能**：Module反向传播中的hook,每次计算module的梯度后，自动调用hook函数。 

**形式**：hook(module, grad_input, grad_output) -> Tensor or None 

**注意事项**：当module有多个输入或输出时，grad_input和grad_output是一个tuple。 

**返回值**：a handle that can be used to remove the added hook by calling handle.remove() 