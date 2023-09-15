#### 爱因斯坦求和规定

爱因斯坦求和约定（einsum）提供了一套既简洁又优雅的规则，可实现包括但不限于：向量内积，向量外积，矩阵乘法，转置和张量收缩（tensor contraction）等张量操作，熟练运用 einsum 可以很方便的实现复杂的张量操作，而且不容易出错。

```python
c = torch.einsum("ik,kj->ij", [a, b])
```

- 自由索引，**出现在箭头右边的索引**，比如上面的例子就是 i 和 j；
- 求和索引，**只出现在箭头左边的索引**，表示中间计算结果需要这个维度上求和之后才能得到输出，比如上面的例子就是 k；

- 规则一，equation 箭头左边，在不同输入之间重复出现的索引表示，把输入张量沿着该维度做乘法操作，比如还是以上面矩阵乘法为例， "ik,kj->ij"，k 在输入中重复出现，所以就是把 a 和 b 沿着 k 这个维度作相乘操作；
- 规则二，只出现在 equation 箭头左边的索引，表示中间计算结果需要在这个维度上求和，也就是上面提到的求和索引；
- 规则三，equation 箭头右边的索引顺序可以是任意的，比如上面的 "ik,kj->ij" 如果写成 "ik,kj->ji"，那么就是返回输出结果的转置，用户只需要定义好索引的顺序，转置操作会在 einsum 内部完成。

```python
torch_ein_out = torch.einsum('ii->i', [a]).numpy()

np_a = a.numpy()
# 循环展开实现
np_out = np.empty((3,), dtype=np.int32)
# 自由索引外循环
for i in range(0, 3):
    # 求和索引内循环
    # 这个例子并没有求和索引，
    # 所以相当于是1
    sum_result = 0
    for inner in range(0, 1):
        sum_result += np_a[i, i]
    np_out[i] = sum_result
    
a = torch.arange(6).reshape(2, 3)
# i = 2, j = 3
torch_ein_out = torch.einsum('ij->ji', [a]).numpy()
torch_org_out = torch.transpose(a, 0, 1).numpy()

np_a = a.numpy()
# 循环展开实现
np_out = np.empty((3, 2), dtype=np.int32)
# 自由索引外循环
for j in range(0, 3):
    for i in range(0, 2):
        # 求和索引内循环
        # 这个例子并没有求和索引
        # 所以相当于是1
        sum_result = 0
        for inner in range(0, 1):
            sum_result += np_a[i, j]
        np_out[j, i] = sum_result
```

