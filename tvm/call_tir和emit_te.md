```python
    @T.prim_func
    def linear1(X: T.Buffer[(1, 128), "float32"], 
                W: T.Buffer[(10, 128), "float32"], 
                B: T.Buffer[(10,), "float32"], 
                Z: T.Buffer[(1, 10), "float32"]):
        T.func_attr({"global_symbol": "linear1", "tir.noalias": True})
        Y = T.alloc_buffer((1, 10), "float32")
        for i, j, k in T.grid(1, 10, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
    
        for i, j in T.grid(1, 10):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"), 
             w0: R.Tensor((128, 784), "float32"), 
             b0: R.Tensor((128,), "float32"), 
             w1: R.Tensor((10, 128), "float32"), 
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_tir(linear0, (x, w0, b0), (1, 128), dtype="float32")
            lv1 = R.call_tir(relu0, (lv0,), (1, 128), dtype="float32")
            out = R.call_tir(linear1, (lv1, w1, b1), (1, 10), dtype="float32")
            R.output(out)
        return out
```

call_tir 接受一个prim_func作为参数，同时接受一个参数列表，再构造一个输出的res tensor，将输入和输出用来调用prim_func，执行之后，将res作为返回值返回。call_tir起到一个构造数据流图的作用



emit_te的作用：

`C = bb.emit_te(te_matmul, A, B).`

- Create an input `te.placeholder` for A and B
- Run them through `te_matmul` function.
- Call into `te.create_prim_func` to create a TensorIR function.
- Generate a call into the function via `call_tir`.

```python
@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: te.Tensor):
    # x_torch = torch.from_dlpack(x)
    # out_torch = torch.from_dlpack(out)
    return torch.maximum(x, torch.Tensor([0.0]))

def te_relu(A: te.Tensor) -> te.Tensor:
    # return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")
    return te.compute(
    A.shape, lambda *i: tvm.tir.call_packed("env.relu", A(*i)), # 调用注册的外部函数
    name="C"
)
```

```python
def map_matmul(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    B = node_map[node.args[1]]
    return bb.emit_te(te_matmul, A, B)

def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)

MyModule = from_fx(
    fx_module,
    input_shapes=[(1,128)],
    call_function_map={
        torch.matmul: map_matmul,
        torch.relu: map_relu, 
    },
    call_module_map={}
)
```

注册外部函数lnumpy_relu到env.relu，然后在te_relu函数中的te.compute中使用tvm.tir.call_packed来调用自己写的relu算子。最后使用emit_te来创建数据流图