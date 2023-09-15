## Tensor IR（TIR）

TIR的数据结构其实是一个AST（抽象语法树），然后这个语法树可以表示变量的声明，初始化，变量的计算，函数调用以及控制流（如if-else条件判断，循环等等）等等。所以只要我们遍历一下TIR对应的AST就可以实现一对一的将其翻译到目标硬件了。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/640.png" alt="图片" style="zoom: 67%;" />

IRModule 是在机器学习编译中保存元张量函数（也即PrimFunc）集合的容器对象，它是TVM进行编译的最小完整单元。TVM不同的前端表示最终都会被封装到IRModule中进行编译，在Linux下IRModule就是一个.so动态链接库。然后PrimFunc叫作元张量函数，它内部封装了一个完整的TIR AST。

现在TVM基于Python AST实现了一种新的特定领域的方言让我们可以直接使用Python来编写TIR AST。我们这里举一个例子：

```python
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer[(128, 128), "float32"],
                B: T.Buffer[(128, 128), "float32"],
                C: T.Buffer[(128, 128), "float32"]):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0)
```

## tvm.ir基础设施

对于IR来说，Type和Expr是尤为关键的两个概念。Type包含基础的数据类型如Int，Float，Double等等，也包含一些自定义的复杂类型比如函数类型，Tensor类型等。而对于Expr来说，既包含可以直接映射到Low-level IR的PrimExpr，又包含RelayExpr。

在TensorTypeNode中，shape也是其中一个属性，所以TVM在做类型推到的时候也要包括Shape的推断。也正是因为在IR中Shape是Type的一部分（比如`Tensor[(m, n)]`和`Tensor[(m, 4)]`是不同的Type）

Relax通过引入一个新的Type叫作DynTensor较好的解决了动态Shape的表示问题，DynTensor包含的信息是Dtype和Shape的纬度，但Shape本身的表达式是独立存储的。也就是`Tensor[(m, n)]`和`Tensor[(_, _)]`都是同一个Type， 但是`Tensor[(_, _)]`和`Tensor[(_, _, _)]`是不同的Type，这样就从原生上支持了动态Shape。

```python
class DynTensorTypeNode : public BaseTensorTypeNode {
 public:
  /*!
   * \brief The number of dimensions of the tensor, use -1 to denote tensor with unknwon number of
   * dimensions.
   */
  int ndim; //现在直接定义ndim而不是shape
  /*! \brief The content data type, use void to denote the dtype is unknown. */
  DataType dtype;
  ...
};
```

我们紧接着看一下Expr的定义（`https://github.com/apache/tvm/blob/main/include/tvm/ir/expr.h`），Expr分成PrimExpr以及RelayExpr。其中PrimExpr保存了一个runtime时候的Dtype，然后例如表示一个整数的Expr就可以通过继承PrimExprNode来实现，IntImm表示的是整数字面值表达式，所以它记录了一个int类型的value成员。

总的来说，无论是高级别的Relay，Relax还是低级别的TIR，它们最终都是由这里的Expr和Type为基础来表达的。因为对于Relay和TIR来讲，它们的op定义都是继承自RelayExprNode：