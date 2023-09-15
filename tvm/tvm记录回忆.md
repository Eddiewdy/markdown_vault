==IRModule==

可以被编译的最小完整单元，TIR和relay都必须封装到IRModule中才能编译。

==PrimFunc==

封装了一个完整的AST，作为IRModule的一个API。每个PrimFunc都对应生成的so库的函数入口。一个IRModule 可以有多个 PrimFunc

==CodeGen==

对 AST 进行中序遍历的过程中，将 AST Node 翻译为相应的编程语法

==TVMScript==

实现了TIR AST，直接复用了 Python AST，可以使用 Python 语法直接编写 AST。TE和TVMScript都可以实现TIR AST

**CodeGen 原理：以 CodeGenC 为例**

TIR 能 lower 成目标源代码，关键是 CodeGen。上面提到的 CodeGenCHost，以及 CodeGenCUDA，都是继承自 CodeGenC，即将 TIR lower 为 C++ 代码。

因为 TIR AST 是一个 Graph 结构（Tree 也是一种特殊的树），因此 CodeGenC 根本上是一个 Graph 遍历器。当 CodeGenC 遍历到某个 TIR Node 的时候，根据 TIR Node 的类型和属性，翻译为相应的 C++ 代码。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230508233907529.png" alt="image-20230508233907529" style="zoom:50%;" />

TE 编译为 TIR 的流程，主要涉及下面几个关键步骤：

1. 将 lambda 表达式，转换为 PrimExpr（即 AST），并封装到 Tensor 的 Operation 当中；
2. 根据 input 和 output 的关系，将所有的 Tensor 拼接为一个完整的 AST Graph；
3. 遍历 AST Graph，根据不同的节点，转化为相应的 TIR 节点。