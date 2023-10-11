**gemm优化步骤**

- 数据分块，利用cache
- 是用寄存器保存矩阵C，减少访存
- 更大的分块4*4
- Neon指令集优化，instrics，SIMD
- 一次计算多行 数据pack

- 使用thread绑定，多线程优化
- 使用gpu共享memory



**本地存储分块 (Local Blocking)**

我们可以进行循环拆分，来增加整体内存复用。特别是，我们引入了局部切分，这样我们只需要从 `A` 和 `B` 加载一次条形数据（上图中的灰色部分），然后使用它们来执行v*v矩阵乘法结果。

==简单说就是使用线程内部寄存器==

这种本地存储的切分有助于减少内存压力，因为条形数据块的每个元素都被重用了 `V` 次。



**共享内存分块 (Shared Memory Blocking)**

我们的第一次尝试没有考虑位于同一个 GPU 线程块中的相邻线程，我们可以将它们需要的数据加载到一块共享内存 (shared memory) 中。

下面的转换完成了这项操作：

```python
i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)
```



**im2col+gemm**

对kernel进行数据pack，4*4，增大列数使得访存友好，同时也对模式矩阵做pack，进行gemm的时候，一次同时计算四行，每次计算8列

tile im2col，是的展开的矩阵利于做分块的gemm



**Array Packing**

改变B矩阵的数据排布方式，使得数据访问的顺序和数据存储的顺序一致



**分块考虑的事情**

- SIMD指令排布是否能排满流水线（根据向量指令的延时和发射数量来计算）
- 是否满足逻辑寄存器的要求
- 分块的大小要能够让乱序发射掩盖数据访存的延时



### 矩阵乘法优化

https://tvm.apache.org/docs/tutorial/tensor_expr_get_started.html#example-2-manually-optimizing-matrix-multiplication-with-te

1. 分块
   - 跟好的利用缓存，将内部分为32*32的小矩阵乘法，放入l1 cache
2. 内部y方向向量化，小矩阵变成广播标量乘向量
   - 注意分块策略
3. 调整循环顺序，是的A矩阵的按行访问（是否真的会带来提升，因为同时也导致B的向量寄存器多次读）
4. 数据packing，改变数据排布方式，使得内层循环访存是连续的

```python
# We have to re-write the algorithm slightly.
bn = 32
packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
C = te.compute(
    (M, N),
    # indexmod 求余数
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
)

s = te.create_schedule(C.op)

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
(k,) = s[C].op.reduce_axis
ko, ki = s[C].split(k, factor=4)

s[C].reorder(xo, yo, ko, xi, ki, yi)
s[C].vectorize(yi)

x, y, z = s[packedB].op.axis
s[packedB].vectorize(z)
s[packedB].parallel(x)

evaluate_operation(s, [A, B, C], target=target, name="mmult", optimization="array packing", log=log)

# Here is the generated IR after array packing.
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

```python
out:
array packing: 0.107791
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True})
        packedB = T.allocate([32768], "float32x32", "global")
        packedB_1 = T.Buffer((32768,), "float32x32", data=packedB)
        # packing
        for x in T.parallel(32):
            for y in range(1024):
                B_1 = T.Buffer((1048576,), data=B.data)
                packedB_1[x * 1024 + y] = B_1[y * 1024 + x * 32:y * 1024 + x * 32 + 32]
        for x_outer, y_outer in T.grid(32, 32):
            # 1024 * 1024 C buffer
            C_1 = T.Buffer((1048576,), data=C.data)
            for x_inner_init in range(32):
                C_1[x_outer * 32768 + x_inner_init * 1024 + y_outer * 32:x_outer * 32768 + x_inner_init * 1024 + y_outer * 32 + 32] = T.Broadcast(T.float32(0), 32)
            for k_outer, x_inner, k_inner in T.grid(256, 32, 4):
                cse_var_3: T.int32 = x_outer * 32768 + x_inner * 1024
                cse_var_2: T.int32 = k_outer * 4
                cse_var_1: T.int32 = cse_var_3 + y_outer * 32
                A_1 = T.Buffer((1048576,), data=A.data)
                C_1[cse_var_1:cse_var_1 + 32] = C_1[cse_var_1:cse_var_1 + 32] + T.Broadcast(A_1[cse_var_3 + cse_var_2 + k_inner], 32) * packedB_1[y_outer * 1024 + cse_var_2 + k_inner]
```

5. 小矩阵共享内存cache
5. 多核心并行
