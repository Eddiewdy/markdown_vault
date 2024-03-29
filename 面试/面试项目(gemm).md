- Intel i7-6700 (Skylake), 4.0 GHz (Turbo Boost), 14 nm. RAM: 16 GB, dual DDR4-2400 CL15 (PC-19200).

  - L1 Data cache = 32 KB, 64 B/line, 8-WAY.
  - L1 Instruction cache = 32 KB, 64 B/line, 8-WAY.
  - L2 cache = 256 KB, 64 B/line, 4-WAY
  - L3 cache = 8 MB, 64 B/line, 16-WAY

  - L1 Data Cache Latency = 4 cycles for simple access via pointer
  - L1 Data Cache Latency = 5 cycles for access with complex address calculation (size_t n, *p; n = p[n]).
  - L2 Cache Latency = 12 cycles
  - L3 Cache Latency = 42 cycles (core 0) (i7-6700 Skylake 4.0 GHz)
  - L3 Cache Latency = 38 cycles (i7-7700K 4 GHz, Kaby Lake)
  - RAM Latency = 42 cycles + 51 ns (i7-6700 Skylake)

### gemm

常规的矩阵乘法就是三重循环，那这种方式显然是对存储系统很不友好的，因为没有利用到体系结构的多级缓存。一开始的gemm优化是在armv7的cpu上，优化的地方首先就是将矩阵分块，内层循环负责一个小矩阵的计算，同时要合理设计内部分块的矩阵大小，因为首先是想把整个矩阵能放进l1中，从而减少访存的延迟，同时内层的小矩阵可以用arm的simd指令做优化，neon 支持128bit向量处理，考虑到处理器每个周期发射两条simd，计算的延时大概5个cycle，那么为了把流水线打满，最大化利用向量计算的性能，内层排布至少要有10条的标量乘向量的指令，这一点也会影响矩阵的分块，除此之外还要考虑到load的延时，ld指令和neon指令的比例要小于处理器提供的LD和FMA单元同周期发射能力的比例，这样能通过乱序处理的机制掩藏访存的延时。后来也在m1上做了实验，针对m1的4发射128位的simd，最后把分块的大小设定为8*12，需要29个simd寄存器。

Apple的M1芯片使用了ARM的64位ARMv8-A架构，这意味着它支持ARM的NEON SIMD（单指令，多数据流）扩展。

在ARMv8-A架构中，有32个128位宽的SIMD寄存器，称为V0到V31

性能效果：xxx

进一步的优化还有改变B矩阵的数据排布方式，做数据的pack，使得内层数据访问的顺序和数据存储的顺序一致

那后来我也在gpu上也对gemm算法进行了优化，gpu上的优化可以进行循环拆分，来增加整体内存复用。gpu的分块可以分的更细力度一些，每个线程块负责一个小矩阵的计算，线程块内的每个线程负责更小矩阵的计算，然后把线程块内的小矩阵放入shared memory中，每个线程计算的时候把数据放入寄存器中进行计算。同时也做了数据的prefetch，使用预取来尽可能地掩盖这种latency，当前小迭代在计算的时候，把下一个迭代的数据load到寄存器中，这种优化方式需要使用两倍的存储空间，来实现读写分离。

为什么是三级分块，不是四级或者两级。因为NV的GPU内存结构是三级的，global mem->shared mem，shared mem->register。通过三级分块可以尽可能地提高数据的复用性。也就是尽可能地将数据放到更近的存储结构上进行多次利用。

bm、bn、bk、rm、rn 64、64、8、8、8

**vmlaq_n_f32****vld1q_f32****vaddq_f32****vfmaq_f32**

**VFMA**

现在armv7的开发板上运行的，单核心，neon 128kb长向量指令，内部分块4*4，寄存器16+4+1=21个<32，随后在m1和gpu上执行的，使用tvm进行调优，

### pass

定义了一个数据流分析的模版，支持前向分析和后向分析，包括域、传递函数、边界条件、交汇运算、交汇函数、以及初始值。基于这个分析模版，实现后向的活跃变量分析，以及前向的可用表达式分析。

活跃变量：x - def U use

可用表达式：gen U x - kill

实现了一个循环不变量的代码优化，遍历基本快，找到循环不变量（所有的操作数都在循环外定值），遍历循环不变量，检查三个条件：s所在的基本块是循环所有出口结点(有后继结点在循环外的结点)的支配结点（数据流分析得到dom tree）、循环中没有其它语句对x赋值（SSA必然满足）、检查循环中对x的引用是否仅由inst到达（SSA必然满足），那么就可以把该inst移动到preheader







### gemm优化点记录

朴素的gemm的计算访存比太低，导致如果要跑满算力，带宽成了瓶颈。因此进行矩阵的分块计算，增大数据复用，提高计算访存比。在tile取64的情况下，性能瓶颈就不在内存上了。测试方式（计算HBM和L2的峰值带宽），估测一个命中率，计算带宽是否满足要求。



share memory的瓶颈，每个thread block中的thread分M_frag * N_frag个点（thread tile），然后做gemm，这里通过FFMA和smem数据的读取的比例可以发现，朴素的gemm的瓶颈在smem的访存带宽。这里可以交换计算的顺序，把内积转换为外积，将A和B和C放在寄存器中，本质上使用高速缓存来换速度。内部分块采用8*8，这里是考虑到单线程内的寄存器限制。



double buffer。当使用外积来计算thread tile中的gemm，使用的寄存器数量就很多，一个SM能调度的warp数量就减少了，就不能通过warp切换来掩盖延迟，因为我们之前的计算都是按照满流水的延迟来计算的。这个时候就要提升单个warp内的指令级并行度。通过把从GM到SM，以及SM到REG的过程使用双倍的存储，将计算和访存流水起来，提高并行度。

```c++
GEMM: M, N, K

Shared_Memory:  A_tile[2], B_tile[2]
Register:       A_frag[2], B_frag[2], C_frag
Register:       A_ldg_buffer, B_ldg_buffer

// load 1'st tile to shared memory
load_tile(A_ldg_buffer)
load_tile(B_ldg_buffer)
A_tile[0].store_tile(A_ldg_buffer)
B_tile[0].store_tile(B_ldg_buffer)

// double buffer index
tile_load_idx = 0
tile_store_idx = 1

C_frag = {0, 0, ..., 0}

// K-loop
for (k_iter = K/K_tile - 1; k_iter > 0; --k_iter) {
    for (i = 0; i < K_tile; ++i) {
        // store tile to shared memory
        if (i == K_tile-1) {
            A_tile[tile_store_idx].store_tile(A_ldg_buffer)
            B_tile[tile_store_idx].store_tile(B_ldg_buffer)
            tile_store_idx ^= 1
            tile_load_idx ^= 1
        }

        // load next fragment to register
        A_frag[(i+1) % 2].load_fragment(A_tile[tile_load_idx][(i+1) % K_tile])
        B_frag[(i+1) % 2].load_fragment(B_tile[tile_load_idx][(i+1) % K_tile])

        // load tile from global memory
        if (i == 0) {
            load_tile(A_ldg_buffer)
            load_tile(B_ldg_buffer)
        }

        ffma(C_frag, A_frag[i % 2], B_frag[i % 2])
    }
}

// FFMA for the last tile
for (i = 0; i < K_tile; ++i) {
        if (i < K_tile-1) {
        // load next fragment to register
        A_frag[(i+1) % 2].load_fragment(A_tile[tile_load_idx][(i+1) % K_tile])
        B_frag[(i+1) % 2].load_fragment(B_tile[tile_load_idx][(i+1) % K_tile])
    }

    ffma(C_frag, A_frag[i % 2], B_frag[i % 2])
}

// store C_tile to global memory
C_frag.store(...)
```



<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230910211234478.png" alt="image-20230910211234478" style="zoom:50%;" />

