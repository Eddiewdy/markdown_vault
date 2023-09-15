**百度昆仑芯 平台软件研发部**

负责深度学习模型到硬件的高性能训练和推理优化的工作，工作内容包括:

llama-65B模型调优，kunlun2芯片上训练性能达到A100的85% 

- 基于megatron仓库的profiling工具的开发

- 算子层面:rmsnorm、rotary embedding算子的优化等
- 框架层面:torch Linear层的findmax优化等

kunlun2芯片FP16 梯度动态scale方案的设计和调研 

llama-7B模型推理优化，

- 包括框架层面findmax优化、GPTQ量化的实现 megatron、deepspeed框架调研，LLM int8、GPTQ量化算法调研





### **profiling工具：**

修改megatron/model/transformer.py代码，给layer norm、core attention、roatry emb、增加log的打印，反向过程用钩子函数注册register backward hook

![Xnip2023-09-11_13-27-30](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/Xnip2023-09-11_13-27-30.jpg)

### **Linear层的findmax优化：**

profiling发现findmax占比很高，针对kunlun2的硬件计算的特点，kunlun2计算所有的浮点数操作都是先找到float的最大值，然后进行量化到int，随后使用mac脉冲阵列计算int乘法，随后反量化回来。

- 训练优化手段：使用ctx，在前向存下weight的max，给反向做mm用，减少一半的findmax。同时在minibatch中，所有的microbatch公用一套weight，所有可以在一次的minibatch迭代中，用attr保存起来

- 推理优化手段：在llm的增量推理中，会做多次的前向，这里的weight也是公用max的，使用attr保存起来，只在第一次前向是计算max

pending：算子层面:rmsnorm、rotary embedding算子的优化等



### **kunlun2芯片FP16 梯度动态scale方案：**

调研Transformer Engine：使用fp8进行训练，针对transformer中的一些kernel进行了重写，进行了一些针对性的kernel融合

kunlun的方案：kunlun2不支持非规格化数的表示（exp为0，f不为0）为了应对fp16针对很小的数梯度消失，所以在反向过程中，使用flaot32存储梯度，在计算的时候cast到fp16，但是要保证最小值在可以表示的范围内，并在计算完成后检查时候没有梯度消失。

### 算子优化

#### kunlun memory的架构

每个XPU Core内部有一定数量的RF(Register File)，每4个XPU Core构成一个物理的group，共享32KB的local memory和512bit的SIMD算力。64个XPU Core组成一个XPU Cluster，共享256KB SM(Shared Memory)。昆仑2的XPU Core可以直接在SM和寄存器中，也可以通过调用原子指令的接口读写。直接的SM读写可能会造成Core间的数据竞争，因此建议使用原子指令的接口读写。



#### rms norm 前向

~~根据cluster的个数进行划分m行，一个cluster负责连续的多行。scale和bias存储在SM中，cluster内的64个core共享256KB的SM。~~

~~![image-20230907221505620](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230907221505620.png)~~

~~进行软分组，一个core负责跨越ngroup行的32*k列的元素的square sum。~~

一个core负责m*n矩阵的一行，计算均方，考虑根据buf_size的大小来确定迭代次数，每次从GM2LM，LM是一个group内部共享的。使用core_id() % n_times来进行偏移，减少GM channel的LD冲突



一个core内计算squaresum的方法，使用simd load和mul，

#### rms norm 反向

- fp16 kernel（n < 2048）：把x dy dx都用LM存，一行由一个core负责。d_bias和d_scale用SM存，计算完后，在最后做一个沿着m维度的reduce，得到n列的d_bias和d_scale
- fp16 large kernel(n < 4096)：由于一行LM放不下，如果把x dy dx都用SM存一行一个core负责
- Fp16（n > 4096）：一行用多个core进行计算，计算玩dx的reduce部分，用cid=0的core进行汇总
- n=8192优化（m=1024 n=8192）：
  - 一个core负责一整行的计算，减少reduce等待的空闲时间，删除SM计算ruduce逻辑
  - pingpong buffer（GM2LM），做运算，在计算dx的同时，进行GM2LM的访存
  - 对n_iter进行cid偏移，让每8个core错位512B进行计算，减少LD bank conflict
  - 计算dscale，dbias交给reduce sum做，
  - 使用SM存储 scale，var
  

带宽性能3倍提升



![image-20230915102613233](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230915102613233.png)



#### dropout_add_rms_norm

算子融合，一个core负责一行的rms_norm，n_times表示需要迭代的次数（从GM到LM），使用core_id() % n_times来进行偏移，减少GM channel的LD冲突





### flash attention

维护一个局部的最大值和局部的ex的和，根据输入不断更新，进行重计算来缓解HBM带宽的压力。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/Xnip2023-09-11_13-31-47.jpg" alt="Xnip2023-09-11_13-31-47" style="zoom:50%;" />



### TransformerEngine





SM 支持的每 block 寄存器最大数量为 32K 或 64K 个 32bit 寄存器，每个线程最大可使用 255 个 32bit 寄存器，编译器也不会为线程分配更多的寄存器，所以从寄存器的角度来说，每个 SM 至少可以支持 128 或者 256 个线程，block_size 为 128 可以杜绝因寄存器数量导致的启动失败，但是很少的 kernel 可以用到这么多的寄存器，同时 SM 上只同时执行 128 或者 256 个线程，也可能会有潜在的性能问题。但把 block_size 设置为 128，相对于 256 和 512 也没有什么损失，128 作为 block_size 的一个通用值是非常合适的。



确定了 block_size 之后便可以进一步确定 grid_size，也就是确定总的线程数量，对于一般的 elementwise kernel 来说，总的线程数量应不大于总的 element 数量，也就是一个线程至少处理一个 element，同时 grid_size 也有上限，为 Maximum x-dimension of a grid of thread blocks ，目前在主流架构上都是 2^31 - 1，对于很多情况都是足够大的值。

GPU 一次可以调度 SM 数量 * 每个 SM 最大 block 数个 block



