![img](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/v2-a3269882e4215e6fe2d2282b893d0b83_1440w.webp)

分布式数据并行多卡训练的时候，BatchNorm 的计算过程（统计均值和方差）在进程之间是独立的，也就是每个进程只能看到本地 GlobalBatchSize / NumGpu 大小的数据。

但是对于一些模型占用显存很大，导致可以上的 batch size 很小这类任务来说，分布式训练的时候就需要用 SyncBatchNorm 来使得统计量更加的准确。

前向步骤

- 每个GPU先单独计算各自本地数据 `X_i` 对应均值和方差（`mean_i` 和 `var_i`） 
  -  welford算法，只需要对数据集进行单次遍历，同时不会出现上溢
  - <img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230526190037987.png" alt="image-20230526190037987" style="zoom:50%;" />
- **kernel 执行的第一步**，所有线程处理完自己所负责的数据，然后同步一下
- **kernel 执行的第二步**，每个 warp 内的线程合并均值和方差，通过 warp 级的同步元语库函数 `__shfl_xor_sync` 来实现 warp 内线程结果的合并。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/v2-c86b6b0eed9c8f7ebccd38522debdb4f_1440w.webp" alt="img" style="zoom:67%;" />

**kernel 执行的最后一步是**，上面每个 warp 内结果合并完，会做一次全局的线程同步。之后再将所有 warp 的结果合并就得到该 thread block 所负责计算的通道均值和方差了。

#### 前向第二步，GPU之间同步均值和方差

通过集合通信操作 `AllGather` 让每个 GPU 上的进程都拿到所有 GPU 上的均值和方差，最后就是每个GPU内计算得到全局的均值和方差，同时更新 `running_mean` 和 `running_var`

### 前向第三步，计算 SyncBN 的输出

最后这一步就一个常规的batchnorm操作，对输入 x 做 normalize 操作得到输出，cuda kernel 就是一个 eltwise 的操作，因为不需要计算均值和方差了。这里就不展开了，有兴趣的读者可以看文末的参考链接，去阅读torch的源码，也可以学习一下对于 `NHWC`格式的 cuda kernel 是如何实现的。

### **反向计算流程**

每个GPU都计算出本地对应的 `weight_grad` ，`bias_grad` ，`sum_dy` 和 `sum_dy_xmu`，具体CUDA kernel 实现思路和前向第一步类似，这里就不展开了，有兴趣可以去阅读源码。

由于分布式数据并行下，权值的梯度会自动做全局同步，所以 SyncBN 就不需要管权值梯度的跨 GPU 的同步。

而对于`sum_dy` 和 `sum_dy_xmu`，则通过集合通信操作 `AllReduce` 将所有GPU上的结果累加，使得每个GPU上持有全局累加的结果。

最后每个 GPU 根据上面的计算公式计算本地输入x对应的梯度，但是需要注意的是，由于 `sum_dy` 和 `sum_dy_xmu`是跨 GPU 全局累加的结果，所以上面公式中的 `rc=B*H*W`要改为 `rc=B*H*W*num_gpu` 。该 CUDA kernel 的实现，根据上述公式，也是一个 eltiwse 的操作，细节可以去阅读torch源码。