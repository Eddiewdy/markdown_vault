### 卷积计算优化

### 实验环境介绍

- Device：Apple m1
- GPU：RTX 3060

### 卷积计算介绍

卷积计算在图像处理和信号处理中是非常常见的操作。在卷积计算中，我们将一个函数（在图像处理中，这通常是一个小的、固定大小的矩阵，被称为“卷积核”或“滤波器”）应用到另一个函数上（例如，一个图像）。

在二维图像处理中，卷积计算的基本公式如下：

```css
(I * K)[i, j] = sum(I[i - m, j - n] * K[m, n]), for all m, n in bounds of K
```

在 CUDA 中加速卷积计算主要依赖于以下几个技术：

1. **并行计算**：在 CUDA 编程模型中，可以利用 GPU 的大量并行处理核心来并行执行卷积计算，从而显著提高计算速度。具体而言，你可以把每个像素点的卷积计算分配给一个 CUDA 线程（或一组线程），然后让这些线程并行执行。
2. **共享内存**：CUDA 设备上的共享内存是一种位于每个线程块内部的高速缓存。通过将卷积核以及每个线程块要处理的输入图像的部分载入共享内存，可以减少从全局内存中读取数据的时间，从而加速卷积计算。
3. **循环展开和指令级并行**：在编写 CUDA kernel 时，可以通过展开内部循环并利用指令级并行（ILP）来进一步提高卷积计算的速度。
4. **利用纹理内存和常量内存**：纹理内存和常量内存是两种特殊类型的内存，适用于某些特定的访问模式。例如，纹理内存对于二维空间局部性的访问模式有很好的性能，而常量内存则适用于所有线程都读取相同数据的情况（如卷积核）。在卷积计算中，可以考虑将卷积核放入常量内存，将输入图像放入纹理内存，从而优化内存访问性能。

### cpu版本

```c++
std::vector<std::vector<float>> Conv2D(const std::vector<std::vector<float>>& input,
                                       const std::vector<std::vector<float>>& kernel) {
    int height = input.size();
    int width = input[0].size();
    int ksize = kernel.size();

    std::vector<std::vector<float>> output(height - ksize + 1, std::vector<float>(width - ksize + 1, 0.0f));

    for (int i = 0; i <= height - ksize; ++i) {
        for (int j = 0; j <= width - ksize; ++j) {
            for (int di = 0; di < ksize; ++di) {
                for (int dj = 0; dj < ksize; ++dj) {
                    output[i][j] += input[i + di][j + dj] * kernel[di][dj];
                }
            }
        }
    }

    return output;
}
```

上面是一个最普通的卷积计算版本，在不同尺寸下的到的计算结果如下：
<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527190940444.png" alt="image-20230527190940444" style="zoom:50%;" />

可以看出表现比较一般，首先采取编译器O3进行优化，的到结果如下：

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527191016928.png" alt="image-20230527191016928" style="zoom:50%;" />

可以看出性能得到大幅度提升，编译器为我们做了循环展开等操作，提升卷积计算的并行性。进一步，我们可以采取SIMD命令进行优化

```c++
void Conv2D_NEON(const std::vector<float>& input,
                 const std::vector<float>& kernel,
                 std::vector<float>& output,
                 const int height,
                 const int width,
                 const int ksize) {
    const int output_size = height - ksize + 1;
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            float32x4_t sum = vdupq_n_f32(0.0f);  // 用于计算的NEON向量
            for (int di = 0; di < ksize; ++di) {
                for (int dj = 0; dj < ksize; ++dj) {
                    float32x4_t a = vld1q_f32(&input[(i + di) * width + j + dj]);  // 加载输入向量
                    float32x4_t b = vld1q_f32(&kernel[di * ksize + dj]);  // 加载卷积核向量
                    sum = vmlaq_f32(sum, a, b);  // 执行向量乘法和加法
                }
            }
            // 将NEON向量的所有元素加起来，得到最后的输出值
            output[i * output_size + j] = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1) +
                                          vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
        }
    }
}
```

这里我们根据硬件平台，使用arm的neon指令集，将内部的循环进行并行。但是由于我们一开始的测试中卷积核较小，的到的并行度不高，这里只能的到较小的加速，如果把kernel变大，这里使用4*4的kernel就可以获得比普通做法和编译器优化更好的效果：

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527194010506.png" alt="image-20230527194010506" style="zoom:50%;" />

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527194030354.png" alt="image-20230527194030354" style="zoom:50%;" />

最后我们把cpu上的一系列结果进行比较，作图如下

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/output.png" alt="output" style="zoom:72%;" />

可以看出，当kernel比较大的时候，使用simd能获得更不错的效果。

### gpu版本

随着GPU技术的进步，它已经在数据处理和并行计算等领域中取得了重大进展。卷积核的优化在计算机视觉领域具有重要价值。为了探索GPU的性能和提高算法的效率，我们进行了一系列的卷积核实验，从最普通的kernel开始，尝试多种优化技术。

#### 二、实验方法与过程

1. **基础卷积核实验**：我们首先在GPU上运行最基本的卷积核实现，将卷积核内的计算映射到GPU的线程，无特殊优化措施。通过这个步骤，我们建立了一个基线，以评估后续优化方法的性能提升。
2. **将Kernel放在寄存器中**：卷积核数据在内存中移动的过程中，会消耗大量的时间。为了降低内存访问的时间开销，我们试图将卷积核放入GPU的寄存器中。寄存器是GPU中最快的存储设备，因此，这个优化策略的目的是减少内存访问的延迟。
3. **使用共享内存**：共享内存是一种在同一块GPU中的所有线程之间共享的内存，我们试图通过将输入数据加载到共享内存中，进一步提升运行速度。这种优化策略的主要目标是减少访问全局内存的开销，因为相对于全局内存，访问共享内存的速度更快。
4. **循环展开**：我们还试图通过循环展开（Loop Unrolling）来提升性能。循环展开是一种常用的优化技巧，通过减少循环迭代的次数来减少计算的时间。在GPU编程中，循环展开可以帮助减少指令的分支和提高指令的并行性。

实验代码如下：

```c++
// naive版本
__global__ void conv2d_naive(float* deviceInput, float* deviceOutput, float* deviceKernel, int width, int height, int ksize) {
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    float result = 0.0f;
    int kHalf = ksize / 2;

    if (xIndex < width && yIndex < height) {
        for (int i = -kHalf; i <= kHalf; ++i) {
            for (int j = -kHalf; j <= kHalf; ++j) {
                int xIndexKernel = i + kHalf;
                int yIndexKernel = j + kHalf;

                int xIndexImage = xIndex + i;
                int yIndexImage = yIndex + j;

                // Boundary check
                if (xIndexImage >= 0 && xIndexImage < width && yIndexImage >= 0 && yIndexImage < height) {
                    result += deviceInput[yIndexImage * width + xIndexImage] * deviceKernel[yIndexKernel * ksize + xIndexKernel];
                }
            }
        }

        deviceOutput[yIndex * width + xIndex] = result;
    }
}
```

```c++
// 共享内存版本
__global__ void conv2d(float* deviceInput, float* deviceOutput, float* deviceKernel, int width, int height, int ksize) {
    int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

    int sharedInputSize = (blockDim.x + ksize - 1) * (blockDim.y + ksize - 1);
    int sharedKernelSize = ksize * ksize;

    __shared__ float sharedMemory[1200];
    float* sharedInput = sharedMemory;
    float* sharedKernel = &sharedMemory[sharedInputSize];

    if (threadIdx.x < ksize && threadIdx.y < ksize) {
        sharedKernel[threadIdx.y * ksize + threadIdx.x] = deviceKernel[threadIdx.y * ksize + threadIdx.x];
    }

    int sharedXIndex = threadIdx.x + ksize / 2;
    int sharedYIndex = threadIdx.y + ksize / 2;

    if (xIndex < width && yIndex < height) {
        sharedInput[sharedYIndex * blockDim.x + sharedXIndex] = deviceInput[yIndex * width + xIndex];
    }

    __syncthreads();

    float result = 0.0f;

    if (xIndex < width && yIndex < height) {
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                result += sharedInput[(sharedYIndex + i) * blockDim.x + sharedXIndex + j] * sharedKernel[i * ksize + j];
            }
        }

        deviceOutput[yIndex * width + xIndex] = result;
    }
}
```

```c++
// 将kernel放在寄存器中
__constant__ float conv_kernel [3][3] {
    {1.0,-1.0, 1.0},
    {-1.0,1.0, -1.0},
    {-1.0,1.0, -1.0}
};
```

随后在不同的数据量下进行对比，作图如下：

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527202040193.png" alt="image-20230527202040193" style="zoom:50%;" />

naive的数据如上

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230527202106359.png" alt="image-20230527202106359" style="zoom:50%;" />

shared memory的数据如上![output2](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/output2.png)

可以看出shared memory可以带来较大的提升，提升很不错的并行性，同时将kernel放在register中也带来了一定的性能提升，最后把gpu的曲线和cpu一起进行比较

![output3](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/output3.png)

### 总结

gpu的优化方式：

- **将Kernel放在寄存器中**：卷积核数据在内存中移动的过程中，会消耗大量的时间。为了降低内存访问的时间开销，我们试图将卷积核放入GPU的寄存器中。寄存器是GPU中最快的存储设备，因此，这个优化策略的目的是减少内存访问的延迟。
- **使用共享内存**：共享内存是一种在同一块GPU中的所有线程之间共享的内存，我们试图通过将输入数据加载到共享内存中，进一步提升运行速度。这种优化策略的主要目标是减少访问全局内存的开销，因为相对于全局内存，访问共享内存的速度更快。
- **循环展开**：我们还试图通过循环展开（Loop Unrolling）来提升性能。循环展开是一种常用的优化技巧，通过减少循环迭代的次数来减少计算的时间。在GPU编程中，循环展开可以帮助减少指令的分支和提高指令的并行性。

cpu的优化方式：

- **simd**
- **编译器加速**

最终来看，gpu在大规模的数据下还是有明显的优势。
