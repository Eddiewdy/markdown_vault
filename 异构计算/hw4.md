## 作业四

### 实验环境介绍

- Device：Apple m1
- OpenCL版本：1.2
- CU：8
- 频率：1000 MHz
- 全局内存：10922 MB

- **最大工作组数：256**

![image-20230330214700630](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230330214700630.png)

### 简单矩阵乘法问题

在矩阵乘法的计算过程中，A和B位于全局内存中，由于在计算的时候需要把数据从全局内存搬运过来，而多个work item搬运的数据存在很大程度的重复性，如果不使用数据共享，那么就会导师数据的多次重复搬运，是的数据的复用率很低，我们会花更多的时间在数据的搬运而不是真实的数据计算上。

以下是简单矩阵乘法的运行时间表现

|                  | 1024    | 2048    | 4096     | 8192     |
| ---------------- | ------- | ------- | -------- | -------- |
| 简单32位浮点乘法 | 0.312ms | 0.923ms | 3.478ms  | 36.345ms |
| 简单64位浮点乘法 | 0.397ms | 1.623ms | 18.239ms | 99.884ms |

### 矩阵乘法优化

根据使用局部内存来存储从全局内存搬运过来的数据从而实现组内的数据共享，基本思想是矩阵分块，把一个大矩阵分城多个小矩阵，小矩阵的数据可以暂时存放在局部内存中。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/SouthEast.png" alt="这里写图片描述" style="zoom:50%;" />

其实就是把之前row*col的方式变成了 多个row和col相乘，究其本质还是对应元素相乘再相加。

中心思想是引入work-group分块计算再相加，work-item的大小还是没变为M*N，不同的是在同一个work-group中把矩阵A和B的对应值保存在局部内存中，之后每个work item在这个group中访问这个局部变量速度会相对访问全局内存更快，后面的大小为k的循环访问的也是局部内存，kernal代码如下：

```c++
__kernel void matrix_mult(__global float* A, __global float* B, __global float* C, const int N) {
    const int TILE_SIZE = 16;
    __local float local_A[16][16];
    __local float local_B[16][16];

    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    int num_tiles = N / TILE_SIZE;

    float sum = 0.0f;

    for (int t = 0; t < num_tiles; t++) {
        int tiled_row = local_row;
        int tiled_col = local_col;

        local_A[tiled_row][tiled_col] = A[global_row * N + (t * TILE_SIZE + tiled_col)];
        local_B[tiled_row][tiled_col] = B[(t * TILE_SIZE + tiled_row) * N + global_col];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += local_A[tiled_row][i] * local_B[i][tiled_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[global_row * N + global_col] = sum;
}
```

**barrier**是为了保证数据的一致性从而进行的work item同步操作，等待所有的数据都存到局部内存后在进行tile的矩阵乘法计算，最后等所有tile都计算完之后，将结果存储到C矩阵对应的位置。设置不同的局部内存大小得到的实验结果如下：

| 局部内存块大小 | 8*8      | 16*16    | 32*32      |      |
| -------------- | -------- | -------- | ---------- | ---- |
| 时间（8192）   | 23.590ms | 13.234ms | 设备不支持 |      |

可以看出，随着局部内存大小的改变，计算时间明显减少。最后，将使用内存分块和简单矩阵乘法进行比较，结果如下。

![image-20230330222648632](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230330222648632.png)