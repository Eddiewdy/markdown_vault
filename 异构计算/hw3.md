## 作业三

### 一、基于OpenCL API，编写OpenCL平台查询和设备查询程序。

首先创建Xcode command line项目，链接opencl的library

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230315095215482.png" alt="image-20230315095215482" style="zoom:50%;" />

在文件中导入opencl头文件和库，使用`clGetPlatformIDs`查询可用的平台数量，创建`cl_platform_id`的容器，使用`clGetPlatformIDs`得到平台id。

随后循环访问所有的平台，使用`clGetPlatformInfo`得到平台名字，供应商，opencl版本等信息，使用`clGetDeviceIDs`查询设备数量和设备id，随后使用`clGetDeviceInfo`循环查询所有设备的峰值频率，内存大小等信息。

查询结果如下：

![image-20230322112539152](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322112539152.png)

### 二、计算建立OpenCL编程和运行环境，编译运行矢量加法的程序。

#### 实现CPU向量加法

```c++
void vector_add_cpu(int n, const int *A, const int *B, int *C) {
    for(int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
    return;
}
```

#### 实现CPU+GPU向量加法

opencl的kernal代码如下:

```c++

__kernel void vecadd(__global const int *a, __global const int *b,  __global int *result)
{
    int gid = get_global_id(0);
    result[gid] = a[gid] + b[gid];
}
```

#### 运行时间计算

使用 c++ 的 time.h 库来计算运行时间，分别记录两种实现的向量加法的开始的结束时间，将 clock() 方法放在运行开始前和结束后来记录时间。

![image-20230322113325677](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322113325677.png)

![image-20230322113436979](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322113436979.png)

![image-20230322113510206](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322113510206.png)



最后改变矩阵的大小，测试不同的实验数据，这里选择矩阵大小从 1e3 到 1e10 来记录运行时间结果。

```c++
std::vector<std::vector<double>> result(2, std::vector<double>());
    for(long i = 1e3; i < 1e10; i = i * 10) {
        vector_add_compare(result, i);
    }
    for(auto hw : result) {
        for(auto t : hw) {
            printf("%lf ", t);
        }
        printf("\n");
    }
```

得到运行时间结果，第一行是 CPU+GPU 的结果，第二行是 CPU 的结果

![image-20230322114129802](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322114129802.png)

使用 matplotlib 作图结果如下：

![image-20230322113932953](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230322113932953.png)

可以看出，当数据量不大的时候，由于opencl需要很多准备和初始化的操作，包括数据的导入导出等拷贝工作，所以运行时间比 CPU 长一些，但当数据量增大，可以看出 GPU 的计算优势就体现出来，当数据量为 1e10 的时候，GPU+CPU 的方式相对于单 CPU 时间缩短 6 倍。

#### opencl初始化步骤

最后，总结一下 opencl 开始计算前的初始化步骤

1. 检测当前机器的OpenCL平台信息：使用OpenCL API函数 `clGetPlatformIDs` 获取可用的OpenCL平台数目和其对应的ID。
2. 选择一个OpenCL平台：通过对平台属性进行筛选，选择最适合当前应用程序的平台。可以使用 `clGetPlatformInfo` 函数获取平台属性信息。
3. 创建一个OpenCL设备列表：使用 `clGetDeviceIDs` 函数获取指定平台上的可用设备数目和其对应的ID。
4. 选择一个OpenCL设备：通过对设备属性进行筛选，选择最适合当前应用程序的设备。可以使用 `clGetDeviceInfo` 函数获取设备属性信息。
5. 创建一个OpenCL上下文：使用 `clCreateContext` 函数创建OpenCL上下文，将设备列表作为参数传递给该函数。
6. 创建一个OpenCL命令队列：使用 `clCreateCommandQueue` 函数创建OpenCL命令队列，将上下文和选择的设备作为参数传递给该函数。
7. 创建OpenCL程序对象：使用 `clCreateProgramWithSource` 函数创建OpenCL程序对象，并将OpenCL代码传递给该函数。
8. 编译OpenCL程序对象：使用 `clBuildProgram` 函数编译OpenCL程序对象，生成可执行的二进制代码。
9. 创建OpenCL内核对象：使用 `clCreateKernel` 函数创建OpenCL内核对象，内核对象对应一个OpenCL程序中的内核函数。
10. 准备数据：将需要进行计算的数据从主机内存复制到OpenCL设备内存中，可以使用 `clCreateBuffer` 函数创建OpenCL缓冲区对象，使用 `clEnqueueWriteBuffer` 函数将主机内存中的数据复制到缓冲区中。
11. 执行内核：使用 `clEnqueueNDRangeKernel` 函数将内核函数提交到命令队列中执行。
12. 获取计算结果：使用 `clEnqueueReadBuffer` 函数将计算结果从OpenCL设备内存复制到主机内存中，以便后续处理和输出。
