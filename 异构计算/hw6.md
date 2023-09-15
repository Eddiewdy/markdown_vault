## 作业六

### 实验环境介绍

- Device：Apple m1
- OpenCL版本：1.2
- CU：8
- 频率：1000 MHz
- 全局内存：10922 MB

- 最大工作组数：256

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230330214700630.png" alt="image-20230330214700630" style="zoom:50%;" />

### OpenCL同步方式

在OpenCL中，主机端的同步方式用于确保主机端与设备之间的数据传输和计算操作按照预期顺序进行，从而避免竞态条件和数据一致性问题。以下是OpenCL主机端的几种同步方式：

1. 事件（Event）：在OpenCL中，主机端可以创建事件来同步设备端的操作。主机端可以在启动设备端的计算操作之后创建一个事件，然后在需要等待设备端操作完成时，通过等待该事件来实现同步。例如，可以使用`clWaitForEvents()`函数来等待一个或多个事件的完成。
2. 队列（Command Queue）：主机端可以使用命令队列来管理设备端的操作。主机端将命令（例如数据传输、计算操作等）放入命令队列中，然后通过命令队列来控制这些操作的执行顺序。可以使用`clFinish()`函数来等待命令队列中的所有操作完成，从而实现同步。
3. 栅栏（Barrier）：在设备端的计算操作中，可以使用栅栏来同步工作组（Workgroup）内的工作项（Workitem）之间的执行顺序。栅栏会在所有工作项都到达栅栏位置之前阻塞工作组的执行，从而确保工作组内的所有工作项都在同一时刻执行相同的操作。可以使用`clBarrier()`函数来在设备端的内核函数中插入栅栏。
4. 事件回调（Event Callback）：主机端可以通过事件回调来在设备端操作完成时得到通知，并在回调函数中进行相应的处理。事件回调允许主机端在设备端的操作完成后立即执行其他操作，从而实现异步执行和同步控制。

### OpenCL的事件机制

OpenCL提供了一种事件机制，用于管理并同步异步的计算任务。事件机制允许应用程序在执行OpenCL命令时进行控制和同步，并在计算完成后获取结果或执行后续的操作。

在OpenCL中，事件是由事件对象（event object）来表示的。事件对象是OpenCL运行时为每个异步执行的命令创建的。可以通过调用OpenCL API函数来创建和管理事件对象，如clCreateUserEvent、clEnqueueNDRangeKernel、clEnqueueReadBuffer等。

事件对象有以下几个主要的属性和用途：

1. 事件状态（event status）：事件对象的状态会随着命令的执行而发生变化，如`CL_QUEUED`、`CL_SUBMITTED`、`CL_RUNNING`和`CL_COMPLETE`等。可以通过调用`clGetEventInfo`函数来查询事件对象的状态。
2. 事件等待（event waiting）：可以使用事件对象来等待其他事件的完成。通过在调用OpenCL命令时传递等待事件作为参数，可以使命令在等待事件完成后再执行。这可以用于实现命令之间的依赖关系和同步。
3. 事件回调（event callback）：可以在事件完成后自动触发事件回调函数，用于执行用户定义的操作。事件回调可以在创建事件对象时通过调用clSetEventCallback函数来注册。
4. 事件时间戳（event timestamp）：事件对象可以包含用于记录命令执行的时间戳信息。可以通过调用clGetEventProfilingInfo函数来查询事件对象的时间戳信息，用于性能分析和优化。

通过使用事件机制，OpenCL应用程序可以在异构计算环境中实现高效的并行计算，并确保计算任务的正确顺序和同步。同时，事件机制还可以用于性能优化和调试，以便更好地管理OpenCL命令的执行和完成。

## 代码示例

现在使用一下的代码示例来详细解释一下OpenCL的事件

**Host端代码涉及事件同步的代码**

```c++
// 加载数据到src1MemObj和src2MemObj
ret = clEnqueueWriteBuffer(command_queue, src1MemObj, CL_FALSE, 0, contentLength, pHostBuffer, 0, NULL, &evt1);
ret = clEnqueueWriteBuffer(command_queue, src2MemObj, CL_FALSE, 0, contentLength, pHostBuffer, 1, &evt1, &evt2);
	...
// 等待src1和src2的数据
clWaitForEvents(2, (cl_event[2]){evt1, evt2});
clReleaseEvent(evt1);
clReleaseEvent(evt2);
evt1 = NULL;
evt2 = NULL;

// 第一个kernel进入队列执行
ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, (const size_t[]){contentLength / sizeof(int)}, &maxWorkGroupSize, 0, NULL, &evt1);
	...
// 第二个kernel进入队列执行
ret = clEnqueueNDRangeKernel(command_queue, kernel2, 1, NULL, (const size_t[]){contentLength / sizeof(int)}, &maxWorkGroupSize, 1, &evt1, &evt2);
```

**kernel代码**

```c++
__kernel void kernel1_test(__global int *pDst, __global int *pSrc1, __global int *pSrc2) {
    int index = get_global_id(0);
    pDst[index] = pSrc1[index] + pSrc2[index];
}

__kernel void kernel2_test(__global int *pDst, __global int *pSrc1, __global int *pSrc2) {
    int index = get_global_id(0);
    pDst[index] = pDst[index] * pSrc1[index] - pSrc2[index];
}
```

这段代码使用OpenCL编程语言来执行并行计算任务，并使用事件对象来进行同步操作。下面是对这段代码的事件同步过程的解释：

1. `clEnqueueWriteBuffer`函数被调用两次，分别将数据从主机内存（`pHostBuffer`）写入到两个设备缓冲区对象（`src1MemObj`和`src2MemObj`）。这两个写缓冲区操作被设置为非阻塞的（`CL_FALSE`），因此不会阻塞主机端的执行，并且分别关联了两个事件对象（`evt1`和`evt2`），用于后续的事件同步。
2. `clWaitForEvents`函数被调用，等待前面两个写缓冲区操作完成。这里传入了两个事件对象（`evt1`和`evt2`）作为参数，表示需要等待这两个事件对象关联的写缓冲区操作完成，才能继续执行下面的代码。这样可以确保数据已经成功写入到设备缓冲区中。
3. `clReleaseEvent`函数被调用两次，分别释放了前面两个事件对象（`evt1`和`evt2`）的资源，并将其设置为`NULL`，防止后续误用。
4. `clEnqueueNDRangeKernel`函数被调用两次，分别将两个内核（`kernel`和`kernel2`）加入到命令队列中，以便在设备上执行。这两个内核都关联了事件对象`evt1`，表示需要等待`evt1`关联的写缓冲区操作完成后，才能执行这两个内核。

总的来说，这段代码使用了事件对象来进行同步操作，确保设备缓冲区的数据已经成功写入，并在内核执行时遵循了依赖关系，保证了正确的执行顺序。

### 运行结果

<img src="/Users/yd/Library/Application Support/typora-user-images/image-20230414153530667.png" alt="image-20230414153530667" style="zoom:70%;" />
