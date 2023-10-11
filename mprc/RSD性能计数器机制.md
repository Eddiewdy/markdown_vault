**TestMain.cpp**

负责准备运行hex代码，控制verilator模拟器的行为。Verilator 会读取和解析硬件设计的源代码，Verilator 通常会为硬件设计的每一个模块生成一个 C++ 类。`VMain_Zynq_Wrapper` 就是这个过程生成的一个类，它模拟了在硬件源代码中对应的 `Main_Zynq_Wrapper` 模块。

这个类包含了一些公有成员，对应于硬件模块的输入、输出和内部信号。在TestMain.cpp中访问这些成员来控制模拟器的行为，例如设置输入信号的值，获取输出信号的值，开始和停止模拟等。



得到主存指针和大小，加载16进制的源代码到主存中。每一行的内容会被切割成4个字符串，每个字符串代表一个**32**位的十六进制数（**8**个十六进制字符），然后将这些十六进制数转化为无符号整数并存储到内存数组中。

```c++
size_t mainMemWordSize = sizeof(top->Main_Zynq_Wrapper->main->memory->body->body__DOT__ram->array) / sizeof(uint32_t);
uint32_t* mainMem = reinterpret_cast<uint32_t*>(&top->Main_Zynq_Wrapper->main->memory->body->body__DOT__ram->array);

loadHexFile(codeFileName, mainMem, 0, true);
```





模拟运行过程：运行一个while循环，不断调用`VMain_Zynq_Wrapper::eval()` ，直到Verilator收到了完成的信号。`VMain_Zynq_Wrapper::eval()` 方法用于模拟硬件设计在一个时钟周期内的行为。

```c++
while (!Verilated::gotFinish()) {
            // クロック更新
            top->clk_p = !top->clk_p;
            top->clk_n = !top->clk_n;
            top->eval();    // 評価
    ...
```



得到dubug寄存器状态。在`VerilatorHelper.h`定义了一个结构体`DebugRegister`对应verilog中的的Debug寄存器，使用宏进行数据的传输，复制到`DebugRegister`结构体

```c++
GetDebugRegister(&debugRegister, top);
```

```c++
struct DebugRegister{

    // DebugRegister of each stage
    NextPCStageDebugRegister npReg[FETCH_WIDTH];
    FetchStageDebugRegister   ifReg[FETCH_WIDTH];
    PreDecodeStageDebugRegister pdReg[DECODE_WIDTH];
    DecodeStageDebugRegister    idReg[DECODE_WIDTH];
    RenameStageDebugRegister    rnReg[RENAME_WIDTH];
    DispatchStageDebugRegister  dsReg[DISPATCH_WIDTH];
    ...
```



`DebugRegister`包括两个部分：

- 流水线每个阶段的状态寄存器，用于进行检测流水线的状态
- 性能寄存器，记录处理器的性能数据



**实现DebugRegister的verilog部分**

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230604223759692.png" alt="image-20230604223759692" style="zoom:50%;" />

**DebugIF.sv**：声明每个流水线阶段的debug寄存器，同时进行modport接口的声明

**DebugTypes.sv**：定义每个debug寄存器的结构体

**PerformanceCounter.sv**：性能计数器的时序逻辑

**PerformanceCounterIF.sv**：跟DebugIF.sv类似，但是描述的是性能计数器

> Modport 是一种用于在接口中定义不同角色的方法，它提供了一种方式来约束接口中信号的访问权限。Modport 可以为接口定义多个角色，并为每个角色定义不同的访问权限。它在复杂的设计中非常有用，特别是在处理那些有多个模块需要使用同一接口，但各个模块对接口的使用方式又不同的情况。
>
> 
>
> `modport` 是 SystemVerilog 中的一个关键字，用于在接口(`interface`)中声明模块端口列表（即输入/输出信号的集合）和它们的方向。`modport` 使得接口更具有灵活性，因为它允许不同的模块以不同的方式使用相同的接口。
>
> 例如，一个接口可能包含了多个信号，但并非所有的模块都需要使用全部的信号。`modport` 可以用来声明这些模块需要使用的特定信号集合，以及这些信号的方向（输入/输出）。

rsd/Processor/Project/VivadoSim/Src/LoadStoreUnit/StoreCommitter.sv

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230604225447504.png" alt="image-20230604225447504" style="zoom:50%;" />

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230604225511095.png" alt="image-20230604225511095" style="zoom:50%;" />

PerformanceCounter.sv

```verilog
// Copyright 2019- RSD contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.

`ifndef RSD_DISABLE_PERFORMANCE_COUNTER

import BasicTypes::*;
import DebugTypes::*;

module PerformanceCounter (
    // 两个modport接口实例
    PerformanceCounterIF.PerformanceCounter port,
    DebugIF.PerformanceCounter debug
);
    PerfCounterPath cur, next;
//这一段代码定义了一个顺向触发的时钟，每当 port.clk 的电平从低变高时（上升沿），就会执行这段代码。在这个代码块中，如果 port.rst 为真，cur 将被赋值为0，否则，cur 将被赋值为 next。
    always_ff @(posedge port.clk) begin
        cur <= port.rst ? '0 : next;
    end
 //定义了一段组合逻辑，描述了 next 如何根据当前的性能指标进行更新。这段代码检查了各种可能的性能问题，例如加载未命中（load miss）、存储未命中（store miss）、指令缓存未命中（IC miss）、存储-加载转发失败（store-load forwarding fail）、内存依赖性预测未命中（mem dep pred miss）、分支预测未命中（branch pred miss）以及在解码阶段检测到的分支预测未命中（branch pred miss detected on decode）。如果检测到这些事件，相应的性能指标就会增加。
    always_comb begin
        next = cur;
        for ( int i = 0; i < LOAD_ISSUE_WIDTH; i++ ) begin
            if (port.loadMiss[i]) begin
                next.numLoadMiss++;
            end
        end
        for ( int i = 0; i < STORE_ISSUE_WIDTH; i++ ) begin
            if (port.storeMiss[i]) begin
                next.numStoreMiss++;
            end
        end
        next.numIC_Miss += port.icMiss ? 1 : 0;
        next.numStoreLoadForwardingFail += port.storeLoadForwardingFail ? 1 : 0;
        next.numMemDepPredMiss += port.memDepPredMiss ? 1 : 0;
        next.numBranchPredMiss += port.branchPredMiss ? 1 : 0;
        next.numBranchPredMissDetectedOnDecode += port.branchPredMissDetectedOnDecode ? 1 : 0;

        // 把当前的性能指标导出到了模块的接口。
        port.perfCounter = cur;  // Export current values
`ifndef RSD_DISABLE_DEBUG_REGISTER
        debug.perfCounter = next;    // Export next values for updating registers in debug
`endif
    end
    

endmodule : PerformanceCounter

`else

module PerformanceCounter (
    PerformanceCounterIF.PerformanceCounter port
);
    always_comb begin
        port.perfCounter = '0; // Suppressing warning.
    end
endmodule : PerformanceCounter

`endif

```





![image-20230628143511199](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230628143511199.png)
