GPU是一个Processing Unit，PU就需要ISA

>The instruction set consists of addressing modes, instructions, native data types, registers, memory architecture, interrupt, and exception handling, and external I/O.

显卡的功能：

- 显示输出 DP
- GFX 2D 3D
- VPU 视频codec处理

每一个CUDA处理器有一个完全流水的逻辑计算单元

GPU流水线：

- 顶点运算
- 像素运算

unified shader统一vertex和pixel shader



GPU计算的特点

- 特有的并行性
  - 顶点、片元、像素处理过程独立，无控制相关
- 图形渲染程序结构简单
  - 基本无分支跳转
- 数据访问模式简单，可预测性好
  - 连接地址读写访问
  - 便于延时隐藏

GPU设计特点

- 深度流水技术

- SIMT技术
  - 更高效的多线程切换
  - 多物理寄存器堆