模块：
module /模块名/ ( input /输入端口名/, output /输出端口名/ ); /语句/ endmodule

模块实例化：

```verilog
module Full_Adder
(
input a,
input b,
input cin,
output cout,
output sum
);
assign {cout, sum} = a + b + cin;
endmodule
Full_Adder adder_1
(
.a (a[0]),
.b (b[0]),
.cin (cin),
.cout (cout[0]),
.sum (sum[0])
);
```

连线定义：wire 连线名

连续赋值：
assign LHS = RHS /RHS赋给LHS/

向量：
wire [/*起始位*/ +: /*位宽*/] /*向量名*/; /*声明，从起始位递增*/
wire [/*起始位*/ +: /*位宽*/] /*向量名*/; /*从起始位递减*/
{/*向量A*/ , /*向量B*/, ...., /*向量X*/} /*拼接多个向量*/
{/*整数*/ {/*向量*/}} /*拼接符的另一种用法*/

符号溢出：两正数相加得到负数或两负数相加得到正数
检测方法：

```verilog
module top_module (
input [7:0] a,
input [7:0] b,
output [7:0] s,
output overflow
);
assign s = a + b;
assign overflow = (a[7]&b[7]&~s[7]) | (~a[7]&~b[7]&s[7]);
//当输入值符号位均为0且输出符号位为1时，或输入值符号位均为1且输出符号为0时，发生符号溢出
endmodule
```



#### 结构化描述方式

通过对器件的调用(实例化)，并使用线网来连接各器件的描述方式。

```verilog
xor x1 (S1, A, B);
xor x2 (Sum, S1, Cin);
and A1 (T3, A, B ); and A2 (T2, B, Cin); and A3 (T1, A, Cin);
or O1 (Cout, T1, T2, T3 );

// 用一位全加器实现两位全加器
module Four_bit_FA (FA, FB, FCin, FSum, FCout ); 
    parameter SIZE = 2;
    input [SIZE:1] FA;
    input [SIZE:1] FB;
    input FCin;
    output [SIZE:1] FSum; output FCout;
    wire FTemp;
    FA_struct FA1(
        .A (FA[1]),
        .B (FB[1]),
        .Cin (FCin) , .Sum (FSum[1]), .Cout (Ftemp)
    );
    FA_struct FA2(
        .A (FA[2]),
        .B (FB[2]), .Cin (FTemp) ,
        .Sum (FSum[2]),
        .Cout (FCount ) );
endmodule
```

#### 数据流描述方式：使用赋值语句

**'timescale 1ns/100ps**：'timescale 是Verilog HDL 提供的预编译处理命令， 1ns 表示时间单位是1ns ，100ps表示 时间精度是100ps。根据该命令，编译工具才可以认知 #2 为2ns。

**assign #2 S1= A ^B;** 时延2ns



#### 行为方式：采用对信号行为级的描述

1、只有寄存器类型的信号才可以在always和initial 语句中进行赋值，类型定义通过reg语句实现。
2、always 语句是一直重复执行，由敏感表(always 语句括号内的变量)中的变量触发。 
3、always 语句从0 时刻开始。
4、在 begin 和 end 之间的语句是顺序执行，属于串行语句。



基数表示法：[size ] 'base value



### RSD

RSD 支持多种高级微架构功能，例如推测式load和store执行、内存依赖预测器 (MDP)、推测性调度和非阻塞缓存。

RSD的高性能和高效率是通过两种方式：**FPGA 友好的推测调度**，**多端口 RAM 的组件。**

前端使用的分支预测器：gshare

发射队列特殊机制：随机选择、指令重放机制

banked RAM：在banked RAM中，多个请求被分发到多个bank，以模拟多端口的RAM。 



指令重放：

- store data array按照指令先后顺序保存所有的没有被提交的store指令
- 一个load在STQ中搜索所有比他更老的已经生成了地址的store指令，选择其中最年轻的指令做数据forward
- 一个store在STQ中搜索所有**完成了**且比他**更年轻**且有**相同内存地址**的load指令，将这条load指令以及其依赖指令重放

