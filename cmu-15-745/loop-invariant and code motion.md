- 循环不变量的计算
- 代码提升

### 归纳变量的优化

#### **如何找到循环**

重要概念：

- **Dominance**：x主导w，当且仅当w到达w的所有路径都会先到达x

- **回边**：A back edge is an arc t->h whose head h dominates its tail t
- **nature loop**：The natural loop of a back edge t->h is the smallest set of nodes that
  includes t and h, and has no predecessors outside the set, except for the
  predecessors of the header h.

算法：

1. 找到图中的Dominance关系

   - 形成一个数据流分析问题

   <img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230302094833119.png" alt="image-20230302094833119" style="zoom:30%;" />

2. 识别回边

3. 根据回边找到循环

#### 归纳变量优化的步骤

**basic induction variable**：a variable X whose only definitions within the loop are assignments of the form:$X = X+c \ or \ X = X-c,$
where c is either a constant or a loop-invariant variable. (e.g., i)

**induction variable**：

- a basic induction variable B, or

- a variable defined once within the loop, whose value is a linear functionof some basic induction variable at the time of the definition:

  $A = c1 * B + c2$ (e.g., t1, t2)

**FAMILY of a basic induction variable B**：the set of induction variables A such that each time A is assigned in the loop,
the value of A is a linear function of B.

优化的方式：

1. 强度削弱：将基础归纳变量B的FAMILY归纳变量进行一下变换

   - 创建一个新变量A‘
   - 在preheader初始化A‘
   - 跟踪B的更新
   - 添加A的赋值语句

   <img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230302095752968.png" alt="image-20230302095752968" style="zoom:50%;" />

2. 优化非归纳变量

   - 复制传播
   - 死代码消除

3. 优化基础归纳变量

   - <img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230302095944955.png" alt="image-20230302095944955" style="zoom:50%;" />

#### 寻找基础归纳变量FAMILY的方式

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230302100208839.png" alt="image-20230302100208839" style="zoom:50%;" />

例子：

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230302100219473.png" alt="image-20230302100219473" style="zoom:50%;" />

### 代码提升

**循环不变量**：操作数都定义在循环外面，或者本身就是不变量

**算法1**：循环不变量的检测

- 计算到达定值
- 
- 如果所有到达表达式A=B+C的对于B和C的定义都来自于循环外部
- 循环上述步骤，all reaching definitions of B are outside the loop OR there is ==exactly one reaching definition for B== and it is from a loop-invariant statement inside the loop
- 直到循环不变表达式的集合不再变化

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230307145459875.png" alt="image-20230307145459875" style="zoom:80%;" />