## MIPS Cache

P6600核心包含三个缓存：L1指令缓存、L1数据缓存和共享的L2缓存。

L1指令缓存通过两个64位数据路径（datapath）连接到指令获取单元（IFU），允许每个周期最多进行四次指令获取。L1数据缓存包含两个64位数据路径（datapath），允许每个周期最多进行两次数据读/写操作。L2缓存嵌入在一致性管理器（CM2）内，并通过可配置的128位或256位OCP接口与外部存储器进行通信。

![image-20230915094053237](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230915094053237.png)

### L1 ICache

L1指令缓存由三个主要部分组成：**标签、数据和路径选择**。它使用**VIPT**。每个Set有4个way，并使用一个单一的有效位来标识整个32字节的cache line是否在缓存中。缓存还使用最近最少使用（LRU）算法来决定替换哪一路的数据。

![image-20230915094651323](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230915094651323.png)

#### Cache Aliasing

在P6600核心中，指令缓存是**VIPT**。由于单一路的Cache的大小大于最小的TLB页面大小，因此存在虚拟别名（virtual aliasing）的可能性。这意味着如果一个物理地址可能存在于缓存内的多个索引中。**这种虚拟别名只在缓存大小大于16 KB时才会出现。**

##### 硬件开启方式

P6600核心的**Config7IAR**位总是被设置，以指示存在指令缓存虚拟别名硬件。核心允许一个物理地址通过不同的虚拟地址访问时位于多个索引中。当由于CACHE或SYNCI指令发出无效请求时，核心会逐个检查给定物理地址可能的所有别名位置。

硬件可以通过Config7IVAD位进行启用和禁用。清除此位时，用于消除指令缓存虚拟别名的硬件将被启用，这样虚拟别名就会在硬件中被管理，无需软件干预。

#### 预编码和奇偶校验

P6600核心的L1指令缓存有一些额外的预编码（precode）位，这些位让issue单元能快速检测到分支和跳转指令，这些预编码位标明在一个64位的issue大小内转移指令的类型和位置。

L1指令缓存还包括16个奇偶校验位，用于128位数据的每个字节，以及标签数组对每个标签有5个奇偶校验位，用于预编码字段和物理标签、锁定和有效位。LRU数组没有任何奇偶校验。



#### 替换策略

P6600核心的L1指令缓存使用LRU策略来管理替换，但会排除任何已锁定的路。

在出现miss时，选定行的标签和路径选择条目的锁定和LRU位可用于确定将被选择的路径。LRU字段在路径选择数组中会按照以下规则更新：

- 缓存命中：与之关联的路径更新为最近使用的，其他路径的相对顺序不变。
- 缓存refill：被填充的路径更新为最近使用的。
- CACHE指令：LRU位的更新取决于要执行的操作类型，包括各种情况如索引无效、索引加载标签等。

如果所有路径都是有效的，那么锁定的路径都将从LRU替换的中排除。对于未锁定的路径，使用LRU位来识别最近最少使用的路径，并选择该路径进行替换。

#### Cache Line Locking

...

#### Software Cache Management

在P6600架构中，L1指令缓存并不完全是Coherent，因此有时需要操作系统进行干预。CACHE指令是这种干预的基础构建块，用于**正确处理DMA数据和缓存初始化**。CACHE指令在写入指令时也有作用。

在P6600架构中，CACHE指令以`cache op, addr`的形式编写，其中`addr`是一个地址格式，与加载/存储指令相同。CACHE是特权指令，只能在内核模式下运行。但是，SYNCI指令可以在用户模式下工作。可以简化某些缓存管理任务。CACHE和SYNCI指令为系统提供了高度的控制能力。

CACHE指令的17:16位选择要处理的缓存：00：L1 I-cache，01：L1 D-cache，10：reserved， 11：L2 cache

![image-20230915100450657](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230915100450657.png)

CACHE指令有三种变体，它们在如何选择要操作的缓存行方面有所不同：

- 命中型缓存操作（Hit-type）：提供一个地址（就像加载/存储一样），该地址在缓存中查找。如果该位置在缓存中（即“命中”），则在该行上执行缓存操作。如果该位置不在缓存中，什么也不会发生。
  
- 地址型缓存操作（Address-type）：提供某个内存数据的地址，该地址就像一个缓存访问一样被处理 - 如果缓存之前是无效的，数据将从内存中获取。
  
- 索引型缓存操作（Index-type）：使用地址的尽可能多的低位来选择缓存行中的字节，然后选择缓存行内的一路。需要知道Config1寄存器中缓存的大小，以准确了解字段边界的位置。

MIPS64规范允许CPU设计者选择是从虚拟地址还是物理地址获取索引。对于索引型操作，MIPS推荐使用kseg0地址，这样虚拟地址和物理地址是相同的，也避免了缓存别名（aliasing）的可能。

>kseg0地址：
>
>在 MIPS 架构中，`kseg0` 是一段虚拟地址空间，通常用于内核模式操作。它的地址范围从 `0x80000000` 到 `0x9FFFFFFF`。`kseg0` 区域的主要特点是其地址会被直接映射到物理地址，通过简单地减去 `0x80000000` 的偏移量。换句话说，`kseg0` 地址 `0x80000000` 对应于物理地址 `0x00000000`，`kseg0` 地址 `0x80000010` 对应于物理地址 `0x00000010`，依此类推。
>
>该段内存通常用于存储内核代码和数据，并且通常是被缓存的，这意味着访问这段内存的速度通常比直接访问物理内存要快。
>
>以下是 `kseg0` 的一些重要特点：
>
>- **直接映射**：不需要 TLB（Translation Lookaside Buffer）转换。
>  
>- **缓存**：通常情况下，通过 `kseg0` 访问的内存是缓存的，除非缓存被明确禁用。
>
>- **仅限内核模式**：通常只有在 MIPS 的内核模式下才能访问 `kseg0`。
>
>- **无需 MMU**：因为 `kseg0` 是直接映射的，所以即使 MMU（Memory Management Unit）被禁用，也可以访问它。
>
>这样的设计使得操作系统内核可以快速、简单地访问物理内存，同时还可以享受缓存带来的性能优势。然而，因为这是一个直接映射，所以只能映射一个有限大小（通常是 512MB）的物理内存区域。对于更大的物理内存支持，通常会使用其他机制，如更复杂的虚拟内存映射。
>
>`kseg0` 和它的非缓存版本 `kseg1` 是 MIPS 架构的标准一部分，大多数 MIPS 实现都会有这两个区段。

#### CP0（Coprocessor 0） Register Interface

不同的CP0寄存器用于指令缓存操作。具体地，这些CP0寄存器包括Config1、CacheErr、ITagLo、ITagHi、IDataLo和IDataHi。

例如在Config1寄存器中：

- IS字段（位24:22）指示指令缓存每路有多少个集合。P6600的L1指令缓存支持每路256个集合，用于配置32 KB的缓存，或每路512个集合，用于配置64 KB的缓存。
- IL字段（位21:19）指示指令缓存的行大小。P6600的L1指令缓存支持固定的32字节行大小，该字段的默认值为4。
- IA字段（位18:16）指示指令缓存的集合关联度。P6600的L1指令缓存固定为4路集合关联，该字段的默认值为3。

其他的就不一一介绍，等到需要使用的时候再查

#### Cache初始化

关于初始化部分的代码进行了简单的理解，下面的部分代码的解释：



在使用之前，缓存必须被初始化为一个已知的状态，即所有的缓存项都必须被置为无效。这个代码可以初始化缓存，找到缓存Set的总数，然后使用缓存指令遍历缓存Set，使每个缓存Set无效（invalidated）。

```assembly
LEAF (init_icache)
// For this Core there is always an L1 instuction cache // The IS field determines how many sets there are
// IS = 2 there are 256 sets
// IS = 3 there are 512 sets
// $11 set to line size, will be used to increment through the cache tags 
li $11, 32 # Line size is always 32 bytes.
```



该指令缓存的行大小总是32字节，有4个way，并且大小可以是32 KB或64 KB。通过Config1寄存器的IS字段（每路的set数）来确定缓存的大小，该字段可以有两个值：0x2表示32 KB缓存，0x3表示64 KB缓存。

```assembly
mfc0  $10, $16, 1 # Read C0_Config1 mfc0：Move From Coprocessor 0
# 这一行代码从 C0_Config1 寄存器中读取值并存放在 $10 寄存器中。
ext   $12, $10, 22, 3 # Extract IS
# 这条指令从 $10 寄存器中的第 22 位开始，提取 3 位，并将结果存储在 $12 寄存器中。（IS）
li    $14, 2 # Used to test against
```

首先进行一系列检查和设置，以确定缓存的大小和迭代值，如果检查结果为真，代码将使用分支延迟槽（总是会被执行）来为32 KB缓存设置集合迭代值为256，然后跳转到`Isets_done`。如果检查结果为假，代码确定缓存大小为64 KB。此时，代码在分支延迟槽中仍然将迭代值设置为256，但随后会跳过并再次将其设置为512，以适应64 KB的缓存。

```assembly
beq $14, $12, Isets_done # if IS = 2
li $12, 256 # sets = 256
li $12, 512 # else sets = 512 Skipped if branch taken
```



```assembly
Isets_done:
lui   $14, 0x8000 # Get a KSeg0 address for cacheops
# clear the lock bit, valid bit, and the LRF bit
mtc0  $0, $28     # Clear C0_ITagLo to invalidate entry

next_icache_tag:
cache 0x8, 0($14) # Index Store tag Cache op

# Loop maintenance
add   $12, -1                 # Decrement set counter
bne   $12, $0, next_icache_tag # Done yet?
add   $14, $11 # Increment line address by line size

done_icache:
# Modify return address to kseg0 which is cacheable
# (for code linked in kseg1.)
# Debugging consideration; leave commented during debugging
# ins r31_return_addr, $0, 29, 1
jr  r31_return_addr
nop
```

>`lui` 指令用于将一个 16 位的立即数载入寄存器的高 16 位，低 16 位被清零。

`lui`指令将0x8000加载到上16位，并清除GPR 14寄存器的低16位。

清除标签寄存器执行两个功能：

- 将名为PTagLo的物理标签地址设置为0，这确保物理地址位高位被清零

- 清除该set的有效位，这确保该set是空的，并且可以根据需要填充。

随后，代码使用Move to Coprocessor Zero（MTC0）指令清除tag寄存器，从而使set为空并可根据需要进行填充。

>在 MIPS 的系统控制协处理器（Coprocessor 0）中，寄存器 `$28` 通常对应于 `C0_TagLo` 或 `C0_ITagLo` 寄存器（取决于具体的 MIPS 架构版本和实现），这些寄存器通常用于缓存操作。这里的 `mtc0 $0, $28` 指令的意图是清除 `C0_ITagLo` 寄存器，以使一个或多个缓存项无效。





然后，代码使用CACHE指令进行`Index Store Tag`操作，然后通过增加虚拟地址来初始化缓存的下一个set。最后，代码进行循环，逐一减少循环计数器，并检查是否已经到达零。如果没有，它将回到tag继续执行。

`cache`指令在L1 指令缓存上使用`Index Store Tag`来操作，因此操作字段的编码值为0x8。前两位是L1指令缓存的2'b 00，`Index Store Tag`的操作代码在二位、三位和四位中编码为3'b 010。

>`Index Store Tag` 是一种索引型缓存操作，它直接操作特定缓存行的标签（Tag）。在缓存系统中，标签用于标识缓存行中数据的来源地址。该操作使用地址的低位位（根据缓存大小和结构）来确定要操作的缓存行索引，然后将该缓存行的标签存储在给定的地址。

```assembly
next_icache_tag:
cache 0x8, 0($14) # Index Store tag Cache op
```

##### 初始化循环

**使用虚拟地址作为基址**: Cache 指令使用存储在基址寄存器（在本例中是 `$14`）中的虚拟地址。

**索引和路（way）的大小**: 索引字段的大小根据缓存路的大小而变化。越大的路，索引字段也越大。

**地址增量和路（way）位**: 代码并没有显式地设置路位。它通过简单地增加虚拟地址（按缓存行大小），使得每次循环时 Cache 指令都会初始化缓存的下一个集合。这最终会因为溢出到路位而导致缓存设置为下一个路中的索引 0。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230918141434710.png" alt="image-20230918141434710" style="zoom:50%;" />

##### 循环维护

**递减循环计数器**: 首先减小循环计数器（ `$12` ）。

```
add $12, -1  # Decrement set counter
```

**检查是否达到零**: 然后检查它是否减少到零。如果没有，跳回标签（`next_icache_tag`）。

```
bne $12, $0, next_icache_tag  # Done yet?
```

**分支延迟槽中的增量**: 分支延迟槽中的指令（总是会执行）用于将虚拟地址（`$14`）增加到缓存的下一个集合（set）。寄存器 `$11` 存储缓存行大小（以字节为单位）。

```
add $14, $11  # Increment line address by line size
```






<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230918140308209.png" alt="image-20230918140308209" style="zoom:50%;" />

#### 测试方式

L1指令缓存标签数组可以通过CACHE指令`Index Store Tag`的和`Index Load Tag`进行测试。`Index Store Tag`将ITagLo和ITagHi寄存器的内容写入选定的标签条目中。`Index Load Tag`将选定的标签条目读取到ITagLo和ITagHi寄存器中。

> Index Load Tag: 3'b 010
