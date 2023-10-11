### virtual memory

程序员眼中的memory是虚拟内存，所有的程序虚拟内存范围都是 0-2^64^ B 大小。

处理器看到的是物理内存，大部分时候物理内存小于虚拟内存。物理内存地址和物理内存是一一对应的。 

虚拟内存和物理内存的页面大小是一致的。

虚拟内存超出的内容存储在disk中

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230314193819570.png" alt="image-20230314193819570" style="zoom:50%;" />

#### 虚拟内存到物理内存的翻译

页号作为index查页表，找到页帧号，offset一样。  虚拟内存地址：页号索引+页内偏移



flat页表需要过大的空间，采用多级页表，把页号分成两个部分，有些没有映射的外层页表就不需要内部页表了，从而达到节省空间的目的。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230314201217427.png" alt="image-20230314201217427" style="zoom: 33%;" />

load/store的步骤

- 计算虚拟地址
- 计算页号
- 根据页号得到物理页帧号（table walking）如果是多级页表，那么访存时间会成倍增长
- offset+页帧号得到物理地址

  <img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230314212727442.png" alt="image-20230314212727442" style="zoom:50%;" />

PIPT：先访问TLB得到物理地址，再访问cache得到数据，cache的index是根据物理地址来的

VIPT：使用虚拟地址同时访问tlb和cache，但是上下文切换会导致cache flush

VIPT：兼顾两者，但是要考虑aliasing的问题，解决方法：控制cache的大小



cache大小 <= 组相连 * 页面大小

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230316211258697.png" alt="image-20230316211258697" style="zoom:50%;" />