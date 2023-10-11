Spatial Correlation：相关性分布在稀疏的空间范围内

cache block的大小限制了空间局部性的发掘

### 名词定义：

**spatial region generation**：一个空间区域内的记录

**spatial pattern**：block访存记录 bit vector

### SMS的两个部分

**active generation table**：records spatial patterns as the processor accesses spatial regions and trains the predictor

**pattern history table**：stores previously-observed spatial patterns, and is accessed at the start of each spatial region generation to predict the pattern of future accesses.

### 具体设计

![image-20221018233704173](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20221018233704173.png)

每次L1访存先搜索AGT的accumulation table，如果找到则更改Pattern，否则到Filter Table去找tag，找到了则判断offset，不一样则将这个entry转移到Accumulation table，并更新Pattern，如果Filter Table也没有，则将这个trigger设置为trigger access作为新的spatial region generation

碰到generation中的block被替换出cache则将这个pattern添加到PHT（pattern history table）