### SRRIP

![image-20221013234753314](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20221013234753314.png)

设置M和INS，M表示RRIP的bit位数，INS表示插入的指令初始化的大小，表示long re-reference / distant re-reference

每次替换替换数字为M的cache line（distant re-reference）

没有M的，则所有一起downgrade（+1）直到有M

### Bimodal RRIP

***Bimodal RRIP (BRRIP)*** that inserts majority of cache blocks with a *distant* re-reference interval prediction (i.e., RRPV of 2M–1) and infrequently (with low probability) inserts new cache blocks with a *long* re-reference interval prediction (i.e., RRPV of 2M–2).

### DRRIP

BRRIP+SRRIP+Set dueling

Set dueling：两个 *Set Dueling Monitors* 以及一个PSEL（计数器）决定余下的cache set的策略

