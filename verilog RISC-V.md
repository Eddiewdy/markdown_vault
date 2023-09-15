### RSD

RSD 支持多种高级微架构功能，例如推测式load和store执行、内存依赖预测器 (MDP)、推测性调度和非阻塞缓存。

RSD的高性能和高效率是通过两种方式：FPGA 友好的推测调度，多端口 RAM 的组件。



前端分支预测器：gshare

发射队列：随机选择、指令重放机制

指令重放：

- store data array按照指令先后顺序保存所有的没有被提交的store指令
- 一个load在STQ中搜索所有比他更老的已经生成了地址的store指令，选择其中最年轻的指令做数据forward
- 一个store在STQ中搜索所有**完成了**且比他**更年轻**且有**相同内存地址**的load指令，将这条load指令以及其依赖指令重放

banked RAM：在banked RAM中，多个请求被分发到多个bank。
