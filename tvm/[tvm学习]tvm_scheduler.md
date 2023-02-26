### **scheduler**在整个TVM软件栈中的位置

tvm将计算图转换为te（tensor expression），用户手动指定计算策略scheduler，然后tvm生成特定后端的代码。tvm能否产生高性能代码的关键就在于scheduler的指定是否合理。

![图片](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/640.png)