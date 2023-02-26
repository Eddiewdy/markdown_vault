## tvm模型转换

1. 输入数据预处理

2. 加载模型

3. ==使用TVM的Relay将模型转换为Graph IR====

   ```python
   target = 'llvm'
   input_name = 'input.1'
   shape_dict = {input_name: x.shape}
   mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
   ```

   得到的mod是一个函数，函数的输入是ONNX模型中的Tensor的shape信息，包含输入和带权重OP的权重Tensor的信息。params则保存了ONNX模型所有OP的权重信息，以一个字典的形式存放，字典的key就是权重Tensor的名字，而字典的value则是TVM的Ndarry，存储了真实的权重。

4. 对计算图模型进行优化以及执行

   ```python
   with tvm.transform.PassContext(opt_level=1):
       intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)
   
   ######################################################################
   # Execute on TVM
   # ---------------------------------------------
   dtype = "float32"
   tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()
   
   print(np.argmax(tvm_output))
   ```

## tvm如何将ONNX转换成Realy IR









### vscode tip

>On Windows:
>
>Alt + ← ... navigate back
>
>Alt + → ... navigate forward
>
>
>
>On Mac:
>
>Ctrl + - ... navigate back
>
>Ctrl + Shift + - ... navigate forward
>
>
>
>On Ubuntu Linux:
>
>Ctrl + Alt + - ... navigate back
>
>Ctrl + Shift + - ... navigate forward









### 问题：

GraphProto类是什么

