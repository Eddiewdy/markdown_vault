### GPTQ量化

GPTQ 的核心思想是逐一量化模型的各个层，对于每个层寻找可以满足下式的量化结果：

![image-20230717154437984](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230717154437984.png)

即对于每个需要被量化的层(对应参数W)，希望量化前后该层输出变化尽量小。



**使用的仓库代码：**

根据下面这个仓库总结一下GPTQ的过程

https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/triton

切换到cuda分支：git checkout cuda



**GPTQ总体过程：**

1. 加载量化前的模型和模型参数（.pt），获取dataloader
2. 按层进行量化，得到量化后的字典quantizers——调用`llama_sequential(model, dataloader, dev)`
3. 查询字典quantizers，将量化后的模型参数加载，进行模型linear层的替换（linear替换为QuantLinear），得到替换后的model——调用`llama_pack()`
4. 前向推理：前向的时候Linear已经被替换为QuantLinear进行前向，QuantLinear前向会先将参数反量化然后前向，然后调用torch.matmul



**核心代码：**

gptq.py：负责具体的量化接口 gptq.fasterquant()

quant.py：定义了量化替换的线性层 class QuantLinear()

llama.py：GPTQ的量化入口

​		llama_sequential(model, dataloader, dev)：返回量化后的权重字典quantizers

​		llama_pack()：替换model的linear层，加载quantizers中的权重





代码注释

```python
@torch.no_grad()
# GPTQ量化入口
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}
	# 使用钩子函数，得到模型的attention_mask和position_ids，用于之后的fasterquant的输入
    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
	
    # 开始量化过程
    
    print('Ready.')

    quantizers = {}
    observer = Observer()
    
    # 对每层tranformer进行量化
    
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+------------------+--------------+------------+-----------+-------+')
        print('|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |')
        print('+==================+==============+============+===========+=======+')

        layer = layers[i].to(dev)
        full = find_layers(layer)
        if args.true_sequential:
            sequential = [['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'], ['self_attn.o_proj'], ['mlp.up_proj', 'mlp.gate_proj'], ['mlp.down_proj']]
        else:
            sequential = [list(full.keys())]
		
        # 按序列进行量化
        # 结果以字典的形式保存在quantizers中
        # gptq.fasterquant是GPTQ量化的api
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=args.observe)
                gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)

            def add_batch(name):

                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp
			
            # 注册钩子，一个batch 128个层的输入输出，GPTQ按batch进行量化
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
			
        
            for name in subset:
                # GPTQ的核心调用
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                # 存入字典
                quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)

                if args.observe:
                    observer.submit(name=name, layerid=i, gptq=gptq[name], error=error)
                else:
                    gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print('+------------------+--------------+------------+-----------+-------+')
        print('\n')

    model.config.use_cache = use_cache

    return quantizers
```





```python
# 将量化的结果（quantizers字典）加载到model中
def llama_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    # 将model中的被量化的层替换成QuantLinear
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    # 得到所有的QuantLinear
    qlayers = find_layers(model, [quant.QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        # 加载量化后的参数到对应的layer
        # 具体的实现在quant_linear.py
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model
```



