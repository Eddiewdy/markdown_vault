## vLLM调研

------------------------

### 一句话概括

vLLM 是来自 UC Berkeley 在 LLM 推理方面的最新工作，最大亮点是采用 Paged Attention 技术，结合 Continuous Batching，极大地优化了 realtime 场景下的 LLM serving 的 throughput 与内存使用效率。

### 简单的Demo

```python
prompts = [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
```



### 整体框架

- LLMEngine：是整个系统的入口，`add_request`负责输入prompt请求，`step`迭代推理，最终返回LLM生成的结果。其内部组合了一个Scheduler和一组Worker。

  - Scheduler：在每个推理步调度可处理的Sequence输入信息，其组合包含了一个BlockSpaceManager

  - - BlockSpaceManager：维护gpu显存和cpu内存的使用情况，以及Sequence对应Cache的BlockTable信息。

  - Worker：在每个推理步执行LlamaForCausalLM推理，并返回采样后结果。除一个LLM模型外，其另一个核心组件是CacheEngine。

  - - CacheEngine：负责执行相关gpu、cpu空间的换入、换出、拷贝等操作。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919232856570.png" alt="image-20230919232856570" style="zoom:50%;" />

### LLMEngine

再把demo的代码放过来看一看。

```python
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
```

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919234704503.png" alt="image-20230919234704503" style="zoom:50%;" />

#### \__init__

`LLM` 是对 LLM serving 部分的封装，也是核心部分。首先它会初始化这个类。初始化过程中大部分参数都会被用来构造 `EngineArgs`，这是一个 dataclass，封装了 Engine 的初始化参数。然后构建 `LLM Engine`。一个 `LLM` 只有一个 `LLM Engine`，所以它就是对 Engine 再包一层。

初始化 `LLM Engine` 时候会先调用 `create_engine_configs` 将 `EngineArgs` 分解成 `ModelConfig`，`CacheConfig`， `ParallelConfig`和`SchedulerConfig`。其中

- `ModelConfig` 包括了对 model 和 tokenizer 的定义，dtype 和随机数 seed 以及是否用 pretrained weights 还是 dummy weights 等。
- `CacheConfig` 包括 block_size（每个 block 多大）， gpu_utilization（GPU 利用率，后面 allocate 的时候占多少 GPU）和 swap_space（swap 的空间大小）。默认 block_size=16，swap_space=4GiB。
- `ParallelConfig` 包括了 tensor_parallel_size 和 pipeline_parallel_size，即张量并行和流水线并行的 size，由于我们是单卡，这两个都是 1。
- `SchdulerConfig` 包括了 max_num_batched_tokens（一个 iteration 最多处理多少个 tokens），max_num_seqs（一个 iteration 最多能处理多少数量的 sequences）以及 max_seq_len（最大生成多长的 context length，也就是一个 sequence 的最长长度，包含 prompt 部分和 generated 部分）。

然后对于每个 device（也即每张卡 / 每个 rank）创建一个 Worker。Worker 是运行 model 的单位。一个 Engine 管理所有的 workers。同时给这个 engine 创建它的 scheduler，以及初始化这个 engine 的 KV cache。

在初始化 KV cache 的时候会调用 Worker 的 CacheEngine，初始化 gpu、cpu 空间，计算能容纳多少个 block。每个block包含block_size 个 token 对应的各层 KVCache 大小。在后续的组织中都会将 Sequence 对应的 KVCache 分成 block_size 大小的 cache block，以方便管理对应 block 的空间。

#### add_request

`add_request`接口执行多次，接收多个待处理的prompt，将prompt处理成对应token的Sequence。每个输入prompt构造一个SequenceGroup， 其中包含了多个重复的Sequence为后续beam search做准备。SequenceGroup会最终保存到Scheduler中，以进行后续的调度。

在 Sampling 设置中有一个 best_of 参数，它指定了每个 prompt 我们生成多少个 output sequences。在使用 beam search 时候这个参数也作为 beam width。

所以对每个 request，我们就需要构造  best_of  个 Sequence。每个 Sequence 就是对应一个从 prompt 生成出来的完整语句。这些个 Sequence 组成一个 SequenceGroup，也就是一个 request 对应一个 seq group。

构造 Sequence 的时候会给这个 Sequence 分配相应的 logical token blocks。一个 logical blocks 对应 block_size 个 token ids。初始化的时候会把 token ids 分段存储在 logical blocks 中，最后一个 block 可能不满。

#### step

step 执行一个推理步。首先Scheduler会调度一组SequenceGroup和相关信息作为当前推理步的执行输入，除了输入外，还会包含当前SequenceGroup所需 KVCache 的换入换出信息。然后，Worker 会将执行一次 LLM 推理（CacheEngine 先准备 KVCache）。Worker 采样的输出结果会再次更新到 Scheduler 中的 SequenceGroup 内，以更新其内部的状态。最后，多次 step 循环，直到所有 prompt 对应的 SequenceGroup 都生成结束。

### Scheduler

Scheduler 中维护了三个队列：waitting、running、swapped。每个队列中的元素都是一个 SequenceGroup。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919234827459.png" alt="image-20230919234827459" style="zoom:50%;" />

- waitting：等待计算 KVCache 的 SequenceGroup（也就是 prompt 序列）

- running：执行推理的 SequenceGroup，会在当前 step 中作为输入，一共包含两类：

- - prompt：来自 waitting，未计算 KVCache 的SequenceGroup
  - generate token：计算过 KVCache 的 SequenceGroup，准备生成下一个 token

- swapped：KVCache 暂时换出到 cpu 内存的 SequenceGroup

在每次接口 schedule 执行时，会调度几个队列之间的 SequenceGroup，维护队列间的状态，使得当前执行推理尽可能占满显存空间。详细逻辑如上图中的数字标号顺序所示，通过调度能实现两种解决显存不足的策略，一个是换出到cpu中，另一个就是重计算。

总结来说，Scheduler 每次调用 schedule 都会把上个时刻的这三个队列往下一个时刻进行一次更新。按照优先级顺序：首先它优先保证 running 队列继续运行，其次它尝试将 swapped 的任务 swap out 回来，如果没有可用的 swapped 队列则会尝试将 waiting 队列中的任务放到 running。

当 SequenceGroup 推理新增了token时，update 接口会更新一遍 SequenceGroup 内的状态。如下图所示，SequenceGroup内包含一组 beam search 的 seq ，每次执行推理的时候，每个seq采样s次，那么久会生成 n*s 个生成的 token，根据这里面token保留置信度topn个生成结果。下图中的结果就是n=4的情况，可以看到 topn 保留的 output 里 seq1 和 seq3 都来自原始输入 seq1（parent_seq=1），此时需要 BlockSpaceManager 将原始的 seq3 释放（因为保留的output里没有seq3的输出），然后从 seq1拷贝/ fork 到 seq3，再将新 token 添加到各个 seq 中。

![image-20230919235051666](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919235051666.png)

### BlockSpaceManager

一个 scheduler 还包含一个 BlockSpaceManager，BlockSpaceManager 的功能是管理各个 SequenceGroup 对应 KVCache 存储信息。每个 Sequence 的 KVCache 序列会分成多个block_size 长度的 cache block，每个 cache block 的位置信息记录在 BlockSpaceManager。下图中，BlockSpaceManager包含一个 block_tables，其记录 cache block 到 gpu 显存或 cpu 内存物理地址的映射。

SequenceGroup 刚加入到 Scheduler 的时候并没有分配空间，第一次进入 running 状态的时候需要向 BlockSpaceManager 申请可用的 block 空间。如下图所示，BlockSpaceManager 分配 block 空间是以一个 SequenceGroup 作为一组输入，而且默认分配空间的时候，所有 SequenceGroup 内的 token 都是一样的（即是相同的prompt），因此会为所有Sequence都指向同一片 cache block 区域，该区域被引用数为 Sequence 数量。

当需要为一个 Sequence 新增 token 时，如下图所示，有两种情况：

- 当前 cache block 空间不足添加新 token，则新增 cache block。
- 当前空间足以添加新 token，但 last block与其他 Sequence 共用时（ref_count>1），如果新 token 还是添加到 last block，那么会与共用 last block 的其他 Sequence 冲突，则需要释放掉 last block（last block 的 ref_count-1），随后给该 Sequence 分配一个新的 last block。最后，返回信息标记原本 last block 内容需要拷贝到这个新的 last block，即 copy-on-write 机制。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919235741476.png" alt="image-20230919235741476" style="zoom:50%;" />

### Cache Engine

实际上，BlockSpaceManager只负责维护cache block到gpu/cpu空间的索引，真正进行换入、换出、拷贝操作都需要通过Worker中CacheEngine进行。CacheEngine 是对于 KV Cache 显存实际进行操作的单元。

初始化时候，它先根据之前 profile 的数据（cpu/gpu blocks数）来 allocate cache。然后再给 caching 操作初始化一个 CUDA Stream，以及给每一个 layer 初始化一个 cuda event 来用做 stream synchronization。

在 vLLM 里，每个 key block 的 shape 是 `num_heads, head_size // x, block_size, x` ，其中 x 是 16 // dtype 的大小。也就是说 fp32 时 x=4，fp16 时 x=8。每个 value block 的 shape 是 num_heads, head_size, block_size 。（为什么 key block 要按 x split？在后面的 kernel 实现里会提到这个加速）

cache engine 里支持了其它两个操作：

- copy。由专门的 cu 函数 `copy_blocks` 支持。
- swap_in 和 swap_out。in 就是 cpu to gpu，out 就是 gpu to cpu。内部实现由专门的 cu 函数 `swap_blocks` 支持。

下面看相关的 cu 函数，实现在 csrc/cache_kernels.cu 中。

- swap_blocks(src, dst, block_mapping)： for 一遍 block_mapping，对于每个 [src, dst] pair（block number to block number）做一个单 block 的 copy。支持 GPU to GPU（必须是同一个 GPU 上），GPU to CPU，CPU to GPU。
- copy_blocks(key_caches, value_caches, block_mapping)：这里的 mapping 是 `int->list[int]`，也就是一个 src block idx 对应多个 dst block idx。copy 的过程用一个 global kernel 并行实现。
- reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
- gather_cached_kv(key, value, key_cache, value_cache, slot_mapping)

### Worker

Worker 是对单个 GPU 的抽象。

Engine 通过调用 _run_workers("<method_name>", *args, get_all_outputs, **kwargs) 来在 所有 workers 上执行方法。如果 get_all_outputs 设成 True，那么它会将所有 workers 的返回结果包装成 List 来返回。否则，它只会返回第一个 worker 的结果，并且 assert 所有 workers 的输出都是一样的。在实际执行中主要会调用如下方法（方法名, get_all_outputs=False/True）：

- profile_num_avaiable_block，True：通过一次 "试运行" 来 profile peak memory。 每张卡的 blocks 个数可能不同（显存不同），所以需要 get all outputs。由于 vLLM 使用一个中心化的管理单元，因此我们会对 profile 出来的 blocks 个数取 min。
- init_cache_engine，False：初始化 cache engine。

#### 执行推理

Worker在执行时首先进行两个操作。

- 缓存更新：SchedulerOutputs 包含了前面提到的当前所需 swap in/swap out/copy 的 cache block 信息，然后通过CacheEngine 自定义的 op 去执行缓存搬运操作，得到 cuda stream 的 event，后续会在推理 LLM 各层的时候用到。
- 准备输入token序列 __prepare_input：上图右侧的框内是预处理的过程，将 SequenceGroupMetadata 包含 Scehduler 调度得到 running 的所有 SequenceGroup 组合一个 flatten 的 token 序列，作为 LLM 的初始输入。Scheduler 中提到过，running 队列中当前执行的 SequenceGroup 有两类：一类未计算 prompt（前缀）的 KVCache，这部分需要完整的 prompt token 输入去计算各层的 KVCache（全量推理）。另一类已经计算并缓存前缀的 KVCache，因此只需要 last token 作为输入计算下一个 generation token 的分布（增量推理）。如图所示，输入 token 序列的前半部分是多个 prompt 的 token 全量推理序列，后半部分是各个增量推理序列的 last token。此外，全量推理的 SequenceGroup 中多个 Sequence 共享prompt，因此只需要任意一个 Sequence 的 prompt 作用输入就行。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230920090452722.png" alt="image-20230920090452722" style="zoom:67%;" />

随后会执行推理的过程。`__prepare_input`组装的 flatten token 在各层映射成 flatten hidden state 。 除了线性层、激活层等token 独立计算的层以外，attention 层的计算涉及不同 token 的 hidden state 依赖。下图主要展开了 Attention 层的四个主要步骤：

- prompt 全量推理：prompt序列通过集成 xformers 的 attention 算子计算得到下个 layer 的 hidden state 。由于这里attention 计算的输入是完整的 tensor，不是 KVCache 中分散的 cache block，所以可以用第三方的 attention 算子完成计算。
- 等待缓存事件：CacheEngine 中发送了异步缓存操作，因此只有当前层序列的 cache block 完成缓存更新，才能进一步获取KVCache 或者记录 KVCache，这种异步的实现能通过 overlap 计算和缓存搬运，节省一部分缓存搬运时间。
- 记录当前 KVCache：当前层输入的 hidden state 作为 KVCache 通过自定义算子记录到对应 cache block 内，这里记录所有有效 token 的 hidden state，包括 prompt 和 last token（last token是前几次step中新增的，所以也没有缓存hidden state到KVCache）。
- generation token 增量推理：vLLM 的核心 PageAttention 在此实现，这里通过一个自定义算子，实现了基于BlockTable分散KVCache的增量attention计算。

最后 LLM 内的采样器进行采样，将 beam_search 结果（新 token ）返回给 Worker 输出。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230920090400179.png" alt="image-20230920090400179" style="zoom:50%;" />

### single_query_kv_attention

在后续的生成（generated by generation tokens）是通过 single_query_cached_kv_attention 实现的，也就是 vLLM 的 Paged Attention。

传入的参数 shape 如下：

```python3
output: torch.Tensor,           # [num_generation_tokens, num_heads, head_size]
query: torch.Tensor,            # [num_generation_tokens, num_heads, head_size]
key_cache: torch.Tensor,        # [num_blocks, num_heads, head_size/x, block_size, x]
value_cache: torch.Tensor,      # [num_blocks, num_heads, head_size, block_size]
input_metadata: InputMetadata,
```

kernel 的 grid 是 `(num_heads, num_seqs)` ，block 大小是 `(NUM_THREADS,)`（这个参数代码里是 128）。每个 warp 是 32 个 threads。也就是说一个 thread 计算的是一个 head，一个 seq 的部分数据。在 warp 之内还有一层抽象，代码里被称作 `thread group`，按代码的意思是一个 warp 里对不同块的切分（也就是 warp_size / block_size）。所以如果现在我们运行 vLLM 时使用大于 32 的 block size，其实会产生浪费。

```text
WARP_SIZE = 32 (threads)  
THREAD_GROUP_SIZE = max(WARP_SIZE / BLOCK_SIZE, 1) (threads) # warp = thread_group * block
NUM_WARPS = NUM_THREADS / WARP_SIZE
```

这里代码里对 K 和 Q 分别定义了 vec type。注释里说 vec size 的制定规则是让一个 thread group 一次刚好处理 16bytes 的数据。所以一个 vec type 就有 `16 / (thread_group_size * sizeof(scalar_t)` 个 scalar_t 类型。

第一步是把 Q load 到 SM 来。在一个 thread group 中，每一个 thread 负责 query 的一部分。例如若 group size 是 4，那么 group 的第一个 thread 就负责 Q 矩阵的第 0, 4, 8, ... 个 vectors。第二个 thread 就负责 Q 矩阵的第 1, 5, 9, ... 个 vectors。

Load 完 Q 之后就是处理 key 和 QK dot 的部分。这里的 x 和之前在 Python 代码定义的 x 是一个意思（`16 / sizeof(scalar_t)`。关于这个 x，在 FasterTransformer 的代码里也可以看到有相关注释：

>The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The values for x are chosen to create chunks of 16 bytes.

为了某些 data locality 而设置的排法。QK 的计算最后归结为一句 `Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs)`，其中 q_vecs 和 k_vecs 每个都有 NUM_VECS_PER_THREAD 个前面定义的 "vecs" 类型。这个 kernel 调用包含了同一个 thread group 内的 reduction。同时在计算 QK dot 时候也顺便算了 qk_max，为 softmax 做准备。

算完 QK dot 后的 qk_max 只是同一个 thread group 内的 reduction。之后还需要做 warp 内的 reduction（这部分通过原语 `__shfl_xor_sync` 来完成）以及 warp 之间的 reduction（这部分通过读写 shared memory 完成）。之后采用相似的方式来完成 logits 和 logits @ trans(V) 的计算。