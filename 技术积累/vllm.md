## vLLM调研

------------------------

### 一句话概括

vLLM 是来自 UC Berkeley 在 LLM 推理方面的工作，最大亮点是采用 Paged Attention 技术，结合一些 cpu 和 gpu 管理技术，极大地优化了 LLM serving 的 throughput 与内存使用效率。

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

```python
class LLM:
    """这是一个名为LLM（语言模型）的Python类，这个类用于从给定的提示和采样参数生成文本。
    类的主要部分包括tokenizer（用于将输入文本分词）、语言模型（可能分布在多个GPU上执行）
    以及为中间状态分配的GPU内存空间（也被称为KV缓存）。给定一批提示和采样参数，
    该类将使用智能批处理机制和高效的内存管理从模型中生成文本。

    这个类设计用于离线推理。在线服务的话，应使用AsyncLLMEngine类。
    对于参数列表，可以参见EngineArgs。

    Args:
        model: HuggingFace Transformers模型的名称或路径.
        tokenizer: HuggingFace Transformers分词器的名称或路径。默认为None。.
        tokenizer_mode: 分词器模式。"auto"将使用快速分词器（如果可用），
        "slow"将总是使用慢速分词器。默认为"auto"。.
        trust_remote_code: 当下载模型和分词器时，是否信任远程代码
        （例如，来自HuggingFace的代码）。默认为False。
        tensor_parallel_size: 用于分布式执行的GPU数量，使用张量并行性。默认为1。
        dtype: 模型权重和激活的数据类型。目前，我们支持float32、float16和bfloat16。
        如果是auto，我们使用在模型配置文件中指定的torch_dtype属性。
        但是，如果配置中的torch_dtype是float32，我们将使用float16。默认为"auto"。
        seed: 初始化采样的随机数生成器的种子。默认为0。
    """

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs, # 其它关键字参数。
    ) -> None:
        # 在初始化函数中，首先检查kwargs中是否包含"disable_log_stats"键，
        # 如果没有，则在kwargs中添加该键并设置其值为True。
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        # 使用所有给定的参数（包括通过kwargs传递的任何额外参数）来初始化EngineArgs对象，
        # 然后使用这些参数来初始化LLMEngine对象
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        # 初始化一个名为request_counter的Counter对象，用于请求计数。
        self.request_counter = Counter()
```



`LLM` 是对 LLM serving 部分的封装，也是核心部分。首先它会初始化这个类。初始化过程中大部分参数都会被用来构造 `EngineArgs`，这是一个 dataclass，封装了 Engine 的初始化参数。然后构建 `LLM Engine`。一个 `LLM` 只有一个 `LLM Engine`，所以它就是对 Engine 再包一层。

初始化 `LLM Engine` 时候会先调用 `create_engine_configs` 将 `EngineArgs` 分解成 `ModelConfig`，`CacheConfig`， `ParallelConfig`和`SchedulerConfig`。其中

- `ModelConfig` 包括了对 model 和 tokenizer 的定义，dtype 和随机数 seed 以及是否用 pretrained weights 还是 dummy weights 等。
- `CacheConfig` 包括 block_size（每个 block 多大）， gpu_utilization（GPU 利用率，后面 allocate 的时候占多少 GPU）和 swap_space（swap 的空间大小）。默认 block_size=16，swap_space=4GiB。
- `ParallelConfig` 包括了 tensor_parallel_size 和 pipeline_parallel_size，即张量并行和流水线并行的大小，单卡这两个都是 1。
- `SchdulerConfig` 包括了 max_num_batched_tokens（一个 iteration 最多处理多少个 tokens），max_num_seqs（一个 iteration 最多能处理多少数量的 sequences）以及 max_seq_len（最大生成多长的 context length，也就是一个 sequence 的最长长度，包含 prompt 部分和 generated 部分）。

继续看 \__init__ 的后半部分:

```python
class LLMEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
       
		...
    
        # 对于每个 device（也即每张卡 / 每个 rank）创建一个 Worker。
        # Worker 是运行 model 的单位。一个 Engine 管理所有的 workers。
        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)
    
    	# 初始化这个 engine 的 KV cache。
        # Profile the memory usage and initialize the cache.
        self._init_cache()

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []
```



然后对于每个 device（也即每张卡 / 每个 rank）创建一个 Worker。Worker 是运行 model 的单位。一个 Engine 管理所有的 workers。同时给这个 engine 创建它的 scheduler，以及初始化这个 engine 的 KV cache。

在初始化 KV cache 的时候会调用 Worker 的 CacheEngine，初始化 gpu、cpu 空间，计算能容纳多少个 block。每个block包含block_size 个 token 对应的各层 KV cache 大小。在后续的组织中都会将 Sequence 对应的 KV cache 分成 block_size 大小的 cache block，以方便管理对应 block 的空间。

```python
# _init_workers 这个方法是 LLMEngine 类的一个私有方法，其主要目的是初始化worker。
# 这些worker负责在硬件资源（如GPU）上执行计算任务。
# 这个函数只接受一个参数，即 distributed_init_method，它是一个字符串，用于指定分布式执行的初始化方法。
def _init_workers(self, distributed_init_method: str):
    # 从vllm.worker.worker模块中导入Worker类。这个导入操作被放在了函数内部，
    # 这样做的目的是为了避免在CUDA_VISIBLE_DEVICES被Worker类设定之前就导入
    # 了torch.cuda/xformers，因为那样可能会产生问题。
    from vllm.worker.worker import Worker  # pylint: disable=import-outside-toplevel
    
    # 断言self.parallel_config.world_size（并行世界的大小）等于1，如果不等于1，
    # 则会抛出错误，提示用户需要使用Ray框架进行并行计算。
    assert self.parallel_config.world_size == 1, (
        "Ray is required if parallel_config.world_size > 1.")

    self.workers: List[Worker] = []
    # 创建一个新的 Worker 对象，并将其添加到 self.workers 列表中。
    # 每个 Worker 对象都需要以下参数：
    # self.model_config，self.parallel_config，self.scheduler_config，
    # 以及工作节点的 rank（在这个例子中，rank 是0，表示这是第一个，也是唯一的工作节点）
    # 和 distributed_init_method。
    worker = Worker(
        self.model_config,
        self.parallel_config,
        self.scheduler_config,
        0,
        distributed_init_method,
    )
    # 调用_run_workers方法，参数为 "init_model" 
    # 和 get_all_outputs=True，对所有的worker进行初始化。
    self.workers.append(worker)
    self._run_workers(
        "init_model",
        get_all_outputs=True,
    )

# _init_cache函数是LLMEngine类的一个私有方法，不接受任何参数，没有返回值。
# 其目标是测量内存使用并初始化KV（键值）Cache。
def _init_cache(self) -> None:
    """Profiles the memory usage and initializes the KV cache."""
    # Get the maximum number of blocks that can be allocated on GPU and CPU.
    # 使用_run_workers方法来获取可以在GPU和CPU上分配的最大块数量。
    # _run_workers函数执行的方法是profile_num_available_blocks，并且提供了如块大小、
    # GPU内存使用率和CPU交换空间等参数，所有这些参数都是从cache_config对象中提取出来的。
    num_blocks = self._run_workers(
        "profile_num_available_blocks",
        get_all_outputs=True,
        block_size=self.cache_config.block_size,
        gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
        cpu_swap_space=self.cache_config.swap_space_bytes,
    )

    # 找到所有workers中可用块的最小值，以确保所有的内存操作都可以应用到所有worker。
    # 在这个步骤中，函数分别计算了GPU和CPU的块数量。
    num_gpu_blocks = min(b[0] for b in num_blocks)
    num_cpu_blocks = min(b[1] for b in num_blocks)
    # FIXME(woosuk): Change to debug log.
    logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                f"# CPU blocks: {num_cpu_blocks}")

    # 如果GPU的块数量小于等于0，函数将抛出一个值错误。
    # 这是为了确保在初始化引擎时，为缓存块提供足够的可用内存。
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    
    # 根据计算的块数量，更新cache_config对象的num_gpu_blocks和num_cpu_blocks属性。
    self.cache_config.num_gpu_blocks = num_gpu_blocks
    self.cache_config.num_cpu_blocks = num_cpu_blocks

    # Initialize the cache.
    # 使用_run_workers方法初始化缓存。此步骤中的_run_workers执行的方法
    # 是init_cache_engine，并且提供了cache_config对象作为参数。
    self._run_workers("init_cache_engine", cache_config=self.cache_config)
```



#### add_request

```python
# add_request函数是LLMEngine类的一个方法，它接受一个请求并将其加入到scheduler的请求池中。
# 这个请求在调用engine.step()函数时由调度器进行处理，具体的调度策略由调度器决定。
def add_request(
        self,
        request_id: str, # 请求的唯一ID。
        prompt: Optional[str], # prompt字符串。如果提供了prompt_token_ids，这个参数可以为None。
        sampling_params: SamplingParams, # 用于文本生成的采样参数。
        # prompt的token ID。如果它为None，则使用分词器将提示转换为token ID。
        prompt_token_ids: Optional[List[int]] = None, 
        arrival_time: Optional[float] = None, # 请求的到达时间。如果为None，则使用当前时间。
    ) -> None:
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        # 每一个序列代表一次独立的文本生成任务。它们的数量由sampling_params.best_of决定。
        # 每个序列都包含了唯一的seq_id，提示和标记ID，以及block_size（块大小）。
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        # 创建序列组（SequenceGroup）。一个序列组包含了一组相关的序列，
        # 它们共享相同的请求ID和采样参数，并且在同一时间到达。
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        # 将序列组添加到调度器中。这样，当调用engine.step()函数时，
        # 调度器就可以根据它的调度策略处理这些序列组。
        self.scheduler.add_seq_group(seq_group)
```



`add_request` 接口执行多次，接收多个待处理的 prompt，将 prompt 处理成对应 token 的 Sequence。每个输入 prompt 构造一个 SequenceGroup， 其中包含了多个重复的 Sequence 为后续 beam search 做准备。SequenceGroup 会最终保存到 Scheduler中，以进行后续的调度。

在 sampling_params 中有一个 best_of 参数，它指定了每个 prompt 我们生成多少个 output sequences。在使用 beam search 时候这个参数也作为 beam width。

所以对每个 request，我们就需要构造  best_of  个 Sequence。每个 Sequence 就是对应一个从 prompt 生成出来的完整语句。这些个 Sequence 组成一个 SequenceGroup，也就是一个 request 对应一个 seq group。

构造 Sequence 的时候会给这个 Sequence 分配相应的 logical_token_blocks。一个 LogicalTokenBlock 对象对应 block_size 个 token ids。初始化的时候会把 token ids 分段存储在 logical blocks 中，最后一个 block 可能不满。

SequenceGroup 构造好之后会给到 Scheduler（self.scheduler.add_seq_group(seq_group)），然后在 engine 的 `step` 中`seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()` 这行代码会把 schdule 出来的序列包装成SequenceGroupMetadata。接下来，在执行 worker 的 `execute_model` 函数时会通过 `_prepare_inputs` 转成 tokens_tensor，position_tensor 和 InputMetadata，然后 `execute_model` 函数实际上包含模型的前向推理和 Sample 过程，它的返回值数据集结构是 Dict[int, SequenceOutputs] ，也就是 seq id -> 对应的输出，输出包含着 log prob。 这个输出会被传到 scheduler 的 `update`  接口，用于更新对应的 running sequences。

#### step

```python
# 这个函数是 LLMEngine 类的一个关键函数，其功能是执行一次迭代并返回新生成的结果。
# 这个函数的执行过程可以分解为以下步骤：
def step(self) -> List[RequestOutput]:
    """Performs one decoding iteration and returns newly generated results.

    This function performs one decoding iteration of the engine. It first
    schedules the sequences to be executed in the next iteration and the
    token blocks to be swapped in/out/copy. Then, it executes the model
    and updates the scheduler with the model outputs. Finally, it decodes
    the sequences and returns the newly generated results.
    """
    # 首先，调用 self.scheduler.schedule() 进行调度，返回要在下一次迭代中执行的序列，
    # 以及要被换入，换出，复制的 token 块。
    seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
    # 然后，检查 scheduler_outputs 是否为空。如果为空并且没有被忽略的序列组，
    # 则表示没有需要做的工作，函数返回空列表。如果存在被忽略的序列组，那么我们需要将它们作为请求输出返回。
    if scheduler_outputs.is_empty():
        if not scheduler_outputs.ignored_seq_groups:
            # Nothing to do.
            return []
        # If there are ignored seq groups, we need to return them as the
        # request outputs.
        return [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]

    # Execute the model.
    # 如果 scheduler_outputs 不为空，那么就会执行模型，将 seq_group_metadata_list、
    # blocks_to_swap_in、blocks_to_swap_out 和 blocks_to_copy 作为参数传给 _run_workers 
    # 方法。这一步可能包括将一些状态从内存移到 GPU，执行模型计算，以及将一些状态从 GPU 移回内存。
    output = self._run_workers(
        "execute_model",
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        blocks_to_copy=scheduler_outputs.blocks_to_copy,
    )
    # Update the scheduler with the model outputs.
    # 之后，使用模型的输出结果来更新调度器。
    seq_groups = self.scheduler.update(output)

    # Decode the sequences.
    # 然后对序列进行解码，并终止满足停止条件的序列。完成的序列组将被释放。
    self._decode_sequences(seq_groups)
    # Stop the sequences that meet the stopping criteria.
    self._stop_sequences(seq_groups)
    # Free the finished sequence groups.
    self.scheduler.free_finished_seq_groups()

    # Create the outputs.
    # 最后，创建输出结果。对于每一个序列组，都将其转换为 RequestOutput 对象
    # 并添加到输出列表中。如果 log_stats 为真，那么还会记录系统状态。
    request_outputs: List[RequestOutput] = []
    for seq_group in seq_groups + scheduler_outputs.ignored_seq_groups:
        request_output = RequestOutput.from_seq_group(seq_group)
        request_outputs.append(request_output)

    if self.log_stats:
        # Log the system stats.
        self._log_system_stats(scheduler_outputs.prompt_run,
                               scheduler_outputs.num_batched_tokens)
    return request_outputs
```

step 执行一个推理步。首先Scheduler会调度一组SequenceGroup和相关信息作为当前推理步的执行输入，除了输入外，还会包含当前SequenceGroup所需 KVCache 的换入换出信息。然后，Worker 会将执行一次 LLM 推理（CacheEngine 先准备 KVCache）。Worker 采样的输出结果会再次更新到 Scheduler 中的 SequenceGroup 内，以更新其内部的状态。最后，多次 step 循环，直到所有 prompt 对应的 SequenceGroup 都生成结束。

### Scheduler

Scheduler 中维护了三个队列：waitting、running、swapped。每个队列中的元素都是一个 SequenceGroup。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919234827459.png" alt="image-20230919234827459" style="zoom:50%;" />

- waitting：等待计算 KVCache 的 SequenceGroup（也就是 prompt 序列）

- running：执行推理的 SequenceGroup，会在当前 step 中作为输入，一共包含两类：

- - prompt：来自 waitting，未计算 KVCache 的SequenceGroup
  - generate token：计算过 KVCache 的 SequenceGroup，准备生成下一个 token

- swapped：KV Cache 暂时换出到 cpu 内存的 SequenceGroup

在每次接口 schedule 执行时，会调度几个队列之间的 SequenceGroup，维护队列间的状态，使得当前执行推理尽可能占满显存空间。通过调度能实现两种解决显存不足的策略，一个是换出到cpu中，另一个就是重计算。详细逻辑如下：

首先在 `llm_engine.add_request` 函数中把根据输入 prompt 构造好的 SequenceGroup 添加到 scheduler 的 waiting 队列里（`self.scheduler.add_seq_group(seq_group)`）。scheduler的初始化函数中定义了三个代表状态的 List[SequenceGroup] `waiting`, `running`, `swapped` 以及初始化了一个 block_manager 来管理 logical blocks 和 physical blocks 之间的映射关系。然后是最核心的`_schedule`函数。

首先，当 `waiting` 不空时，它会把一些 `waiting` 中的SequenceGroup加到 `running` 中，但需要满足一些条件比如block_manager里面是否还有足够的空间可以塞得下这个SequenceGroup(对应 `if not self.block_manager.can_allocate(seq_group): break`)，当前序列的prompt长度是否超出了`prompt_limit`，已经批处理的token数量（ `num_batched_tokens` ）加上当前序列组（seq_group）的token数量（num_prompt_tokens）是否超过了配置中的最大token限制，如果超过了，循环就会中断。还有一些限制可以看下面的代码。

然后，scheduler会遍历`running`里面的每个SequenceGroup，然后检查 block_manager 是否够塞得下。 如果不行，则它会驱逐 running 队列中优先级最低的 SequenceGroup，如果空间够的话，则会对这个 SequenceGroup allocate 相应的 physical blocks，然后将其放入 update 后的 `running` 列表中。经过这个过程，scheduler 更新了 `running` 列表，并把部分任务驱逐掉。

接下来，scheduler会过一遍`swapped`里面的每个SequenceGroup，尝试 swap in 那些能够 swap 的 SequenceGroup，并把它们放到新的 `running` 列表中。

scheduler做完上述过程之后，最后会把相应的信息（swap in/out 的 blocks，blocks_to_copy）包装成 SchedulerOutputs 对象供后面 worker 进行 Model Execution（也包含序列本身相关的信息 seq_group_metadata_list）。

总结来说，Scheduler 每次调用 schedule 都会把上个时刻的这三个队列往下一个时刻进行一次更新。按照优先级顺序：首先它优先保证 running 队列继续运行，其次它尝试将 swapped 的任务 swap out 回来，如果没有可用的 swapped 队列则会尝试将 waiting 队列中的任务放到 running。

当 SequenceGroup 推理新增了 token 时，update 接口会更新一遍 SequenceGroup 内的状态。如下图所示，SequenceGroup内包含一组 beam search 的 seq ，每次执行推理的时候，每个 seq 采样 s 次，那么会生成 n*s 个生成的 token，根据这里面token保留置信度 top n 个生成结果。下图中的结果就是 n=4 的情况，可以看到 top n 保留的 output 里 seq1 和 seq3 都来自原始输入 seq1（parent_seq=1），此时需要 BlockSpaceManager 将原始的 seq3 释放（因为保留的output里没有seq3的输出），然后从 seq1 fork 到 seq3，再将新 token 添加到各个 seq 中。

![image-20230919235051666](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919235051666.png)

```python
import enum
import time
from typing import Dict, List, Optional, Tuple

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)


# 这定义了一个新的枚举类PreemptionMode，它继承自Python的内置enum.Enum类。
# PreemptionMode枚举为预占模式提供了一个明确的类型化表示，有两种可选的模式：SWAP和RECOMPUTE。
class PreemptionMode(enum.Enum):
    # 这是一段解释预占模式的文档字符串。它描述了两种不同的预占模式：
    # Swapping: 当序列被抢占时，将其块交换到CPU内存中，并在恢复序列时将它们再次交换回来。
    # Recomputation: 当序列被抢占时，丢弃其块，并在恢复序列时重新计算它们，将序列视为新的提示。
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()

# 这段代码定义了一个名为SchedulerOutputs的类。该类似乎旨在为某种调度操作提供输出或结果。
class SchedulerOutputs:

    def __init__(
        self,
        scheduled_seq_groups: List[SequenceGroup], # 被调度的序列组的列表。
        prompt_run: bool, # 一个布尔值，可能表示是否根据给定的提示执行了某种运行。
        num_batched_tokens: int, # 批处理的token数。
        blocks_to_swap_in: Dict[int, int], 
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        ignored_seq_groups: List[SequenceGroup], # 被忽略的序列组的列表。
    ) -> None:
        self.scheduled_seq_groups = scheduled_seq_groups
        self.prompt_run = prompt_run
        self.num_batched_tokens = num_batched_tokens
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # Swap in and swap out should never happen at the same time.
        assert not (blocks_to_swap_in and blocks_to_swap_out)
        self.ignored_seq_groups = ignored_seq_groups

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_seq_groups and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        # scheduler_config: 调度器的配置，类型为 SchedulerConfig。
        scheduler_config: SchedulerConfig,
        # cache_config: 缓存的配置，类型为 CacheConfig。
        cache_config: CacheConfig,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        # Instantiate the scheduling policy.
        # 使用 PolicyFactory 的 get_policy 方法为调度策略分配一个实例。
        # 这里固定选择了 "fcfs"（可能表示"先来先服务"）策略。
        self.policy = PolicyFactory.get_policy(policy_name="fcfs")
        # Create the block space manager.
        # 创建一个 BlockSpaceManager 的实例，该实例管理数据块的空间。
        # 它使用 cache_config 中的配置数据，包括块大小、GPU块数和CPU块数。
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # Sequence groups in the WAITING state.
        self.waiting: List[SequenceGroup] = []
        # Sequence groups in the RUNNING state.
        self.running: List[SequenceGroup] = []
        # Sequence groups in the SWAPPED state.
        self.swapped: List[SequenceGroup] = []

    # 这个函数是 Scheduler 类的成员函数，用于将新的 SequenceGroup 添加到等待队列中。
    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # Add sequence groups to the waiting queue.
        self.waiting.append(seq_group)

    # 该函数是 Scheduler 类的成员函数，用于根据提供的 request_id 中止一个 SequenceGroup。
    def abort_seq_group(self, request_id: str) -> None:
        # 这是一个外层循环，它遍历三个队列：等待、运行和交换。
        # 这意味着它会检查所有的 SequenceGroup，无论它们处于哪种状态。
        for state_queue in [self.waiting, self.running, self.swapped]:
            # 这是一个内部循环，遍历当前状态队列中的每一个 SequenceGroup
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # Remove the sequence group from the state queue.
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    # 如果三个队列中的任何一个非空，那么这意味着仍有未完成的序列，函数返回True，否则返回False。
    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    # 该方法返回三个队列（waiting, running, swapped）中的SequenceGroup的总数。
    # 它通过取每个队列的长度并将它们加在一起来做到这一点。
    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    # 这个函数是Scheduler类中的一个复杂的私有方法，它尝试安排SequenceGroup实例的执行，
    # 可能需要进行资源的分配、替换和拷贝。函数的主要目的是返回一个SchedulerOutputs对象，
    # 它包含了执行的相关信息。
    def _schedule(self) -> SchedulerOutputs:
        # Blocks that need to be swaped or copied before model execution.
        # 初始化几个字典，用于跟踪需要在模型执行前需要换入，换出，复制的块
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # Fix the current time.
        # 获取当前时间，这可能会被用来决定哪些任务应该被优先调度。
        now = time.time()

        # Join waiting sequences if possible.
        # 检查是否有序列组被交换到CPU。如果没有，尝试合并等待中的序列。
        if not self.swapped:
            ignored_seq_groups: List[SequenceGroup] = []
            scheduled: List[SequenceGroup] = []
            num_batched_tokens = 0
            # Optimization: We do not sort the waiting queue since the preempted
            # sequence groups are added to the front and the new sequence groups
            # are added to the back.
            # 当等待队列不为空时，获取队列中的第一个序列组。
            while self.waiting:
                seq_group = self.waiting[0]
                  
                # 获取当前序列组中的第一个序列的长度（tokens数量）。
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                # 计算允许的最大prompt长度。
                prompt_limit = min(
                    self.scheduler_config.max_model_len,
                    self.scheduler_config.max_num_batched_tokens)
                # 如果当前序列超过了上述限制，发出警告并将该序列组标记为被忽略。
                if num_prompt_tokens > prompt_limit:
                    logger.warning(
                        f"Input prompt ({num_prompt_tokens} tokens) is too long"
                        f" and exceeds limit of {prompt_limit}")
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.FINISHED_IGNORED
                    ignored_seq_groups.append(seq_group)
                    self.waiting.pop(0)
                    break

                # If the sequence group cannot be allocated, stop.
                # 检查是否有足够的块空间来为该序列组分配资源。
                if not self.block_manager.can_allocate(seq_group):
                    break

                # 这里检查已经批处理的token数量（num_batched_tokens）加上当前序列组
                #（seq_group）的token数量（num_prompt_tokens）是否超过了配置中的
                # 最大token限制。如果超过了，循环就会中断。
                if (num_batched_tokens + num_prompt_tokens >
                        self.scheduler_config.max_num_batched_tokens):
                    break

                # 这里获取等待状态下的序列数。
                num_new_seqs = seq_group.num_seqs(
                    status=SequenceStatus.WAITING)
                # 这里计算了当前正在运行状态的所有序列组中的序列数量。
                num_curr_seqs = sum(
                    seq_group.num_seqs(status=SequenceStatus.RUNNING)
                    for seq_group in self.running)
                # 检查当前正在运行的序列数量和新的序列数量的总和是否超过配置中的最大序列限制。
                # 如果超过了，循环就会中断。
                if (num_curr_seqs + num_new_seqs >
                        self.scheduler_config.max_num_seqs):
                    break

                # 从等待队列的前端移除并获取一个序列组。
                seq_group = self.waiting.pop(0)
                # 为从等待队列中获取的序列组分配资源。
                self._allocate(seq_group)
                # 将这个序列组添加到正在运行的队列中。
                self.running.append(seq_group)
                # 更新已批处理的token数量，加上当前序列组的token数量。
                num_batched_tokens += num_prompt_tokens
                # 将这个序列组添加到已调度的队列中。
                scheduled.append(seq_group)

            # 这里检查scheduled列表是否不为空。scheduled列表保存了在当前调度周期中被成功调度的序列组。
            if scheduled:
                # 这行开始创建一个SchedulerOutputs对象，并将以下参数传递给它：
                # 将被成功调度的序列组列表传递给scheduled_seq_groups参数。
                # 这是一个标识符，说明序列组是基于输入提示运行的。
                # 当前已批处理的token数量。
                # 需要从CPU内存中换入的块的映射。
                # 需要换出到CPU内存的块的映射。
                # 需要在GPU内存中复制的块的映射。
                # 由于某种原因（如输入提示太长）而被忽略的序列组列表。
                scheduler_outputs = SchedulerOutputs(
                    scheduled_seq_groups=scheduled,
                    prompt_run=True,
                    num_batched_tokens=num_batched_tokens,
                    blocks_to_swap_in=blocks_to_swap_in,
                    blocks_to_swap_out=blocks_to_swap_out,
                    blocks_to_copy=blocks_to_copy,
                    ignored_seq_groups=ignored_seq_groups,
                )
                return scheduler_outputs

        # 这段代码关注在没有足够的空闲插槽可用以保持所有序列组处于RUNNING状态时的抢占策略。
        # 它包括了哪些序列组应该被抢占，以及如何为当前运行的序列组分配新的token插槽。
        # NOTE(woosuk): Preemption happens only when there is no available slot
        # to keep all the sequence groups in the RUNNING state.
        # In this case, the policy is responsible for deciding which sequence
        # groups to preempt.
        # 这是一个注释，解释了接下来的代码部分。当没有足够的插槽来保持所有序列组处于RUNNING状态时，
        # 就会发生抢占。决定哪个序列组被抢占是由策略决定
        # 这行代码使用策略对象对当前正在运行的序列组列表按优先级进行排序。
        self.running = self.policy.sort_by_priority(now, self.running)

        # Reserve new token slots for the running sequence groups.
        # 这两行代码初始化两个新的列表：running（将要运行的序列组）和preempted（被抢占的序列组）。
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        # 这是一个循环，处理每一个当前正在运行的序列组。每次迭代中，它从self.running列表中取出一个序列组。
        while self.running:
            seq_group = self.running.pop(0)
            # 检查当前序列组是否可以增加新的token插槽。如果不能，进入下面的循环。
            while not self.block_manager.can_append_slot(seq_group):
                # 如果self.running列表仍有序列组，则取出最后一个（优先级最低的）序列组进行抢占。
                if self.running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # 否则，抢占当前的seq_group序列组。
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            # 如果seq_group能够增加新的token插槽，则调用_append_slot方法
            # 为其增加新的插槽，并将其添加到running列表中。
            else:
                # Append new slots to the sequence group.
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        # 在循环结束后，更新self.running为running列表。
        # 这意味着self.running现在包含了所有已成功分配了新插槽的序列组。
        self.running = running

        # Swap in the sequence groups in the SWAPPED state if possible.
        # 这段代码涉及尝试将处于SWAPPED状态的序列组切换回（swap in）为运行状态，如果可能的话。
        # 首先，使用策略对象按优先级对swapped中的序列组进行排序。
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        # 开始一个循环，只要swapped列表不为空，并且没有块要被换出，就继续循环。
        while self.swapped and not blocks_to_swap_out:
            # 获取swapped列表中的第一个序列组。
            seq_group = self.swapped[0]
            # If the sequence group has been preempted in this step, stop.
            # 检查这个序列组是否在这个步骤中被抢占。如果是，就终止循环。
            if seq_group in preempted:
                break
            # If the sequence group cannot be swapped in, stop.
            # 检查是否可以将这个序列组从SWAPPED状态切换回来。如果不可以，就终止循环。
            if not self.block_manager.can_swap_in(seq_group):
                break

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            # 这部分代码确保运行状态的序列总数不超过最大序列数。
            # 它首先计算SWAPPED状态和RUNNING状态的序列数，并检查它们的总和是否超过允许的最大值。
            # 如果超过，就终止循环。
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = sum(
                seq_group.num_seqs(status=SequenceStatus.RUNNING)
                for seq_group in self.running)
            if (num_curr_seqs + num_new_seqs >
                    self.scheduler_config.max_num_seqs):
                break

            # 从swapped列表中移除并获取第一个序列组。
            seq_group = self.swapped.pop(0)
            # 将这个序列组从SWAPPED状态切换回来。
            self._swap_in(seq_group, blocks_to_swap_in)
            # 为这个序列组添加新的插槽。
            self._append_slot(seq_group, blocks_to_copy)
            # 将这个序列组添加到running列表中，意味着现在它正在运行。
            self.running.append(seq_group)

        # 最后，计算RUNNING状态的所有序列的总数。
        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running)

        # 包装成SchedulerOutputs对象返回
        scheduler_outputs = SchedulerOutputs(
            scheduled_seq_groups=self.running,
            prompt_run=False,
            num_batched_tokens=num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=[],
        )
        return scheduler_outputs

    # 这段代码定义了一个名为schedule的方法。这个方法的目的是根据调度器的内部状态
    # 生成一系列SequenceGroupMetadata对象，并将这些对象与调度的输出结果一起返回。
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        # 首先调用_schedule()方法，对序列组进行调度，并将其结果存储在scheduler_outputs变量中。
        # 此方法可能会更改调度器的内部状态，如self.running、self.swapped和self.waiting。
        scheduler_outputs = self._schedule()

        # Create input data structures.
        # 初始化一个列表seq_group_metadata_list来存储计划好的SequenceGroupMetadata。
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        # 开始遍历已计划好的所有序列组。
        for seq_group in scheduler_outputs.scheduled_seq_groups:
            # 为每个序列组初始化两个字典：seq_data（用于存储序列数据）和block_tables（用于存储块表）。
            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            # 遍历序列组中所有处于RUNNING状态的序列。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                # 将当前序列的数据存储在seq_data字典中。
                seq_data[seq_id] = seq.data
                # 使用block_manager为当前序列获取块表，并将其存储在block_tables字典中。
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            # 为当前的序列组创建一个新的SequenceGroupMetadata对象，它包含了该组的所有元数据。
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=scheduler_outputs.prompt_run,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            # 将新创建的SequenceGroupMetadata对象添加到列表seq_group_metadata_list中。
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    # 这段代码定义了一个名为update的函数，用于更新序列组的状态并处理新的序列输出。
    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:
        # 这是一个空列表，稍后将用来存储正在运行且其输出在seq_outputs中的序列组
        scheduled: List[SequenceGroup] = []
        # 这部分代码首先迭代self.running中的所有正在运行的序列组。
        # 对于每一个序列组，它检查该序列组中正在运行的序列是否其输出在seq_outputs中。
        # 如果是，则将该序列组添加到scheduled列表中。
        for seq_group in self.running:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                if seq.seq_id in seq_outputs:
                    scheduled.append(seq_group)
                    break

        # Update the scheduled sequences and free blocks.
        for seq_group in scheduled:
            # Process beam search results before processing the new tokens.
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                # 对于每一个正在运行的序列，它首先检查该序列是否是父序列的一个fork
                #（这是束搜索的一个特性）。如果是，它释放当前序列，并对父序列进行fork。
                if seq.seq_id != output.parent_seq_id:
                    # The sequence is a fork of the parent sequence (beam
                    # search). Free the current sequence.
                    self.block_manager.free(seq)
                    # Fork the parent sequence.
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # Process the new tokens.
            # 对于每一个正在运行的序列，它将新的token追加到该序列中。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # Append a new token to the sequence.
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
        return scheduled

    # free_seq 是一个方法，用于释放与一个给定序列关联的资源，并更新该序列的状态。
    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    # free_finished_seq_groups方法则负责从self.running列表中移除已完成的序列组。
    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    # 这段代码定义了一个名为_allocate的方法。这个方法的主要目的是为一个指定的SequenceGroup分配资源，
    # 并将其中的所有序列的状态设置为RUNNING。
    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    # 这段代码定义了一个名为_append_slot的方法。它的主要功能是为SequenceGroup
    # 中正在运行的序列追加一个资源或内存块，并同时更新一个叫做blocks_to_copy的字典。
    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            # 对每一个正在运行的序列`seq`，调用`self.block_manager`
            # 的`append_slot`方法尝试为其追加一个资源或内存块。
            ret = self.block_manager.append_slot(seq)
            # 返回值`ret`可能是一个包含两个整数的元组，或者为`None`。
            if ret is not None:
                # 如果`ret`不是`None`，则将其解包为两个整数`src_block`和`dst_block`。
                src_block, dst_block = ret
                # 检查`src_block`是否已经在`blocks_to_copy`字典中：
                if src_block in blocks_to_copy:
                    # 如果是，将`dst_block`追加到对应的列表中。
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    # 如果不是，创建一个新的条目，其中`src_block`是键，
                    # 值是一个包含`dst_block`的新列表。
                    blocks_to_copy[src_block] = [dst_block]

    # 这段代码定义了一个名为_preempt的私有方法，这个方法负责预占或
    # 中断SequenceGroup的执行，要么通过重新计算，要么通过交换内存。
    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # If preemption mode is not specified, we determine the mode as follows:
        # We use recomputation by default since it incurs lower overhead than
        # swapping. However, when the sequence group has multiple sequences
        # (e.g., beam search), recomputation is not supported. In such a case,
        # we use swapping instead.
        # FIXME(woosuk): This makes our scheduling policy a bit bizarre.
        # As swapped sequences are prioritized over waiting sequences,
        # sequence groups with multiple sequences are implicitly prioritized
        # over sequence groups with a single sequence.
        # TODO(woosuk): Support recomputation for sequence groups with multiple
        # sequences. This may require a more sophisticated CUDA kernel.
        # 如果调用时没有明确指定预占模式，那么这部分代码会根据SequenceGroup中
        # 运行状态的序列数量来决定使用哪种模式。
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            # 如果有一个正在运行的序列，则默认使用RECOMPUTE模式。
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        # 如果是RECOMPUTE，调用_preempt_by_recompute方法。
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        # 如果是SWAP，调用_preempt_by_swap方法并传入blocks_to_swap_out。
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, "Invalid preemption mode."

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: For FCFS, we insert the preempted sequence group to the front
        # of the waiting queue.
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
```

### BlockSpaceManager

一个 scheduler 还包含一个 BlockSpaceManager，BlockSpaceManager 的功能是管理各个 SequenceGroup 对应 KV Cache 存储信息。每个 Sequence 的 KV Cache 序列会分成多个block_size 长度的 cache block，每个 cache block 的位置信息记录在 BlockSpaceManager。下图中，BlockSpaceManager包含一个 block_tables，其记录 cache block 到 gpu 显存或 cpu 内存物理地址的映射。

SequenceGroup 刚加入到 Scheduler 的时候并没有分配空间，第一次进入 running 状态的时候需要向 BlockSpaceManager 申请可用的 block 空间。如下图所示，BlockSpaceManager 分配 block 空间是以一个 SequenceGroup 作为一组输入，而且默认分配空间的时候，所有 SequenceGroup 内的 token 都是一样的（即是相同的prompt），因此会为所有 Sequence 都指向同一片 cache block 区域，该区域被引用数为 Sequence 数量。

当需要为一个 Sequence 新增 token 时，如下图所示，有两种情况：

- 当前 cache block 空间不足添加新 token，则新增 cache block。
- 当前空间足以添加新 token，但 last block与其他 Sequence 共用时（ref_count>1），如果新 token 还是添加到 last block，那么会与共用 last block 的其他 Sequence 冲突，则需要释放掉 last block（last block 的 ref_count-1），随后给该 Sequence 分配一个新的 last block。最后，返回信息标记原本 last block 内容需要拷贝到这个新的 last block，即 copy-on-write 机制。(拷贝是先在 scheduler 里生成 blocks_to_copy 这个信息，传到 engine 部分由对应的 cuda kernel 真正执行)

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230919235741476.png" alt="image-20230919235741476" style="zoom:50%;" />

![image-20230925092949817](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230925092949817.png)

### Cache Engine

实际上，BlockSpaceManager 只负责维护 cache block 到 gpu/cpu 空间的索引，真正进行换入、换出、拷贝操作都需要通过Worker中 CacheEngine 进行。CacheEngine 是对于 KV Cache 显存实际进行操作的单元。

初始化时候，它先根据之前 profile 的数据（cpu/gpu blocks数）来 allocate cache。然后再给 caching 操作初始化一个 CUDA Stream，以及给每一个 layer 初始化一个 cuda event 来用做 stream synchronization。

在 vLLM 里，每个 key block 的 shape 是 `num_heads, head_size // x, block_size, x` ，其中 x 是 16 // dtype 的大小。也就是说 fp32 时 x=4，fp16 时 x=8。每个 value block 的 shape 是 num_heads, head_size, block_size 。（为什么 key block 要按 x split？在后面的 kernel 实现里会提到这个加速）

cache engine 里支持了其它两个操作：

- copy ：由专门的 cu 函数 `copy_blocks` 支持。
- swap_in 和 swap_out ： in 就是 cpu to gpu，out 就是 gpu to cpu。内部实现由专门的 cu 函数 `swap_blocks` 支持。

下面看相关的 cu 函数，实现在 csrc/cache_kernels.cu 中。

- swap_blocks (src, dst, block_mapping)： 循环一遍 block_mapping，对于每个 [src, dst] pair（block number to block number）做一个单 block 的 copy。支持 GPU to GPU（必须是同一个 GPU 上），GPU to CPU，CPU to GPU。

```c++
void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(
      src_device.index() == dst_device.index(),
      "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  void *src_ptr = src.data_ptr();
  void *dst_ptr = dst.data_ptr();

  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(
      dst_ptr + dst_offset,
      src_ptr + src_offset,
      block_size_in_bytes,
      memcpy_type,
      stream);
  }
}
```

- copy_blocks (key_caches, value_caches, block_mapping)：这里的 mapping 是 `int->list[int]`，也就是一个 src block idx 对应多个 dst block idx。copy 的过程用一个 global kernel 并行实现。

```c++
// Grid: (num_layers, num_pairs)
template<typename scalar_t>
__global__ void copy_blocks_kernel(
  int64_t* key_cache_ptrs,
  int64_t* value_cache_ptrs,
  const int* __restrict__ block_mapping,
  const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache = reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int src_block_number = block_mapping[2 * pair_idx];
  int dst_block_number = block_mapping[2 * pair_idx + 1];

  const int src_block_offset = src_block_number * numel_per_block;
  const int dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int src_offset = src_block_offset + i;
    int dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}
```



- reshape_and_cache (key, value, key_cache, value_cache, slot_mapping)
- gather_cached_kv (key, value, key_cache, value_cache, slot_mapping)



Cache Engine的源代码解析：

```python
KVCache = Tuple[torch.Tensor, torch.Tensor]

# CacheEngine类的主要责任是初始化和管理GPU和CPU上的KV Cache，并为KV Cache 操作如交换和复制提供方法。
class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """
    # 这个构造函数接受三个参数：cache_config、model_config和parallel_config，
    # 分别对应缓存配置、模型配置和并行配置。
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> None:
        # 下面三行代码保存了传入构造函数的配置信息，供类的其他方法使用。
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config

        # 根据模型配置提取了头的大小、层数、头的数量和数据类型，并保存为类的成员变量。
        self.head_size = model_config.get_head_size()
        self.num_layers = model_config.get_num_layers(parallel_config)
        self.num_heads = model_config.get_num_heads(parallel_config)
        self.dtype = model_config.dtype

        # 这里从缓存配置中获取块的大小、GPU上的块数量和CPU上的块数量。
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        # Initialize the cache.
        # 这两行代码调用了这个类的2个成员函数来分配GPU和CPU缓存，并将结果保存为类的成员变量。
        self.gpu_cache = self.allocate_gpu_cache()
        self.cpu_cache = self.allocate_cpu_cache()

        # Initialize the stream for caching operations.
        # 首先创建了一个新的CUDA流并保存。接下来，它使用assert确保新创建的流不是当前的CUDA流。
        self.cache_stream = torch.cuda.Stream()
        assert self.cache_stream != torch.cuda.current_stream()
        # Initialize the events for stream synchronization.
        # 这行代码为每层创建了一个CUDA事件，并保存为一个列表。CUDA事件主要用于同步CUDA流。
        self.events = [torch.cuda.Event() for _ in range(self.num_layers)]
    
    # 这是CacheEngine类的一个成员函数get_key_block_shape，该函数的目的是返回KV Cache中key block的shape（维度）
    def get_key_block_shape(self) -> Tuple[int, int, int, int]:
        # torch.tensor([], dtype=self.dtype)：创建一个空的Tensor，其数据类型由类的dtype属性指定。
        # .element_size()：此方法返回该Tensor的数据类型的元素大小（以字节为单位）。
        # 例如，如果dtype是torch.float32（即32位浮点数），那么element_size将是4（字节）。
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        # 这行代码将16除以前面计算得到的element_size（并执行整数除法），得到的结果赋值给变量x。
        # 假设dtype是torch.float32（元素大小为4字节），那么x将是4。
        x = 16 // element_size
        # 这里构建并返回一个由四个整数构成的元组，这些整数描述了key block的形状。具体来说，形状的每个维度如下：
        # 头的数量（由类的num_heads属性指定）。
        # 头的大小除以x。
        # 块的大小（由类的block_size属性指定）。
        # x。
        return (
            self.num_heads,
            self.head_size // x,
            self.block_size,
            x,
        )
    
    # 返回value block的形状
    def get_value_block_shape(self) -> Tuple[int, int, int]:
        return (
            self.num_heads,
            self.head_size,
            self.block_size,
        )
    
    # 在GPU上申请key_block和value_block的内存
    def allocate_gpu_cache(self) -> List[KVCache]:
        gpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_gpu_blocks, *key_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            value_blocks = torch.empty(
                size=(self.num_gpu_blocks, *value_block_shape),
                dtype=self.dtype,
                device="cuda",
            )
            gpu_cache.append((key_blocks, value_blocks))
        return gpu_cache
    
    # 在CPU上申请key_block和value_block的内存
    def allocate_cpu_cache(self) -> List[KVCache]:
        cpu_cache: List[KVCache] = []
        key_block_shape = self.get_key_block_shape()
        value_block_shape = self.get_value_block_shape()
        pin_memory = not in_wsl()
        if not pin_memory:
            # Pinning memory in WSL is not supported.
            # https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications
            logger.warning("Using 'pin_memory=False' as WSL is detected. "
                           "This may slow down the performance.")
        for _ in range(self.num_layers):
            key_blocks = torch.empty(
                size=(self.num_cpu_blocks, *key_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            value_blocks = torch.empty(
                size=(self.num_cpu_blocks, *value_block_shape),
                dtype=self.dtype,
                pin_memory=pin_memory,
            )
            cpu_cache.append((key_blocks, value_blocks))
        return cpu_cache

    def _swap(
        self,
        src: List[KVCache],
        dst: List[KVCache],
        src_to_dst: Dict[int, int],
    ) -> None:
        with torch.cuda.stream(self.cache_stream):
            for i in range(self.num_layers):
                src_key_cache, src_value_cache = src[i]
                dst_key_cache, dst_value_cache = dst[i]
                # Copy the key blocks.
                cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
                # Copy the value blocks.
                cache_ops.swap_blocks(src_value_cache, dst_value_cache,
                                      src_to_dst)
                event = self.events[i]
                event.record(stream=self.cache_stream)

    # paged attention的swap操作，有点像操作系统里的 swap 概念。in 就是 cpu to gpu，
    # out 就是 gpu to cpu。内部实现由专门的 cu 函数 swap_blocks 支持。 
    def swap_in(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.cpu_cache, self.gpu_cache, src_to_dst)

    def swap_out(self, src_to_dst: Dict[int, int]) -> None:
        self._swap(self.gpu_cache, self.cpu_cache, src_to_dst)

    # paged attention的copy操作，由专门的 cu 函数 copy_blocks 支持。
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None:
        key_caches = [key_cache for key_cache, _ in self.gpu_cache]
        value_caches = [value_cache for _, value_cache in self.gpu_cache]
        # NOTE(woosuk): This operation implicitly synchronizes the CPU and GPU.
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
    
    # 这个函数get_cache_block_size是CacheEngine类的静态方法，用于计算缓存块的大小。
    @staticmethod
    def get_cache_block_size(
        block_size: int,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_heads(parallel_config)
        num_layers = model_config.get_num_layers(parallel_config)

        key_cache_block = block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_layers * (key_cache_block + value_cache_block)
        dtype_size = _get_dtype_size(model_config.dtype)
        return dtype_size * total


def _get_dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()
```



### Worker

Worker 是对单个 GPU 的抽象。

Engine 通过调用 `_run_workers("<method_name>", *args, get_all_outputs, **kwargs) ` 来在 所有 workers 上执行方法。如果 get_all_outputs 设成 True，那么它会将所有 workers 的返回结果包装成 List 来返回。否则，它只会返回第一个 worker 的结果，并且 assert 所有 workers 的输出都是一样的。在实际执行中主要会调用如下方法（方法名, get_all_outputs=False/True）：

- profile_num_avaiable_block，True：通过一次 "试运行" 来 profile peak memory。 每张卡的 blocks 个数可能不同（显存不同），所以需要 get all outputs。由于 vLLM 使用一个中心化的管理单元，因此我们会对 profile 出来的 blocks 个数取 min。
- init_cache_engine，False：初始化 cache engine。
- init_model：对模型进行初始化。
- execute_model：执行模型。



> `llm_engine`通过 `_run_workers("<method_name>", *args, get_all_outputs, **kwargs)` 来和上面的函数建立起联系。从`llm_engine.step`函数我们基本可以看到scheduler，worker的关系：
>
> 
>
> 在 `LLM` 类的 `genarete` 函数中，对于每个输入的prompt，都会给 `llm engine` 生成一个 request 并添加到 scheduler 里。然后调用 `_run_engine` 函数，这个函数的逻辑是对于所有未完成的 requests，就调用 `llm engine` 的 `step` 函数得到这一步的 outputs，然后 append 到返回的 List 里。在`step`函数里，由scheduler获取本次要作为输入的 `seq_group_metadata_list` ，同时产生一个 `scheduler_outputs`。然后 engine 会调用 worker 的 `execute_model` 来执行对 `seq_group_metadata_list` 的模型前向计算。

#### 执行推理(execute_model)

Worker在执行时首先进行两个操作

- 缓存更新：SchedulerOutputs 包含了前面提到的当前所需 swap in/swap out/copy 的 cache block 信息，然后通过CacheEngine 自定义的 op 去执行缓存搬运操作，得到 cuda stream 的 event，后续会在推理 LLM 各层的时候用到。
- 准备输入token序列 __prepare_input：下图右侧的框内是预处理的过程，将 SequenceGroupMetadata 包含 Scehduler 调度得到 running 的所有 SequenceGroup 组合一个 flatten 的 token 序列，作为 LLM 的初始输入。Scheduler 中提到过，running 队列中当前执行的 SequenceGroup 有两类：一类未计算 prompt（前缀）的 KVCache，这部分需要完整的 prompt token 输入去计算各层的 KVCache（全量推理）。另一类已经计算并缓存前缀的 KVCache，因此只需要 last token 作为输入计算下一个 generation token 的分布（增量推理）。如图所示，输入 token 序列的前半部分是多个 prompt 的 token 全量推理序列，后半部分是各个增量推理序列的 last token。此外，全量推理的 SequenceGroup 中多个 Sequence 共享prompt，因此只需要任意一个 Sequence 的 prompt 作用输入就行。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230920090452722.png" alt="image-20230920090452722"  />

随后会执行推理的过程。`__prepare_input`组装的 flatten token 在各层映射成 flatten hidden state 。 除了线性层、激活层等token 独立计算的层以外，attention 层的计算涉及不同 token 的 hidden state 依赖。下图主要展开了 Attention 层的四个主要步骤：

- prompt 全量推理：prompt序列通过集成 xformers 的 attention 算子计算得到下个 layer 的 hidden state 。由于这里attention 计算的输入是完整的 tensor，不是 KVCache 中分散的 cache block，所以可以用第三方的 attention 算子完成计算。
- 等待缓存事件：Cache Engine 中发送了异步缓存操作，因此只有当前层序列的 cache block 完成缓存更新，才能进一步获取KV Cache 或者记录 KV Cache，这种异步的实现能通过 overlap 计算和缓存搬运，节省一部分缓存搬运时间。
- 记录当前 KV Cache（调用 `cache_op.reshape_and_cache`）：当前层输入的 hidden state 作为 KV Cache 通过自定义算子记录到对应 cache block 内，这里记录所有有效 token 的 hidden state，包括 prompt 和 last token（last token是前几次step中新增的，所以也没有缓存hidden state到KV Cache）。
- generation token 增量推理：vLLM 的核心 PageAttention 在此实现，这里通过一个自定义算子，实现了基于 BlockTable 分散 KVCache 的增量 attention计算。

最后 LLM 内的采样器进行采样，将 beam_search 结果（新 token ）返回给 Worker 输出。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230920090400179.png" alt="image-20230920090400179"  />

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

kernel 的 grid 是 `(num_heads, num_seqs)` ，block 大小是 `(NUM_THREADS,)`（这个参数代码里是 128）。每个 warp 是 32 个 threads。也就是说一个 thread 计算的是一个 head，一个 seq 的部分数据。在 warp 之内还有一层抽象，代码里被称作 `thread group`，按代码的意思是一个 warp 里对不同块的切分（也就是 warp_size / block_size）。

```text
WARP_SIZE = 32 (threads)  
THREAD_GROUP_SIZE = max(WARP_SIZE / BLOCK_SIZE, 1) (threads) # warp = thread_group * block
NUM_WARPS = NUM_THREADS / WARP_SIZE
```

这里代码里对 K 和 Q 分别定义了 vec type。注释里说 vec size 的制定规则是让一个 thread group 一次刚好处理 16bytes 的数据。所以一个 vec type 就有 `16 / (thread_group_size * sizeof(scalar_t)` 个 scalar_t 类型。

第一步是把 Q load 到 SM 。在一个 thread group 中，每一个 thread 负责 query 的一部分。例如若 group size 是 4，那么 group 的第一个 thread 就负责 Q 矩阵的第 0, 4, 8, ... 个 vectors。第二个 thread 就负责 Q 矩阵的第 1, 5, 9, ... 个 vectors。

Load Q 之后就是处理 key 和 QK dot 的部分。这里的 x 和之前在 Python 代码定义的 x 是一个意思（`16 / sizeof(scalar_t)`。关于这个 x，在 FasterTransformer 的代码里也可以看到有相关注释：

>The layout of the cache buffer for the keys is [B, H, Dh/x, L, x] where x == 8 for FP16 and x == 4 for FP32 where the fastest moving dimension (contiguous data) is the rightmost one. The values for x are chosen to create chunks of 16 bytes.

为了某些 data locality 而设置的排法。QK 的计算最后归结为一句 `Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs)`，其中 q_vecs 和 k_vecs 每个都有 NUM_VECS_PER_THREAD 个前面定义的 "vecs" 类型。这个 kernel 调用包含了同一个 thread group 内的 reduction。同时在计算 QK dot 时候也顺便算了 qk_max，为 softmax 做准备。

算完 QK dot 后的 qk_max 只是同一个 thread group 内的 reduction。之后还需要做 warp 内的 reduction（这部分通过原语 `__shfl_xor_sync` 来完成）以及 warp 之间的 reduction（这部分通过读写 shared memory 完成）。之后采用相似的方式来完成 logits 和 logits @ trans(V) 的计算。



### 总结

1、记录 cache block 到 gpu 显存或 cpu 内存物理地址的映射减少显存

2、通过 Scheduler 的三个队列的调度增大吞吐

3、PageAttention 的高性能 cuda kernel 实现

4、一些小技巧（copy on write 等等）