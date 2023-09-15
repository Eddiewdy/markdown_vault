## LLM.int8()

由于LLM需要大量 GPU 显存才能运行，因此我们需要找到降低资源需求而同时保持模型性能的方法。int8量化不会降低大模型的预测性能，而且可以将大模型的内存占用量减少 2 倍。

### 模型量化的两种方式

- 零点量化 (zero-point quantization) 
  - 第一步：值域映射，即通过缩放将原始的数值范围映射为量化后的数值范围
  - 第二步：零点调整，即通过平移将映射后的数据的最小值对齐为目标值域的最小值
- 最大绝对值 (absolute maximum quantization，absmax) 

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/quant-freeze.png" alt="quant-freeze" style="zoom:50%;" />

### LLM.int8()大模型的零退化矩阵乘法的量化

1. 从输入的隐含状态中，按列提取异常值 (即大于某个阈值的值)。
2. 对 FP16 离群值矩阵和 Int8 非离群值矩阵分别作矩阵乘法。
3. 反量化非离群值的矩阵乘结果并其与离群值矩阵乘结果相加，获得最终的 FP16 结果。



**矩阵乘法的量化**

1. 计算隐含状态后，使用==自定义阈值==提取离群值，并将矩阵分解为两部分
2. 离群值部分使用 FP16 表示，因此它是一个经典的矩阵乘法，8 位矩阵乘法是通过使用向量量化将权重和隐含状态分别量化为 8 位精度 - 即按行量化权重矩阵，并按列量化隐含状态，然后再进行相应向量乘加操作
3. 将量化矩阵乘法反量化至半精度，与第一个矩阵乘法的结果相加得到最终结果。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/Matmul.png" alt="Matmul.png" style="zoom:50%;" />

> OPT-175B模型

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230707151521652.png" alt="image-20230707151521652" style="zoom:50%;" />

### 离群点分布特征

While you have 150,000 outliers per sequence in a 6.7B transformer, ==they only occur in 6 feature dimensions== (6 different indices “i” as in X[:, :, i]).

- 离群点是渐变的，根据困惑度增长

- 异常值特征的数量与困惑度严格成比例。



**不同transfomer层之间的“协调”作用**

transfomer模型的机制：一个stream学习解释输入的特征，另一个stream学习删除其他特征的特征。去除噪声、与上下文无关的特征是做出准确预测的关键。在早期层中删除的噪声、与上下文无关的特征越多，在后面的层中所拥有的高级特征的冲突就越少。transfomer删除特征的方式是，对于特征为度乘一个绝对值很大的正数或者负数，从而产生**离群值**。

在小于6.7B这个规模上，离群值仍然是概率性的。离群值们主要出现在某些维度中，但这些维度在mini-batch之间以及层与层之间可能会略有变化。在这种规模下，各层尚未学会通过同一维度协调离群值。当模型扩大到6.7B以上时，100% 的层对离群值使用相同的维度。

因此作者根据这个观察确定判断离群值的标准：

**特征的大小至少为 6.0，影响至少 25% 的层，并且影响至少 6% 的序列维度。**（the magnitude of the feature is at least 6.0, affects at least 25% of layers, and affects at least 6% of the sequence dimensions.）

**为什么用6.0来区分？**作者的实验，使用混合精度分解，如果我们将任何大小为 6 或更大的特征视为离群特征，则困惑度退化就会停止。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230708122449459.png" alt="image-20230708122449459" style="zoom:50%;" />

### 代码浅析

LLM.int8()定义了一个Linear module叫`Linear8bitLt`，前向的过程中调用`bnb.matmul()`，这个矩阵乘法在/bitsandbytes/bitsandbytes/autograd/_functions.py中定义

```python
def matmul(
    A: tensor,
    B: tensor,
    out: tensor = None,
    state: MatmulLtState = None,
    threshold=0.0,
    bias=None
):
    state = state or MatmulLtState()
    if threshold > 0.0:
        state.threshold = threshold
    return MatMul8bitLt.apply(A, B, out, bias, state)
```

`MatMul8bitLt`是一个继承了torch.autograd.Function的类，通过继承这个类并实现`forward`和`backward`方法来创建自定义的可微函数。`forward`功能如下

1. 对A进行量化、对B进行量化，调用`F.double_quant`，得到量化结果、scale、coo_tensor
2. 对coo_tensor进行处理，调用`F.extract_outliers`提取离群值，得到离群值矩阵`subA`和`subB`
3. 分别做矩阵乘法（int8和float16）
4. 反量化后加上float16的结果，最后返回

继续往下看，`F.double_quant`的核心包括

- get_colrow_absmax()，通过调用`lib.cget_col_row_stats`来得到行列的abs最大值，使用cuda kernel实现
- lib.cdouble_rowcol_quant，底层调用cuda kernel `kDoubleRowColQuant`，传入行列的最大值，得到量化后的结果和RowIdx和ColIdx（离群值的信息）

最后分析一下kDoubleRowColQuant这个kernel，代码位置在/bitsandbytes/csrc/kernels.cu

```c++
template <int THREADS, int ITEMS_PER_THREAD, int TILE_ROWS, int TILE_COLS, int SPARSE_DECOMP> __global__ void kDoubleRowColQuant(half *__restrict__ const A, float *__restrict__ const rowStats, float * __restrict__ const colStats, char *out_col_normed, char *out_row_normed, int *rowidx, int *colidx, half *val, int * __restrict__ nnz_block_ptr, float threshold, int rows, int cols, int tiledCols)
{
  // assumes TILE_SIZE == THREADS*ITEMS_PER_THREAD
  // Each thread reads the same column but multiple rows
  // Rows are loaded in shared memory and access is shared across the threadblock (broadcast)

  // 0. Load row stats data into shared memory; load col stat (1 fixed per thread)
  // 1. Load data row by row (should be at least with TILE_SIZE = 512)
  // 2. quantize data with row/col stats
  // 3. Store data (TILE_SIZE = 512 is a bit slow, but should still be close enough to good performance)

  // each block loads TILE_COLs columns and TILE_ROW rows
  // after reading a tile the row counter increase by TILE_ROWS
  // the col counter reset after reading TILE_COL elements
  const int base_row = ((blockIdx.x*TILE_COLS)/tiledCols)*TILE_ROWS;
  // col increases by TILE_SIZE for each block and wraps back to 0 after tiledCols is reached
  const int base_col = (blockIdx.x*TILE_COLS) % tiledCols;
  const int base_idx = (base_row*cols) + base_col;
  const int items_per_load = ITEMS_PER_THREAD*THREADS;

  typedef cub::BlockLoad<half, THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE> LoadHalf;
  __shared__ typename LoadHalf::TempStorage loadhalf;
  typedef cub::BlockStore<char, THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_VECTORIZE> StoreInt8;
  __shared__ typename StoreInt8::TempStorage storeint8;

  __shared__ float smem_row_stats[TILE_ROWS];
  __shared__ unsigned int smem_nnz_row_idx[TILE_ROWS];

  half local_data[ITEMS_PER_THREAD];
  float local_col_stats[ITEMS_PER_THREAD];
  char local_quantized_data[ITEMS_PER_THREAD];

  // 0. Load row stats data into shared memory; load col stat (1 fixed per thread)
  #pragma unroll ITEMS_PER_THREAD
  for(int j = 0; j < ITEMS_PER_THREAD; j++)
    if(base_col+(threadIdx.x*ITEMS_PER_THREAD) + j < cols)
      local_col_stats[j] = __fdividef(127.0f, colStats[base_col+(threadIdx.x*ITEMS_PER_THREAD)+j]);

  for(int i = threadIdx.x; i < TILE_ROWS; i+=blockDim.x)
  {
    if(base_row + i < rows)
      smem_row_stats[i] = rowStats[base_row+i];

    if(SPARSE_DECOMP)
      smem_nnz_row_idx[i] = nnz_block_ptr[(TILE_ROWS*blockIdx.x) + i];
  }
  __syncthreads();

  // we load row after row from the base_position
  // 1. Load data row by row (should be at least with TILE_SIZE = 512)
  for(int row = 0; row < TILE_ROWS; row++)
  {
    if(base_row + row >= rows){ break; }
    int i = base_idx + (row*cols);
    int valid_items = cols - base_col > items_per_load ? items_per_load : cols - base_col;


    LoadHalf(loadhalf).Load(&(A[i]), local_data, valid_items, 0.0f);
    float row_stat = __fdividef(127.0f, smem_row_stats[row]);

    // 2. quantize data with row/col stats
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      // we already pre-normalized the col/row stat:
      // what this does is float/absmax*127 = int8
      if(SPARSE_DECOMP)
      {
        if(fabsf((float)local_data[j]) >= threshold)
        {
          local_quantized_data[j] = 0;

					int old_idx = atomicInc(&smem_nnz_row_idx[row], UINT_MAX);

          rowidx[old_idx] = base_row+row;
          colidx[old_idx] = base_col+(threadIdx.x*ITEMS_PER_THREAD)+j;
          val[old_idx] = local_data[j];
        }
				else
				{
					local_quantized_data[j] = (char)(rintf(__half2float(local_data[j])*row_stat));
				}
      }
      else
        local_quantized_data[j] = (char)(rintf(__half2float(local_data[j])*row_stat));
    }

    StoreInt8(storeint8).Store(&(out_row_normed[i]), local_quantized_data, valid_items);

    // 2. quantize data with row/col stats
    #pragma unroll ITEMS_PER_THREAD
    for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      // we already pre-normalized the col/row stat:
      // what this does is float/absmax*127 = int8
			local_quantized_data[j] = (char)(rintf(__half2float(local_data[j])*local_col_stats[j]));
    }

    __syncthreads();
    StoreInt8(storeint8).Store(&(out_col_normed[i]), local_quantized_data, valid_items);

  }
}
```

- 基本设置：函数首先确定每个CUDA线程块处理的基本行（base_row）和列（base_col）的索引。每个线程块负责处理一个“tile”，其中包含`TILE_ROWS`行和`TILE_COLS`列（16，64*4）。

- 读取统计信息：每个线程从全局内存中读取与其对应列有关的列统计信息（colStats），并将行统计信息（rowStats）加载到共享内存中。如果SPARSE_DECOMP为真，则同时会加载离群值信息（nnz_block_ptr）到smem_nnz_row_idx中。

- 数据加载与量化：每个线程将负责加载其对应列的多行数据，并进行行归一化，转换为量化数据。如果选择了SPARSE_DECOMP，并且数据的绝对值大于阈值，则离群数据将被设为0，并更新离群值索引。否则，数据将被量化并存储。

```c++
for(int j = 0; j < ITEMS_PER_THREAD; j++)
    {
      if(SPARSE_DECOMP)
      {
        if(fabsf((float)local_data[j]) >= threshold)
        {
          local_quantized_data[j] = 0;
          // 原子操作，将该行的离群值计数器加一，并将原来的值返回
          // 这里的目的是记录这一行中的离群值的索引
		  int old_idx = atomicInc(&smem_nnz_row_idx[row], UINT_MAX);
          rowidx[old_idx] = base_row+row;
          colidx[old_idx] = base_col+(threadIdx.x*ITEMS_PER_THREAD)+j;
          val[old_idx] = local_data[j];
        }
				else
				{
					local_quantized_data[j] = (char)(rintf(__half2float(local_data[j])*row_stat));
				}
      }
      else
        local_quantized_data[j] = (char)(rintf(__half2float(local_data[j])*row_stat));
    }

```



- 数据存储：量化后的数据将被存储回全局内存。





### Demo

```python
import torch
import torch.nn as nn

from bitsandbytes.nn import Linear8bitLt

# Utility function

def get_model_memory_footprint(model):
    r"""
        Partially copied and inspired from: https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    """
    return sum([param.nelement() * param.element_size() for param in model.parameters()])

# Main script

fp16_model = nn.Sequential(
    nn.Linear(64, 64),
    nn.Linear(64, 64)
).to(torch.float16)

# Train and save your model!

torch.save(fp16_model.state_dict(), "model.pt")

# Define your int8 model!

int8_model = nn.Sequential(
    Linear8bitLt(64, 64, has_fp16_weights=False),
    Linear8bitLt(64, 64, has_fp16_weights=False)
)

int8_model.load_state_dict(torch.load("model.pt"))
int8_model = int8_model.to(0) # Quantization happens here

input_ = torch.randn(8, 64, dtype=torch.float16)
hidden_states = int8_model(input_.to(0))

mem_int8 = get_model_memory_footprint(int8_model)
mem_fp16 = get_model_memory_footprint(fp16_model)

print(f"Relative difference: {mem_fp16/mem_int8}")
```

