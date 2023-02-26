## TVM PackedFunc实现

为了便于Python和C++混合编程，TVM使用了统一的PackedFunc机制。PackedFunc可以将C++中的各类函数打包成统一的函数接口，并自动导出到Python模块中进行调用，并且也支持从Python中注册一个函数，并伪装成PackedFunc在C++和Python中调用。

<img src="https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7%202020-01-10%2010.55.45.png" alt="img" style="zoom:50%;" />

### 预备知识

#### Python ctypes混合编程

ctypes是Python自带的跨语言函数调用库，ctypes提供了简单的C数据类型，可以将C/C++动态库中的函数包装成Python函数进行调用。

- 导出C++函数

  首先在C++中定义一个全局函数，并编译生成C++动态库。

  >1. 编译生成so库
  >
  >g++ src.cpp -fPIC -shared -o libxxx.so  //使用源文件生成so库
  >
  >gcc -shared -fPIC test.o -o libtest.so  //使用目标文件生成so库
  >
  >
  >
  >2. 使用so库
  >
  >gcc -o  main main.c -L. -lmax

  ```c++
  // test.h
  extern "C" {
  int add(int a, int b);
  }
  ```

  ```c++
  // test.cc
  #include "test.h"
  int add(int a, int b) {
    return a + b;
  }
  ```

  用ctypes模块在Python中加载生成的动态库（test.so），并调用C++中的函数。

  ```python
  import ctypes
  
  # Load shared library
  _LIB = ctypes.CDLL("./test.so", ctypes.RTLD_GLOBAL)
  
  a = ctypes.c_int(1)
  b = ctypes.c_int(2)
  # Call C func in Python
  print(_LIB.add(a, b))
  # Or
  print(_LIB.add(1, 2))
  ```

- 传递Python函数到C++

  ctypes也支持将Python函数转换成C类型的函数，并在C/C++中进行调用。

  ```python
  def add(a, b):
    return a + b
  ```

  Python add有两个参数a和b，返回值类型与a和b的类型一致。**在C++中可以为Python add定义一个函数原型 int(int, int)**。

  https://www.geeksforgeeks.org/typedef-in-cpp/

  > 函数原型定义方式
  >
  > Below is the syntax, example, and code to display the usage of typedef with function pointers.
  >
  > **Syntax:**
  >
  > ```
  > typedef <return_type> (*<alias_name>)(<parameter_type>,<parameter_type>,....);
  > ```
  >
  > **Example:**
  >
  > ```c++
  > typedef int (*fun_ptr)(int, int);
  > fun_ptr new_ptr = &function; 
  > ```

  

  ```c++
  extern "C" {
  typedef int (*PyCFunc)(int, int);
  int call_py_func(PyCFunc f, int a, int b);
  }
  ```

  

  ```c++
  #include "test.h"
  int call_py_func(PyCFunc f, int a, int b) {
    return f(a, b);
  }
  ```

  使用ctypes将Python函数转换成C function，传入C++中进行调用。

  ```python
  import ctypes
  
  cfunc = ctypes.CFUNCTYPE(
      ctypes.c_int, # return type
      ctypes.c_int, # arg0 type
      ctypes.c_int  # arg1 type
      )
  
  f = cfunc(add)
  # CFUNCTYPE is callable in Python
  print(f(5, 1))
  
  # Call Python func in C
  print(_LIB.call_py_func(f, 5, 1))
  ```







知乎大佬的详细解释：

## 一、基本原理

TVM使用python的ctypes模块来调用c++代码提供的API，ctypes是python内建的可以用于调用C/C++动态链接库函数的功能模块，ctypes官方文档（[https://docs.python.org/3/library/ctypes.html](https://link.zhihu.com/?target=https%3A//docs.python.org/3/library/ctypes.html)）是这样介绍的：

*ctypes is a foreign function library for Python.It provides C compatible data types, and allows calling functions in DLLs or shared libraries. It can be used to wrap these libraries in pure Python.*

对于动态链接库提供的API，需要使用符合c语言编译和链接约定的API，因为python的ctype只和c兼容，而c++编译器会对函数和变量名进行name mangling，所以需要使用*__cplusplus*宏和*extern "C"*来得到符合c语言编译和链接约定的API，以TVM给python提供的接口为例：

```python
// TVM给python提供的接口主要都在这个文件：
// include/tvm/runtime/c_runtime_api.h，
// 下面主要展示了__cplusplus和extern "C"的用法，
// 以及几个关键的API。
#ifdef __cplusplus
extern "C" {
#endif

int TVMFuncListGlobalNames(...);
int TVMFuncGetGlobal(...);
int TVMFuncCall(...);
    
#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
```

## 二、加载TVM动态库

TVM的python代码从python/tvm/__init__.py中开始真正执行，即：

```python
from ._ffi.base import TVMError, __version__
```

这句简单的import代码，会执行python/tvm/_ffi/__init__.py：

```python
from .base import register_error
from .registry import register_func
from .registry import _init_api, get_global_func
```

上面的第一句，会导致python/tvm/_ffi/base.py中的下面代码被执行：

```python
def _load_lib():
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    return lib, os.path.basename(lib_path[0])

_LIB, _LIB_NAME = _load_lib()
```

上面的lib_path[0]是TVM动态链接库的全路径名称，我是在linux系统做的试验，链接库的名称是/xxx/libtvm.so（不同的系统动态库的名字会有所不同，windows系统是.dll，苹果系统是.dylib，linux系统是.so），在_load_lib函数执行完成后，_LIB和_LIB_NAME都完成了初始化，其中_LIB是一个ctypes.CDLL类型的变量，可以认为它是能够操作TVM动态链接库的export symbols的一个全局句柄，_LIB_NAME是libtvm.so这个字符串。这样后续在python中，我们就能通过_LIB这个桥梁不断的和c++的部分进行交互。

## 三、python怎么关联c++的PackedFunc

在这个系列的[深入理解TVM：Python/C++互调（上）](https://link.zhihu.com/?target=http%3A//mp.weixin.qq.com/s%3F__biz%3DMzg5MzU4NTU5Nw%3D%3D%26mid%3D2247483724%26idx%3D1%26sn%3D964887138b790a5819f04f220d728af9%26chksm%3Dc02dd09ef75a59886863078c6c3cc91e8c0f52b76a9d9bc9dd0b323aa417057ebc4e7eebfe43%26scene%3D21%23wechat_redirect)中，已经对c++中的PackedFunc做了详细的剖析，这里主要来理清楚python的代码中是怎么使用这个核心组件的，还是通过代码，一步步来看。

python中来获取c++API的底层函数是_get_global_func：

```python
# python/tvm/_ffi/_ctypes/packed_func.py
def _get_global_func(func_name):
    handle = ctypes.c_void_p()
    _LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle))
    return _make_packed_func(handle, False)
```

这里面handle是一个相当于void类型的指针变量，因为从ctypes的官方文档中可以查到，c_void_p对应的primitive C compatible data type是：

| ctype type | c type | python type |
| ---------- | ------ | ----------- |
| c_void_p   | void * | int or None |

_get_global_func中调用了TVMFuncGetGlobal这个API，看下这个API的实现就可以发现，handle最终保存了一个c++代码在堆中new出来的PackedFunc对象指针：

```c++
// src/runtime/registry.cc
int TVMFuncGetGlobal(const char* name, TVMFunctionHandle* out) {
  const tvm::runtime::PackedFunc* fp 
      = tvm::runtime::Registry::Get(name);
  *out = new tvm::runtime::PackedFunc(*fp);
}
```

和c++PackedFunc的关联工作这时候才完成一半，在_get_global_func的最后调用了_make_packed_func这个函数：

```python
# python/tvm/_ffi/_ctypes/packed_func.py
def _make_packed_func(handle, is_global):
    obj = PackedFunc.__new__(PackedFuncBase)
    obj.is_global = is_global
    obj.handle = handle
    return obj
```

可以看到_make_packed_func函数中创建了一个定义在python/tvm/runtime/packed_func.py中的python PackedFunc对象，PackedFunc其实是一个空实现，它继承自PackedFuncBase类，PackedFuncBase类中定义了一个__call__函数：

```python
# python/tvm/_ffi/_ctypes/packed_func.py
class PackedFuncBase(object):
  def __call__(self, *args):
    values, tcodes, num_args = _make_tvm_args(args, temp_args)
    ret_val = TVMValue()
    ret_tcode = ctypes.c_int()
    _LIB.TVMFuncCall(
        self.handle,
        values,
        tcodes,
        ctypes.c_int(num_args),
        ctypes.byref(ret_val),
        ctypes.byref(ret_tcode),
    )
    return ret_val
```

从上面可以看出，python的__call__函数调用了C的TVMFuncCall这个API，把前面保存有c++ PackedFunc对象地址的handle以及相关的函数参数传了进去，TVMFuncCall的主体代码如下：

```c++
// src/runtime/c_runtime_api.cc
int TVMFuncCall(TVMFunctionHandle handle, TVMValue* args, ...)
  (*static_cast<const PackedFunc*>(handle))
      .CallPacked(TVMArgs(args, arg_type_codes, num_args), &rv);
}
```

这样就完成了把c++中的PackedFunc映射到了python中的PackedFunc，在python代码中只需要调用python中创建好的PackedFunc对象，就会通过上面分析的过程来一步步调到c++的代码中。

## 四、把注册的函数关联到python各个模块

注册的函数既包括c++中注册的函数，也包括python中注册的函数，其中主要是c++中注册的函数，通过list_global_func_names函数（实际上调用的TVMFuncListGlobalNames这个c++API）可以得到c++中注册的所有函数，目前有1500多个，截图了最开始的十个作为示例给大家看一下：

![img](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/v2-d3e54a52191ed944e6f565c777802415_720w.png)

先看_init_api这个函数，这个函数是把注册函数关联到各个模块的关键：

```python
# python/tvm/_ffi/registry.py
def _init_api(prefix, module_name):
    target_module = sys.modules[module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue
        fname = name[len(prefix) + 1 :]
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = "TVM PackedFunc %s. " % fname
        setattr(target_module, ff.__name__, ff)
```

这里面有三个最主要的点：

- line3：sys.modules是一个全局字典，每当程序员导入新的模块，sys.modules将自动记录该模块。 当第二次再导入该模块时，python会直接到字典中查找，从而加快了程序运行的速度。
- line9：get_global_func等同于上面章节仔细说了的_get_global_func这个函数，这个函数返回一个python端的PackedFunc对象，它的handle成员存储了c++中new出来的PackedFunc对象（以注册函数作为构造参数）的地址，python端的PackedFunc对象的__call__函数调用了c++的TVMFuncCall这个API，handle作为这个API的参数之一，c++端再把handle转成c++的PackedFunc对象来执行，这样就完成了从python端PackedFunc对象的执行到c++端PackedFunc对象的执行的映射。
- line13：把前面代码构造的python端PackedFunc对象作为属性设置到相应的模块上

然后各个模块中对_init_api来全局调用一次，就完成了关联，我在代码中找了几个作为示例，如下所示：

```python
# python/tvm/runtime/_ffi_api.py
tvm._ffi._init_api("runtime", __name__)

# python/tvm/relay/op/op.py
tvm._ffi._init_api("relay.op", __name__)

# python/tvm/relay/backend/_backend.py
tvm._ffi._init_api("relay.backend", __name__)
```

## 五、举一个例子

以TVM中求绝对值的函数abs为例，这个函数实现在tir模块，函数的功能很简单，不会造成额外的理解负担，我们只关注从python调用是怎么映射到c++中的，先看在c++中abs函数的定义和注册：

```c++
// src/tir/op/op.cc
// 函数定义
PrimExpr abs(PrimExpr x, Span span) { ... }

// 函数注册
TVM_REGISTER_GLOBAL("tir.abs").set_body_typed(tvm::abs);
```

再看python端的调用：

```python
# python/tvm/tir/_ffi_api.py
# 把c++ tir中注册的函数以python PackedFunc
# 对象的形式关联到了_ffi_api这个模块
tvm._ffi._init_api("tir", __name__)

# python/tvm/tir/op.py
# 定义了abs的python函数，其实内部调用了前面
# 关联到_ffi_api这个模块的python PackedFunc对象
def abs(x, span=None):
    return _ffi_api.abs(x, span)
```

最后用户可以这样来使用这个函数：

```python
import tvm
from tvm import tir

rlt = tir.abs(-100)
print("abs(-100) = %d" % (rlt)
```

