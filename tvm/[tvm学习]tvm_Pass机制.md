

pass的基类`PassInfo`，记录pass的名字，在什么样的优化等级会被enable，以及需要的前置pass

```c++
class PassInfoNode : public Object {
  String name;
  int opt_level;
  Array<String> required;
};
```

#### PassContext

`PassContext` 带有用于优化pass的有用信息。例如，它包含错误报告系统，因此pass的作者可以提供有关优化失败原因的注释。`PassContext` 还旨在替换旧的`BuildConfig`，它用于帮助用户配置编译选项，包括优化级别和必需/禁用的pass等。例如，我们可能有一个配置，它在 `opt_level=3` 时执行所有pass，除开使用 `PassContext` 提供的 `disabled_pass=xx`禁用的一些passes 。现在我们可以在 `opt_level=3` 处对所有passes进行全局处理，并排除禁用pass列表中的那些pass。

此外，用户可以通过 `PassContext::Current()`以线程安全的方式获取某个程序范围内可用的context，因为ThreadLocalStore用于保存创建的pass context对象，关于ThreadLocalStore建议看这篇文章：https://zhuanlan.zhihu.com/p/61587053，TVM模仿Java中的ThreadLocalStore在C++层自己实现了用来管理线程。稍后将提供示例以展示我们如何使用 C++ 和 Python API 来创建使用pass context的编译管道。

```c++
class PassContextNode : public Object {
 public:
  int opt_level{2};
  tvm::Array<tvm::Expr> required_pass;
  tvm::Array<tvm::Expr> disabled_pass;
  mutable Optional<DiagnosticContext> diag_ctx;
  Map<String, ObjectRef> config;
  Array<instrument::PassInstrument> instruments;
};

class PassContext : public NodeRef {
 public:
  TVM_DLL static PassContext Create();
  TVM_DLL static PassContext Current();
  TVM_DLL void InstrumentEnterPassContext();
  TVM_DLL void InstrumentExitPassContext();
  TVM_DLL bool InstrumentBeforePass(const IRModule& mod, const PassInfo& info) const;
  TVM_DLL void InstrumentAfterPass(const IRModule& mod, const PassInfo& info) const;
  /* Other fields are omitted. */

 private:
  // The entry of a pass context scope.
  TVM_DLL void EnterWithScope();
  // The exit of a pass context scope.
  TVM_DLL void ExitWithScope();

  // Classes to get the Python `with` like syntax.
  friend class tvm::With<PassContext>;
};

struct PassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;
  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;
  PassContextThreadLocalEntry() {
    default_context = PassContext(make_node<PassContextNode>());
  }
};

/*! \brief The thread-local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
     PassContextThreadLocalStore;
```

#### Pass Construts

- 是一个基类，包括两个虚函数
- pass实现的方法：在一个特定的context下作用在一个IRModule上
- pass作用的粒度：Module to Module

```c++
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod
                            const PassContext& pass_ctx) const = 0;
};
```

#### Module-Level Passes

Module Level Passes主要用于全局和过程间优化 (IPO)，类似于 LLVM 中使用的module pass。Relay 中一些典型的 pass 需要一个模块的 global picture，比如 A-normal form conversion 和 lambda lifting等，都属于这个集合。在此级别，用户甚至可以在一个module中添加和/或删除function。

```c++
class ModulePassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  // Other members/methods are omitted
};
```

`pass_func` 实现了真正的optimization。例如，我们可能需要对module执行死代码消除。我们可以在 `pass_func` 中实现算法并让它在module上运行。然后它将删除死代码，包括module中未使用的函数。请注意，该字段被设计为一个**PackedFunc**，所以这个优化不仅可以使用C++还可以使用Python来实现。

#### Function-Level Passes

Function-level passes用于为给定的 Relay/tir module实现各种内部函数级优化。它一次从module的函数列表中获取一个函数以进行优化，并生成一个重写的 `Relay Function` 或 `tir PrimFunc`。大多数pass可以归入这一类，例如Relay中的常见子表达式消除和inference simplification 以及tir中的向量化和flattening storage等。

请注意，此级别的passes范围是 Relay Function或 tir PrimFunc。因此，我们无法通过这些passes添加或删除函数，因为它们不知道全局信息。

#### Sequential Passes

`SequentialPass` 类似于 Pytorch `nn.Sequential`，它包含许多用于执行的passes。

```
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // Passes need to be executed.
  Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

目前在Relay中只有少数passes 被放入这组中。例如，`FoldScaleAxis` 需要在内部调度 `ForwardFoldScaleAxis` 和 `BackwardFoldScaleAxis`。此外，建议先完成`BackwardFoldScaleAxis`。因此，该pass是`SequentialPass`的理想候选者。

以下代码显示了如何调用sequential pass中的各个pass。

```c++
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::ir::StringImm>();
      ICHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
}
```

在调用pass时，我们首先检查是否启用了此pass。这是通过首先检查用户是否明确禁用该pass，然后检查它是否被用户指定为必需pass来完成的。如果仍然不确定是否启用了此传递，则将检查其 `opt_level`。只有当它的`opt_level`不低于pass context中配置的优化级别时，才会启用并因此执行此pass。

要执行pass，我们首先需要使用pass name在 TVM packed function注册表中已注册的pass。这是可能的，因为每个pass都注册了一个 API 接口，我们将在后面展示。

```c++
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relay._transform." + pass_name;
  const auto* f = Registry::Get(fpass_name);
  ICHECK(f != nullptr) << "Cannot find " << fpass_name
                      << "to create the pass " << pass_name;
  return (*f)();
}
```

提供了一些helper function来创建上述每种类型的Pass。这些helper function也暴露给 Python 前端，以便用户可以方便地使用 Python API 来创建特定的 pass 对象。

```c++
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreateModulePass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);
```

c++注册pass的方式

首先定义一个返回Pass的c++方法，在使用TVM_REGISTER_GLOBAL将该方法注册到tvm全局，从而可以让python调用

foldConstant为例子

```c++
// tvm/src/relay/transforms/fold_constant.cc
Pass FoldConstant(bool fold_qnn) {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext /* pc */) {
        return Downcast<Function>(FoldConstantExpr(f, m, fold_qnn));
      };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}
```

```c++
// tvm/src/relay/ir/transform.cc
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return FunctionPass(pass_func, pass_info);
}
```



```python
# tvm/python/tvm/relay/transform/transform.py
def FoldConstant(fold_qnn=False):
    return _ffi_api.FoldConstant(fold_qnn)
```

```python
# tvm/python/tvm/relay/transform/_ffi_api.py
import tvm._ffi
tvm._ffi._init_api("relay._transform", __name__)
```

