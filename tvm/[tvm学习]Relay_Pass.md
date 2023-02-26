```python
target = "llvm"
target_host = "llvm"
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, target_host=target_host, params=params)
```



### relay

这一节主要结合TVM的文档(`https://tvm.apache.org/docs/dev/relay_intro.html`)来介绍一下NNVM的第二代Relay。Relay的设计目标有以下几点：

- 支持传统的数据流(DataFlow)风格编程。
- 支持functional-style scoping，并融合了编程语言领域的一些知识，带了一些新的特性（支持Let表达式，支持递归等等）
- 支持数据流风格和函数式风格混合编程。

![图片](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/640.png)

当DAG中存在共享节点时，这种歧义是由程序语义的解释不同而引起的。Relay IR注意到了这个区别。其实深度学习框架用户经常使用这种方式构建计算图，其中经常发生DAG节点重用。然后当我们以文本格式打印Relay程序时，我们每行打印一个CallNode，并为每个CallNode分配一个临时ID`(%1, %2)`，以便可以在程序的后续部分中引用每个公共节点。

#### Module

Module可以被看作`Map<GlobalVar, Function>`，其中GlobalVar仅仅是一个表示函数名的ID，上面的程序中GlobalVar是`@muladd`和`@myfunc`。当一个CallNode被用来调用另外一个函数时，相应的GlobalVar被存在CallNode的OP中。它包含了一个间接的等级关系---我们需要使用相应的GlobalVar从Module中查找被调用函数的主体。在这种情况下，我们也可以直接将引用的函数存储为CallNode中的OP。那么为什么需要引入GlobalVar呢？主要原因是为了解耦定义和声明，并支持了函数的递归和延迟声明。

#### 0x2.3 Let Binding and Scopes

至此，已经介绍了如何用深度学习框架中的旧方法来构建计算图。这一节将讨论一个Relay的一个新的构造-let bindings。

Let binding被每一种高级的编程语言应用。在Relay中，他是一个拥有三个字段`Let(var, value, body)`的数据结构。当我们计算一个Let表达式时，我们首先计算value部分，然后将其绑定到var，最后在body表达式中返回计算结果。

我们可以使用一系列的Let绑定来构造一个逻辑上等效于数据流程序的程序，下面的代码示例显示了这个用法：

![图片](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/640-20230219203723266.png)Let表达式构造和数据流程序等价的计算图

嵌套的Let Binding被称作A-normal形式，作为函数式编程语言中的常用IR。通过上面的图我们可以发现虽然这两个程序的语义完全等价，它们的文本表示也一样（除了A-norm形式有let的前缀），但AST抽象语法树却不一样。

由于程序的优化使用了这些AST数据结构并对其进行了变换，这两种不同的结构会影响到最终编译器生成的代码。比如，我们想要检测`add(log(x), y)`这个模式。在数据流程序中，我们可以首先进入add节点，然后直接检查它的第一个参数是不是log。而在A-form的程序中，我们不能直接检查任何东西，因为add节点的输入是`%v1`-我们需要维护一个映射表将变量和它绑定的值进行映射，然后查表才知道`%v1`代表的是log。

#### 0x2.4 为什么我们可能需要Let Binding

Let Binding的一种关键用法是它可以指定计算的scope。我们看一下下面这个没有使用Let Binding的例子：

![图片](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/640-20230219203723318.png)没有使用Let Binding编程的一个例子

当我们尝试在该在哪里计算`%1`节点时，问题就来了。特别的是，虽然文本格式似乎建议我们应该在if的scope之外计算节点`%1`，但AST却不建议这样做。实际上数据流图永远不会定义它的计算scope，这在语义上产生了一些歧义。

当我们有闭包时，这种歧义更加有趣，考虑下面的程序，该程序返回一个闭包。我们不知道在哪里计算`%1`，它可以在闭包的内部和外部。

```
fn (%x) {
  %1 = log(%x)
  %2 = fn(%y) {
    add(%y, %1)
  }
  %2
}
```

Let Binding解决了这些问题，因为值的计算发生在let节点上。在这两个程序中，如果将`%1 = log(%x)`改成`let %v1 = log(%x)`，则我们将计算位置明确指定为if scope和闭包之外。可以看到Let Binding为计算端提供了更精确的范围，并且在生成后端代码时会很有用（因为这种范围在IR中）。

另一方面，没有指定计算scope的数据流形式也有其自身的优势，我们不需要担心在生成代码时将let放到哪里。数据流格式还为后面决定将计算节点放到哪里的Passes提供了更大的自由度。因此，在优化的初始阶段如果发现数据流形式还是挺方便的，那么使用数据流图的编码方法可能不是一个坏主意。目前在Relay中也实现了很多针对数据流图的优化方式。

但是，当我们将IR lower到实际的运行时程序时，我们需要精确的计算scope。特别是当我们使用子函数和闭包时，我们要明确指定计算scope应在哪里发生。在后期执行特定的优化中，可以使用Let Binding来解决此问题。

#### 0x2.5 对IR转换的影响

希望到目前为止，你们已经熟悉两种表示形式。大多数函数式编程语言都以A-normal形式进行分析，分析人员无需注意表达式是DAG。

Relay选择同时支持数据流形式和Let Binding。TVM相信让框架开发者选择熟悉的表达形式很重要。但是这确实对我们写通用的Passes产生了一些影响。由于这里还没介绍Passes，以及对Passes理解不深并且我没有使用过Let表达式来构建网络，就不继续介绍具体有哪些影响了。

详细内容可以参考：https://tvm.apache.org/docs/dev/relay_intro.html#let-binding-and-scopes

#### 示例

```python
#coding=utf-8
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor

# 构造BN
def batch_norm(data,
                     gamma=None,
                     beta=None,
                     moving_mean=None,
                     moving_var=None,
                     **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not gamma:
        gamma = relay.var(name + "_gamma")
    if not beta:
        beta = relay.var(name + "_beta")
    if not moving_mean:
        moving_mean = relay.var(name + "_moving_mean")
    if not moving_var:
        moving_var = relay.var(name + "_moving_var")
    return relay.nn.batch_norm(data,
                               gamma=gamma,
                               beta=beta,
                               moving_mean=moving_mean,
                               moving_var=moving_var,
                               **kwargs)[0]

# 构造卷积
def conv2d(data, weight=None, **kwargs):
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    return relay.nn.conv2d(data, weight, **kwargs)


# 构造卷积+BN+ReLU的simpleNet
def simplenet(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
               padding=(1, 1), epsilon=1e-5):
    conv = conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout='NCHW',
        name=name+'_conv')
    bn = batch_norm(data=conv, epsilon=epsilon, name=name + '_bn')
    act = relay.nn.relu(data=bn)
    return act

data_shape = (1, 3, 224, 224)
kernel_shape = (32, 3, 3, 3)
dtype = "float32"
data = relay.var("data", shape=data_shape, dtype=dtype)
act = simplenet(data, "graph", 32, strides=(2, 2))
func = relay.Function(relay.analysis.free_vars(act), act)

print(func)

np_data = np.random.uniform(-1, 1, (1, 3, 224, 224))

params = {
    "graph_conv_weight": tvm.nd.array(np.random.uniform(-1, 1, (32, 3, 3, 3)).astype(dtype)),
    "graph_bn_gamma": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_beta": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_moving_mean": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
    "graph_bn_moving_var": tvm.nd.array(np.random.uniform(-1, 1, (32)).astype(dtype)),
}

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, "llvm", params=params)

dev = tvm.cpu(0)
dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
# set inputs
m.set_input("data", tvm.nd.array(np_data.astype(dtype)))
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)
```

### Pass

```python
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, "llvm", params=params)
```

Pass是TVM中基于Relay IR进行的一系列优化，可以简化计算图，去除冗余的算子，提高模型的推理效率。TVM将所有的pass都抽象到了`tvm/include/tvm/ir/transform.h`这个文件中，主要包含PassContext，PassInfo，Pass，以及Sequential。

这里的PassContext即是上面Python接口对应的C++实现，它包含了Pass执行依赖的一些参数如优化level，依赖的其它特定Pass以及设置不使用某种指定Pass等。PassInfo是用来记录Pass信息的类，包含Pass的opt_level，name，以及当前Pass需要哪些前置Pass。而Pass这个类就执行pass的主体，这是一个基类，每种Pass具体的C++代码实现在`tvm/src/relay/transforms`中，它们都会继承Pass这个基类。最后，Sequential是一个container，装载所有Pass。

需要说明一下，不是所有的Pass都定义在`tvm/src/relay/transforms`这里，比如下面的第一个例子就在`tvm/src/relay/backend/vm`这个文件夹里。接下来我们将几个Pass的例子。

#### RemoveUnusedFunctions

定义在`tvm/src/relay/backend/vm/removed_unused_funcs.cc`，完成一次图的遍历，把没有遍历到的节点删除掉

#### ToBasicBlockNormalForm

定义在`tvm/src/relay/transforms/to_a_normal_form.cc`，通过这个函数，将每个function转换为基本块的形式，主要实现在`ToBasicBlockNormalFormAux`这个函数中，该函数分为以下几个部分：

- 创建一个DependencyGraph，这个数据结构是一个表达式相互依赖的图结构。
- 计算每个节点的scope `std::pair<NodeScopeMap, ExprSet> scopes = CalcScope(dg)`

```c++
std::pair<NodeScopeMap, ExprSet> CalcScope(const DependencyGraph& dg) {
  NodeScopeMap expr_scope;
  ExprSet lifted_exprs;
  std::unordered_map<DependencyGraph::Node*, Expr> node_to_expr;
  // 首先让每个节点都属于一个单独的scope
  for (auto expr_node : dg.expr_node) {
    node_to_expr[expr_node.second] = expr_node.first;
  }
  bool global_scope_used = false;
  Scope global_scope = std::make_shared<ScopeNode>();
  // 使用LCA算法来更新每个节点的真正scope
  for (auto it = dg.post_dfs_order.rbegin(); it != dg.post_dfs_order.rend(); ++it) {
    DependencyGraph::Node* n = *it;
    auto iit = n->parents.head;
    Scope s;
    if (iit == nullptr) {
      ICHECK(!global_scope_used);
      s = global_scope;
      global_scope_used = true;
    } else {
      s = expr_scope.at(iit->value);
      const auto original_s = s;
      iit = iit->next;
      for (; iit != nullptr; iit = iit->next) {
        s = LCA(s, expr_scope.at(iit->value));
      }
      if (s != original_s && node_to_expr.find(n) != node_to_expr.end()) {
        // filter out exprs whose scope do not matter
        Expr expr = node_to_expr[n];
        if (!expr.as<OpNode>()) {
          lifted_exprs.insert(expr);
        }
      }
    }
    if (n->new_scope) {
      auto child_scope = std::make_shared<ScopeNode>(s);
      expr_scope.insert({n, child_scope});
    } else {
      expr_scope.insert({n, s});
    }
  }
  ICHECK(global_scope_used);
  return std::make_pair(expr_scope, lifted_exprs);
}
```

这个函数首先让每个节点都属于一个单独的scope，然后使用LCA算法来更新每个节点的真正scope。这里简单介绍一下LCA算法以及这里具体是如何求取每个节点的scope的。

最近公共祖先简称 LCA（Lowest Common Ancestor）。两个节点的最近公共祖先，就是这两个点的公共祖先里面，离根最远的那个。为了方便，我们记某点集 的最近公共祖先为 或 。LCA有以下性质，引自OI-wiki：

![image-20230219204740049](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20230219204740049.png)

其实不看这个性质也没关系，了解LCA可以求图中两个节点的最近公共祖先即可。然后CalcScope这个函数的具体思路就是先将每个节点初始化为一个单独的scope，然后按照后DFS序遍历这些节点，对于每一个遍历到的节点（这里记作`n`），看一下它的父亲节点`iit`是否存在，如果不存在则说明当前节点是根节点，它的scope应该为`global_scope`。如果`iit`存在，那么遍历`iit`的子节点，看一下这些节点的`scope`的LCA表达式，如果这个通过LCA求出来的表达式和`iit`节点的表达式完全相同，说明这个子图和当前节点是属于同一个scope的，否则就将当前节点插入到`lifted_exprs`，`lifted_exprs`是一个集合用来保存这个DependencyGraph里面的那些跳转指令节点，这也是为什么上面再插入节点到`lifted_exprs`之前需要判断一下这个节点的类型是否为`OpNode`。另外如果当前枚举的节点有new_scope标志，说明当前节点属于一个新的scope，需要为当前节点分配新的类型为`ScopeNode`的一个智能指针。

通过上面的算法，DependencyGraph中的节点和scope节点的关系就被映射到了一个map中，并且scope节点也被建立起了一个树结构。最后调用这个`Fill::ToBasicBlockNormalForm(e, dg, &scopes.first, &scopes.second);`来创建一个`Fill`类，这个类包含了DependencyGraph以及scope相关的信息，通过`ToBasicBlockNormalForm`成员函数实现基本块转换。

#### EliminateCommonSubexpr

最后再看一个消除公共子表达式的Pass，所谓公共子表达式指的就是具有相同的OP类型以及相同的参数，并且参数的顺序都是完全相同的，那么这些表达式就可以合成一个公共子表达式。举个例子：

```
a = b + c``d = b + c
```

**expr_map存储从op算子到对应使用op表达式的映射**

可以看到这两个表达式时完全一致的，那么经过这个Pass之后计算图就会消除其中一个表达式。代码实现在：`tvm/src/relay/transforms/eliminate_common_subexpr.cc`。这里定义了一个`CommonSubexprEliminator`类，这个类重载了两个`Rewrite_`函数来对expr进行遍历和重写。代码实现如下：

```c++
Expr Rewrite_(const CallNode* call, const Expr& post) final {
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    ICHECK(new_call);
    const OpNode* op = new_call->op.as<OpNode>();
    StructuralEqual attrs_equal;

    if (new_call->args.size() == 0 || op == nullptr || op_stateful.get(GetRef<Op>(op), false)) {
      return new_expr;
    }
    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_call->op);
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const CallNode* candidate = candidate_expr.as<CallNode>()) {
          bool is_equivalent = true;
          if (!attrs_equal(new_call->attrs, candidate->attrs)) {
            continue;
          }
          for (size_t i = 0; i < new_call->args.size(); i++) {
            if (!new_call->args[i].same_as(candidate->args[i]) &&
                !IsEqualScalar(new_call->args[i], candidate->args[i])) {
              is_equivalent = false;
              break;
            }
          }
          if (!is_equivalent) continue;
          return GetRef<Call>(candidate);
        }
      }
    }
    expr_map_[new_call->op].push_back(new_expr);
    return new_expr;
  }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) final {
    Expr new_expr = post;
    const TupleGetItemNode* new_tuple_item = new_expr.as<TupleGetItemNode>();
    ICHECK(new_tuple_item);

    if (fskip_ != nullptr && fskip_(new_expr)) {
      return new_expr;
    }

    auto it = expr_map_.find(new_tuple_item->tuple);
    if (it != expr_map_.end()) {
      for (const Expr& candidate_expr : it->second) {
        if (const TupleGetItemNode* candidate = candidate_expr.as<TupleGetItemNode>()) {
          if (new_tuple_item->index == candidate->index) {
            return GetRef<Expr>(candidate);
          }
        }
      }
    }
    expr_map_[new_tuple_item->tuple].push_back(new_expr);
    return new_expr;
  }
```

















