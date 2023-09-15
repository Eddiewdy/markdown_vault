apple m1 芯片的libzstd库文件的位置在/opt/homebrew/lib

`export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH`

```cmake
cmake -G Unix Makefiles -DCMAKE_BUILD_TYPE=DEBUG -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_OSX_ARCHITECTURES="arm64" ../llvm
```

-DCMAKE_OSX_ARCHITECTURES="arm64" 加上才不会报错，不然默认cmake到x86_64，在链接的时候ld会报错，搞了大半天真的吐了！

同样也可以使用ninja编译系统

```cmake
cmake -G Ninja -DCMAKE_BUILD_TYPE=DEBUG -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_OSX_ARCHITECTURES="arm64" ../llvm
```



llc语法 clang语法也一样

```shell
llc <参数> <文件>
llc -filetype=obj ir
clang ir.o ../rtcalc.c -o expr
./expr 
```



llvm使用load的pass需要开启-enable-new-pm=0

https://groups.google.com/g/llvm-dev/c/kQYV9dCAfSg

>cmu15_745_assignment

```shell
opt -enable-new-pm=0 -load ../build/src/libAssignment1.so -function-info loop.bc -o out
```

