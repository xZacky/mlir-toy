# mlir-toy

This is a mlir toy toturial standalone build project.

[MLIR Toy Tutorial Document](https://mlir.llvm.org/docs/Tutorials/Toy/)

## Install LLVM + MLIR From Source

```
$ git clone https:://github.com/llvm/llvm-project
$ cd llvm-project
$ mkdir build && cd build
$ cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_ASSERTION=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DMLIR_INCLUDE_INTEGRATION_TEST=ON \

$ ninja
$ sudo ninja install
```

## Build Toy Tutorial

```
$ cd mlir-toy
$ mkdir build && cd build
$ cmake -G Ninja ..
$ ninja
```

## Run Example

```
$ ./build/bin/toyc-ch1 test/Ch1/ast.toy -emit=ast
```
