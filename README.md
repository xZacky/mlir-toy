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
If you want to use CUDA backend, you can add option:
```
-DMLIR_ENABLE_CUDA_RUNNER=ON
```

## Build Toy Tutorial

```
$ cd mlir-toy
$ mkdir build && cd build
$ cmake -G Ninja ..
$ ninja
```

## Run example

```
$ cd build
$ ./bin/toyc-ch1 ../test/ast.toy -emit=ast
```