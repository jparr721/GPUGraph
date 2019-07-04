# GPUGraph
Fast graphs for all

## Compiling
### Requirements
- Cuda 10
- ninja
- CMake

### Building
```
$ mkdir build && cd build
$ cmake -GNinja ..
$ ninja
$ ninja install
```
Now link the .so to your executable and BAM, good to go.
