# Run object detection faster than numpy
- Model: YOLOv2-tiny

**Baselines**
- Naive scalar operations
- Fully vectorized NumPy

**CPU optimization**
- OpenBLAS
- AVX (main): Runs 1.2x slower to NumPy, thread-level and instruction-level parallelization

**GPU optimization**
- cuBLAS
- CUDA (main): Runs 2.5x faster to NumPy, Dynamic switching between input stationary and weight stationary dataflow

<p align="center">
  <img width="300" src="./img.png">
</p>

**How to run**
- Locate into proj3/src/
- Set wanted library at line 5 of `yolov2tiny.py` (`from [dnn/dnn_vec/dnn_avx/dnn_cuda] import ...`)
- `make all`
- `python3 __init__.py [in_image] [out_image] [-DEBUG/NDEBUG]`
- debug mode for correctness check and runtime print: `python3 -d ...`
