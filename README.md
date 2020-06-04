# Run object detection faster than numpy
- Model: YOLOv2-tiny

Baseline 1: Naive scalar operations
Baseline 2: Full vectorized NumPy

CPU optimization
- OpenBLAS
- AVX (main): Runs comparable to NumPy

GPU optimization
- cuBLAS
- CUDA (main): Runs 2x faster than NumPy
