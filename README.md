# Offloaded DNN Inference
- Model: YOLOv2-tiny
<p align="center">
  <img width="300" src="./assets/img_2.png">
</p>

**How to run**
- Locate into proj3/src/
- Set wanted library at line 5 of `yolov2tiny.py` (`from [dnn/dnn_vec/dnn_avx/dnn_cuda] import ...`)
- `make all`
- `python3 __init__.py [in_image] [out_image] [-DEBUG/NDEBUG]`
- Debug mode for correctness check and runtime print: `python3 -d ...`

**Results: Heavy optimization**
- Baseline: Naive scalar operations / Fully vectorized NumPy
- AVX (CPU): Thread-level and instruction-level parallelization
- CUDA (GPU): Dynamic switching between input stationary and weight stationary dataflow
<p align="center">
  <img width="500" src="./assets/img_4.png">
</p>

**Ablation Study**
- Dynamic switching helps!
<p align="center">
  <img width="500" src="./assets/img_5.png">
</p>
<p align="center">
  <img width="300" src="./assets/img.png">
</p>

***Results: Light***
- Baseline: Naive scalar operations
- OpenBLAS / cuBLAS
<p align="center">
  <img width="500" src="./assets/img_3.png">
</p>
