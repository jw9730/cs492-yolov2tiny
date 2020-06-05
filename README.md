# Offloaded DNN Inference
- Model: YOLOv2-tiny object detection network
<p align="center">
  <img width="150" src="./assets/img_2.png">
</p>

**How to run**
1. Locate into proj3/src/
2. Set optimization at line 5 of `yolov2tiny.py` (`from [dnn/dnn_vec/dnn_avx/dnn_cuda] import ...`)
3. `make all`
4. `python3 __init__.py [in_image] [out_image] [-DEBUG/NDEBUG]`
5. `python3 -d ...`: Debug mode for correctness check and layer-wise runtime display

**Results: Heavy optimization**
- Baseline: Naive nested loops & Fully vectorized NumPy
- AVX (CPU): Thread-level and instruction-level parallelization
- CUDA (GPU): Dynamic switching between input stationary and weight stationary dataflow
<p align="center">
  <img width="400" src="./assets/img_4.png">
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
  <img width="300" src="./assets/img_3.png">
</p>
