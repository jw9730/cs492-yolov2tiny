openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lblas

avx.so: dnn_avx.c
	gcc -shared -fPIC -o $@ $^ -mavx2 -lpthread -lm

cublas.so: dnn_cublas.cu
	nvcc --compiler-options -fPIC -shared -o $@ $^ -lcublas

cuda.so: dnn_cuda.cu
	nvcc --compiler-options -fPIC -shared -o $@ $^

all: openblas.so avx.so cublas.so cuda.so

clean:
	rm -f openblas.so
	rm -f avx.so
	rm -f cublas.so
	rm -f cuda.so
