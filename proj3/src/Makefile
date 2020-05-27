openblas.so: dnn_openblas.c
	gcc -O3 -shared -fPIC -o $@ $^ -lblas

avx.so: dnn_avx.c
	gcc -O3 -shared -fPIC -o $@ $^ -mavx2 -lpthread -lm

all: openblas.so avx.so

clean:
	rm -f openblas.so
	rm -f avx.so
