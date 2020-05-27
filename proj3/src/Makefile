openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lblas

avx.so: dnn_avx.c
	gcc -o $@ $^ -mavx2 -lpthread -lm

all: openblas.so avx.so

clean:
	rm -f openblas.so
	rm -f avx.so
