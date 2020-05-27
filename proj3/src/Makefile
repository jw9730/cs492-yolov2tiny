openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -cblas

avx.so: dnn_avx.c
	gcc -o $@ $^ -mavx2 -lpthread

clean:
	rm -f openblas.so
	rm -f avx.so
