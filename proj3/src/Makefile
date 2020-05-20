openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^ -lblas

clean:
	rm -f openblas.so
