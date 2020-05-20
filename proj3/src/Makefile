openblas.so: dnn_openblas.c
	gcc -shared -fPIC -o $@ $^

clean:
	rm -f openblas.so
	rm -f dnn_openblas.so
